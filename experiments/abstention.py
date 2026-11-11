"""Abstención exploratoria sobre validación; nunca evalúa el test final.

Desde la raíz: .venv/bin/python -m experiments.abstention --output CARPETA_NUEVA
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shlex

import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from shiftlab.data import load_dataset, make_splits
from shiftlab.models import fit_candidate
from shiftlab.protocol import PROTOCOL, SCENARIOS, code_identity, digest_json, versions
from shiftlab.transforms import array_digest, evaluation_images


SUBSPLIT_SEED = 20261111
MIN_COVERAGE = 0.95
CONDITIONS = ("clean", "noise_005", "noise_010")
EXPLORATORY_NOTE = (
    "Análisis exploratorio: la validación original ya se usó en el desarrollo "
    "de ShiftLab. Esta subdivisión posterior no aporta confirmación independiente."
)
SCORE_NOTE = (
    "max(predict_proba) del Pipeline de logística limpia es una puntuación de "
    "selección, no una probabilidad calibrada de acertar."
)


def split_validation(data, validation_rows):
    """Divide originales una sola vez; sus versiones heredan la asignación."""
    rows = np.asarray(validation_rows)
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or len(rows) < 2
            or len(set(rows.tolist())) != len(rows)
            or (rows < 0).any() or (rows >= len(data.y)).any()):
        raise ValueError("Se requieren índices originales únicos y válidos.")
    rows = np.sort(rows)
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SUBSPLIT_SEED)
    # Fold 0 selecciona el umbral, fold 1 mide: nunca se elige por métricas.
    evaluation_offsets, selection_offsets = next(
        splitter.split(data.X[rows], data.y[rows], data.groups[rows])
    )
    result = {
        "method": "StratifiedGroupKFold", "n_splits": 2,
        "shuffle": True, "seed": SUBSPLIT_SEED,
        "selection_fold": 0, "evaluation_fold": 1,
        "versions_policy": "Partición de originales por hash; todas sus versiones heredan el subconjunto.",
    }
    groups_seen = set()
    for name, offsets in (("selection", selection_offsets), ("evaluation", evaluation_offsets)):
        chosen = rows[offsets]
        groups = set(data.groups[chosen])
        if not len(chosen) or groups_seen & groups:
            raise ValueError("Los subconjuntos deben ser no vacíos y separar grupos exactos.")
        groups_seen.update(groups)
        result[name] = {
            "row_ids": chosen.tolist(), "total": len(chosen), "groups": sorted(groups),
            "class_distribution": {str(c): int((data.y[chosen] == c).sum()) for c in range(10)},
        }
    return result


def _scores(values):
    scores = np.asarray(values, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()):
        raise ValueError("Se requieren puntuaciones finitas no vacías entre 0 y 1.")
    return scores


def choose_threshold(scores, min_coverage=MIN_COVERAGE):
    """Mayor t con mean(score >= t) >= cobertura; empates siempre aceptados.

    Es la k-ésima puntuación descendente, con k=ceil(cobertura*n).
    Fraction evita redondeos al calcular ese mínimo entero.
    Solo recibe puntuaciones limpias, nunca etiquetas ni errores.
    """
    scores = _scores(scores)
    if not 0 < min_coverage <= 1:
        raise ValueError("La cobertura objetivo debe estar en (0,1].")
    required = math.ceil(Fraction(str(min_coverage)) * len(scores))
    return float(np.sort(scores)[-required])


def policy_counts(y_true, y_pred, scores, threshold):
    scores = _scores(scores)
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if truth.shape != scores.shape or prediction.shape != scores.shape or not math.isfinite(threshold):
        raise ValueError("Etiquetas, predicciones y puntuaciones deben estar alineadas; umbral finito.")
    accepted = scores >= threshold
    wrong = truth != prediction
    total, count = len(scores), int(accepted.sum())
    errors_accepted = int((wrong & accepted).sum())
    return {
        "total": total, "accepted": count, "rejected": total - count,
        "errors_accepted": errors_accepted,
        "errors_rejected": int((wrong & ~accepted).sum()),
        "coverage": count / total,
        "accepted_error_rate": errors_accepted / count if count else None,
    }


def risk_coverage_curve(y_true, y_pred, scores):
    """Todos los cortes empíricos, sin dividir empates; descripción, no ajuste."""
    scores = _scores(scores)
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    errors = np.cumsum((np.asarray(y_true) != np.asarray(y_pred))[order])
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(scores) - 1]
    return [{"threshold": float(np.nextafter(float(ordered[0]), np.inf)),
             "coverage": 0.0, "accepted_error_rate": None}] + [
        {"threshold": float(ordered[end]), "coverage": int(end + 1) / len(scores),
         "accepted_error_rate": int(errors[end]) / int(end + 1)}
        for end in ends
    ]


def _predict(model, data, rows, condition):
    images = evaluation_images(rows, condition, data)
    X = images.reshape(len(images), 64)
    with threadpool_limits(limits=PROTOCOL["numeric_threads"]):
        probabilities = model.predict_proba(X)
        predictions = model.predict(X)
    return predictions, probabilities.max(axis=1), array_digest(images)


def run_experiment(prepared_path=Path("artifacts/prepared.json")):
    """Un ajuste sobre todo train, selección limpia y después medición emparejada."""
    prepared_path = Path(prepared_path)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    data = load_dataset()
    splits = make_splits(data)
    identity = {
        "protocol_digest": digest_json(PROTOCOL), "dataset_digest": data.metadata["sha256"],
        "split_digest": splits["digest"], "code_identity": code_identity(), "versions": versions(),
    }
    if (prepared["schema_version"] != "shiftlab-prepared/1"
            or prepared["identity"] != identity or prepared["splits"] != splits
            or prepared["protocol"] != PROTOCOL):
        raise ValueError("La preparación original no coincide con datos, código, configuración o versiones.")
    halves = split_validation(data, splits["validation"]["indices"])
    selection_rows = halves["selection"]["row_ids"]
    evaluation_rows = halves["evaluation"]["row_ids"]
    model, training = fit_candidate("logistic_clean", data, splits["train"]["indices"])
    _, selection_scores, selection_hash = _predict(model, data, selection_rows, "clean")
    # El umbral queda fijado antes de transformar o predecir la segunda mitad.
    threshold = choose_threshold(selection_scores)
    selection_accepted = int((selection_scores >= threshold).sum())
    payload = {
        "schema_version": "shiftlab-abstention/1", "analysis": "exploratory",
        "limitations": [EXPLORATORY_NOTE, SCORE_NOTE,
                        "El 95 % es una restricción empírica en selección limpia, no una garantía en medición o ruido.",
                        "Degradaciones sintéticas; versiones emparejadas, no observaciones independientes."],
        "provenance": {
            "original_identity": identity, "dataset": data.metadata,
            "prepared_path": str(prepared_path),
            "prepared_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
            "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numeric_threads": PROTOCOL["numeric_threads"],
            "final_test_evaluated": False,
        },
        "model": training, "original_train_row_ids": splits["train"]["indices"],
        "original_validation_row_ids": splits["validation"]["indices"],
        "validation_subsets": halves,
        "selection": {
            "condition": "clean", "score": "max(predict_proba)", "acceptance_rule": "score >= threshold",
            "target_coverage": MIN_COVERAGE, "threshold": threshold,
            "rule": "Mayor umbral que conserva al menos ceil(0.95*n) imágenes; se aceptan todos los empates.",
            "row_ids": selection_rows, "scores": selection_scores.tolist(),
            "input_sha256": selection_hash, "total": len(selection_rows),
            "accepted": selection_accepted, "coverage": selection_accepted / len(selection_rows),
            "labels_or_errors_used_for_threshold": False,
        },
        "evaluation": {
            "row_ids": evaluation_rows, "y_true": data.y[evaluation_rows].tolist(),
            "fixed_threshold": threshold, "conditions": {},
            "curve_note": "Curvas descriptivas en la segunda mitad. No se selecciona otro umbral con ellas; destacar el fijado en selección.",
        },
    }
    for condition in CONDITIONS:
        predictions, scores, input_hash = _predict(model, data, evaluation_rows, condition)
        payload["evaluation"]["conditions"][condition] = {
            "label": SCENARIOS[condition]["label"], "transform": SCENARIOS[condition],
            "input_sha256": input_hash, "row_ids": evaluation_rows,
            "y_pred": predictions.tolist(), "scores": scores.tolist(),
            "accepted_mask": (scores >= threshold).tolist(),
            "mandatory": policy_counts(data.y[evaluation_rows], predictions, scores, 0.0),
            "abstention": policy_counts(data.y[evaluation_rows], predictions, scores, threshold),
            "descriptive_curve": risk_coverage_curve(data.y[evaluation_rows], predictions, scores),
        }
    return payload


def coverage_figure(payload):
    figure = go.Figure()
    colors = ("#176a54", "#2b62b6", "#b74c1f")
    for (condition, result), color in zip(payload["evaluation"]["conditions"].items(), colors):
        curve = result["descriptive_curve"]
        label = result["label"]
        figure.add_trace(go.Scatter(
            x=[p["coverage"] for p in curve], y=[p["accepted_error_rate"] for p in curve],
            customdata=[p["threshold"] for p in curve], mode="lines", connectgaps=False,
            name=f"{label} · descriptiva", legendgroup=condition, line={"color": color},
            hovertemplate="Cobertura %{x:.2%}<br>Error aceptadas %{y:.2%}<br>Corte descriptivo %{customdata:.8f}<extra>%{fullData.name}</extra>",
        ))
        for policy, symbol, name in (("mandatory", "circle-open", "obligatoria"),
                                     ("abstention", "star", "umbral fijado")):
            counts = result[policy]
            figure.add_trace(go.Scatter(
                x=[counts["coverage"]], y=[counts["accepted_error_rate"]], mode="markers",
                name=f"{label} · {name}", legendgroup=condition,
                marker={"color": color, "size": 13, "symbol": symbol},
                hovertemplate="Cobertura %{x:.2%}<br>Error aceptadas %{y:.2%}<extra>%{fullData.name}</extra>",
            ))
    threshold = payload["selection"]["threshold"]
    figure.update_layout(
        template="plotly_white", height=650,
        title={"text": f"Abstención exploratoria · segunda mitad de validación<br><sup>★ Umbral fijado {threshold:.8f} · ○ Clasificación obligatoria · líneas descriptivas</sup>"},
        xaxis={"title": "Cobertura (aceptadas / total)", "tickformat": ".0%", "range": [0, 1.02]},
        yaxis={"title": "Tasa de error entre aceptadas", "tickformat": ".1%", "rangemode": "tozero"},
        legend={"orientation": "h", "y": -0.2}, margin={"b": 170},
        annotations=[{"text": "Validación reutilizada durante el desarrollo. max(predict_proba) no está calibrado.<br>Con cero aceptadas, la tasa queda sin definir y no se dibuja.",
                      "xref": "paper", "yref": "paper", "x": 0, "y": -0.44,
                      "showarrow": False, "xanchor": "left"}],
    )
    return figure


def _percentage(value):
    return "sin definir" if value is None else f"{value:.2%}"


def brief_report(payload, command):
    selection = payload["selection"]
    halves = payload["validation_subsets"]
    lines = [
        "Abstención en ShiftLab — análisis exploratorio", "", EXPLORATORY_NOTE, SCORE_NOTE, "",
        f"Comando desde la raíz del proyecto: {command}",
        "Para repetir, usa otra carpeta de salida: se rechaza cualquier destino existente.", "",
        "Modelo original: StandardScaler + LogisticRegression(C=1, solver=lbfgs, max_iter=2000).",
        f"Un solo ajuste, solo con las {len(payload['original_train_row_ids'])} imágenes limpias del train original; pesos 1 y un hilo numérico.",
        f"Validación original: {len(payload['original_validation_row_ids'])} imágenes. StratifiedGroupKFold de dos folds, shuffle=True, semilla {SUBSPLIT_SEED}.",
        f"Selección: {halves['selection']['total']}; medición: {halves['evaluation']['total']}. Grupos SHA256 disjuntos; versiones en el subconjunto de su original.",
        "Estratificación aproximada bajo restricción de grupos; distribuciones de clase e índices completos en el JSON.", "",
        f"UMBRAL FIJADO: {selection['threshold']:.17g}; aceptar si score >= umbral, incluidos empates.",
        f"Elegido solo con puntuaciones limpias de selección, sin usar etiquetas ni errores: {selection['accepted']}/{selection['total']} aceptadas ({_percentage(selection['coverage'])}).",
        "Es el mayor corte con cobertura empírica >=95 % en esa mitad; no garantiza esa cobertura en la otra mitad ni con ruido.",
        "Se fijó antes de transformar o predecir la segunda mitad; no se reajusta por condición.", "",
        "Resultados sobre las mismas imágenes de medición (errores de las rechazadas = los que cometería el clasificador):",
        "Condición | Política | Total | Aceptadas | Rechazadas | Errores aceptadas | Errores rechazadas | Cobertura | Error aceptadas",
    ]
    for result in payload["evaluation"]["conditions"].values():
        for key, label in (("mandatory", "Obligatoria"), ("abstention", "Umbral fijado")):
            c = result[key]
            lines.append(f"{result['label']} | {label} | {c['total']} | {c['accepted']} | {c['rejected']} | {c['errors_accepted']} | {c['errors_rejected']} | {_percentage(c['coverage'])} | {_percentage(c['accepted_error_rate'])}")
    lines.extend(["", "Resultado del umbral fijado:"])
    for result in payload["evaluation"]["conditions"].values():
        baseline, selective = result["mandatory"], result["abstention"]
        if selective["accepted_error_rate"] is None:
            change = "tasa sin definir: ninguna imagen aceptada"
        else:
            delta = 100 * (selective["accepted_error_rate"] - baseline["accepted_error_rate"])
            change = f"cambio de tasa de error {delta:+.2f} puntos porcentuales"
        lines.append(f"- {result['label']}: {_percentage(selective['coverage'])} de cobertura; {change}; {selective['errors_rejected']} errores entre {selective['rejected']} rechazadas.")
    lines.extend([
        "", "Se muestran todas las condiciones, aunque la abstención ayude poco o empeore la tasa.",
        "Las líneas de cobertura-error son descriptivas de medición; sus estrellas marcan el umbral ya fijado y los círculos la clasificación obligatoria.",
        "Sin aceptadas, accepted_error_rate es null en JSON. Las versiones no son muestras independientes; no se estiman garantías poblacionales.",
        "No se ha leído ningún resultado del test final, ni se ha predicho o ajustado sobre test. No se modifican históricos, protocolo ni interfaz.",
        "Archivos: results.json, cobertura-error.html (autónomo, sin CDN), informe.txt.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/prepared.json"))
    parser.add_argument("--output", type=Path, required=True, help="Carpeta nueva; se rechaza si ya existe.")
    args = parser.parse_args(argv)
    # Fallar antes de entrenar o leer preparación si el destino ya existe.
    if args.output.exists():
        parser.error("La carpeta de salida ya existe. Se conserva; usa una carpeta nueva.")
    command = shlex.join([".venv/bin/python", "-m", "experiments.abstention", "--prepared", str(args.prepared), "--output", str(args.output)])
    try:
        payload = run_experiment(args.prepared)
        figure = coverage_figure(payload)
        report = brief_report(payload, command)
        # La carpeta se reclama de forma exclusiva antes de escribir los tres archivos.
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "results.json").open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        with (args.output / "cobertura-error.html").open("x", encoding="utf-8") as handle:
            handle.write(figure.to_html(full_html=True, include_plotlyjs=True, config={"displaylogo": False}))
        with (args.output / "informe.txt").open("x", encoding="utf-8") as handle:
            handle.write(report)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
