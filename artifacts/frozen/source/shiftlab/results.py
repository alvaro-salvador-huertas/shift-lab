"""Resultados archivados: validación, exportación y reconstrucción sin entrenar."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import load_dataset, make_splits
from .protocol import CANDIDATES, PROTOCOL, SCENARIOS, code_identity, digest_json, versions
from .transforms import array_digest, evaluation_images, training_batch


def metrics_from_predictions(y_true, y_pred):
    y, pred = np.asarray(y_true), np.asarray(y_pred)
    if y.ndim != 1 or not len(y) or y.shape != pred.shape or not np.isin(y, range(10)).all() or not np.isin(pred, range(10)).all():
        raise ValueError("Etiquetas/predicciones inválidas: deben estar alineadas y pertenecer a 0…9.")
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=list(range(10)), zero_division=0)
    return {"metrics": {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, labels=list(range(10)), average="macro", zero_division=0))},
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(10))).tolist(),
            "per_class": [{"class": label, "precision": float(precision[label]), "recall": float(recall[label]),
                           "f1": float(f1[label]), "support": int(support[label])} for label in range(10)]}


def paired_difference(y_true, clean_pred, shifted_pred):
    y, clean, shifted = np.asarray(y_true), np.asarray(clean_pred), np.asarray(shifted_pred)
    clean_ok, shifted_ok = clean == y, shifted == y
    clean_metrics = metrics_from_predictions(y, clean)["metrics"]
    shifted_metrics = metrics_from_predictions(y, shifted)["metrics"]
    return {"unit": "same_original_image", "accuracy_difference": shifted_metrics["accuracy"] - clean_metrics["accuracy"],
            "macro_f1_difference": shifted_metrics["macro_f1"] - clean_metrics["macro_f1"],
            "correct_to_wrong": int((clean_ok & ~shifted_ok).sum()), "wrong_to_correct": int((~clean_ok & shifted_ok).sum()),
            "both_correct": int((clean_ok & shifted_ok).sum()), "both_wrong": int((~clean_ok & ~shifted_ok).sum()),
            "per_image_correct_change": (shifted_ok.astype(int) - clean_ok.astype(int)).tolist()}


def validate_results(payload, strict_code=True):
    try:
        data, splits = load_dataset(), make_splits()
        if payload["schema_version"] != "shiftlab-results/1" or payload["stage"] not in ("validation", "final"):
            raise ValueError("Formato de resultados incompatible.")
        if payload["protocol"] != PROTOCOL or payload["splits"] != splits:
            raise ValueError("El protocolo o las particiones no corresponden a esta ejecución.")
        provenance = payload["provenance"]
        if (provenance["dataset"] != data.metadata or provenance["split_digest"] != splits["digest"]
                or provenance["versions"] != versions()):
            raise ValueError("Dataset, particiones o versiones incompatibles.")
        if strict_code and provenance["code_digest"] != code_identity()["digest"]:
            raise ValueError("El código experimental cambió respecto al resultado archivado.")
        split = "test" if payload["stage"] == "final" else "validation"
        rows = splits[split]["indices"]
        y = data.y[rows].tolist()
        if payload["evaluation"]["split"] != split or payload["evaluation"]["row_ids"] != rows or payload["evaluation"]["y_true"] != y:
            raise ValueError("Las etiquetas o filas no corresponden al conjunto evaluado.")
        if set(payload["candidates"]) != set(CANDIDATES):
            raise ValueError("Faltan candidatos del protocolo.")
        image_digests = {key: {"sha256": array_digest(evaluation_images(rows, key, data)), "shape": [len(rows), 8, 8]}
                         for key in SCENARIOS}
        if payload["evaluation"]["scenario_inputs"] != image_digests:
            raise ValueError("Las imágenes evaluadas no corresponden a las degradaciones declaradas.")
        for candidate_id, candidate in payload["candidates"].items():
            if set(candidate["scenarios"]) != set(SCENARIOS):
                raise ValueError("Faltan escenarios de evaluación.")
            spec, training = CANDIDATES[candidate_id], candidate["training"]
            expected = training_batch(data, splits["train"]["indices"], spec["augmentation"])
            for key in ("original_indices", "row_ids", "versions", "weights"):
                if training[key] != expected[key].tolist():
                    raise ValueError("Aumentación, pesos o procedencia de entrenamiento incompatibles.")
            for stage in training["fit_audit"].values():
                if stage["original_indices"] != splits["train"]["indices"] or stage["n_rows"] != len(expected["X"]):
                    raise ValueError("Una etapa se ajustó fuera del entrenamiento declarado.")
                if stage["weights_sha256"] != array_digest(expected["weights"]) or stage["y_sha256"] != array_digest(expected["y"]):
                    raise ValueError("Pesos o etiquetas de ajuste incompatibles.")
            clean = candidate["scenarios"]["clean"]["y_pred"]
            for scenario in candidate["scenarios"].values():
                if scenario["row_ids"] != rows or scenario["y_true"] != y:
                    raise ValueError("Evaluaciones no emparejadas sobre las mismas imágenes.")
                recomputed = metrics_from_predictions(y, scenario["y_pred"])
                for key in recomputed:
                    if recomputed[key] != scenario[key]:
                        raise ValueError("Las métricas, clases o matriz no corresponden a las predicciones.")
                if paired_difference(y, clean, scenario["y_pred"]) != scenario["paired_vs_clean"]:
                    raise ValueError("Diferencias emparejadas incoherentes.")
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("El archivo de resultados está incompleto o mal formado.") from exc


def load_results(path="artifacts/final/results.json"):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("No se encuentran resultados válidos. Ejecuta la CLI y carga el informe generado.") from exc
    validate_results(payload)
    return payload


def export_results(payload):
    validate_results(payload)
    return json.dumps(deepcopy(payload), ensure_ascii=False, indent=2, allow_nan=False)


def images_for_result(payload, scenario_id):
    if scenario_id not in SCENARIOS:
        raise ValueError("Escenario desconocido.")
    data = load_dataset()
    if data.metadata["sha256"] != payload["provenance"]["dataset"]["sha256"]:
        raise ValueError("El dataset no corresponde al resultado archivado.")
    images = evaluation_images(payload["evaluation"]["row_ids"], scenario_id, data)
    if array_digest(images) != payload["evaluation"]["scenario_inputs"][scenario_id]["sha256"]:
        raise ValueError("Las imágenes reconstruidas no coinciden con las evaluadas.")
    return images


def image_pair(payload, scenario_id, row_id):
    rows = payload["evaluation"]["row_ids"]
    if row_id not in rows:
        raise ValueError("La imagen no pertenece al conjunto evaluado.")
    offset = rows.index(row_id)
    original = load_dataset().images[row_id].copy()
    return original, images_for_result(payload, scenario_id)[offset].copy()
