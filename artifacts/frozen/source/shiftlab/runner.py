"""Acciones explícitas del estudio. Test solo se evalúa después de congelar."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from threadpoolctl import threadpool_limits

from .data import load_dataset, make_splits
from .models import fit_candidate
from .protocol import CANDIDATES, PROTOCOL, SCENARIOS, code_identity, digest_json, versions
from .results import metrics_from_predictions, paired_difference, validate_results
from .transforms import array_digest, evaluation_images


def _now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # La apertura exclusiva impide sobreescribir resultados incluso si dos
    # comandos se lanzan accidentalmente a la vez.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _identity(data, splits):
    return {"protocol_digest": digest_json(PROTOCOL), "dataset_digest": data.metadata["sha256"],
            "split_digest": splits["digest"], "code_identity": code_identity(), "versions": versions()}


def training_grid(data, splits, output):
    rows = [next(row for row in splits["train"]["indices"] if data.y[row] == label) for label in range(10)]
    cell, left, header = 96, 105, 80
    canvas = Image.new("RGB", (left + cell * len(SCENARIOS), header + cell * len(rows)), "#111923")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 10), "SHIFT LAB / SOLO TRAIN · misma imagen en diez condiciones", fill="#f4f7fa", font=font)
    for column, scenario in enumerate(SCENARIOS):
        draw.text((left + column * cell, 45), scenario, fill="#c8e2ed", font=font)
        images = evaluation_images(rows, scenario, data)
        for line, (row, image) in enumerate(zip(rows, images)):
            if column == 0:
                draw.text((8, header + line * cell + 22), f"dígito {data.y[row]}\nfila {row}", fill="#f4f7fa", font=font)
            tile = Image.fromarray(np.rint(image * 255).astype(np.uint8)).resize((80, 80), resample=Image.Resampling.NEAREST).convert("RGB")
            canvas.paste(tile, (left + column * cell, header + line * cell))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Se conserva la cuadrícula existente: {output}")
    canvas.save(output)
    return {"path": str(output), "row_ids": rows, "split": "train", "stream": "evaluation",
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def prepare(output_root="artifacts"):
    root = Path(output_root)
    if (root / "prepared.json").exists():
        raise FileExistsError("Ya existe la preparación. Se conserva; usa otra carpeta para otra ejecución.")
    data, splits = load_dataset(), make_splits()
    grid = training_grid(data, splits, root / "training-transformations.png")
    value = {"schema_version": "shiftlab-prepared/1", "created_at": _now(), "protocol": deepcopy(PROTOCOL),
             "identity": _identity(data, splits), "dataset": data.metadata, "splits": splits, "training_grid": grid}
    write_new(root / "prepared.json", value)
    return value


def _prepared(root):
    try:
        value = json.loads((Path(root) / "prepared.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Ejecuta prepare antes de entrenar.") from exc
    data, splits = load_dataset(), make_splits()
    if value["identity"] != _identity(data, splits):
        raise ValueError("El código, entorno o protocolo cambió tras prepare. Conserva esa carpeta y prepara otra.")
    return value, data, splits


def cost(output_root="artifacts"):
    """Coste acotado: cinco ajustes completos, solo predicción limpia de validación."""
    prepared, data, splits = _prepared(output_root)
    started, entries = time.perf_counter(), []
    for key in CANDIDATES:
        iteration = time.perf_counter()
        model, record = fit_candidate(key, data, splits["train"]["indices"])
        with threadpool_limits(limits=1):
            pred = model.predict(data.X[splits["validation"]["indices"]])
        entries.append({"candidate": key, "seconds": time.perf_counter() - iteration,
                        "warnings": record["training"]["warnings"], "prediction_count": len(pred), "split": "validation"})
    elapsed = time.perf_counter() - started
    # Cinco ajustes ya medidos. Se presupuesta ×20 por las predicciones y margen.
    estimated = elapsed * 20
    value = {"schema_version": "shiftlab-cost/1", "created_at": _now(), "identity": prepared["identity"],
             "entries": entries, "elapsed_seconds": elapsed, "conservative_estimate_seconds": estimated,
             "budget_seconds": 600, "within_budget": estimated < 600,
             "note": "Sin predicciones ni métricas de test; no se seleccionan configuraciones por la puntuación."}
    write_new(Path(output_root) / "cost.json", value)
    return value


def _evaluate(stage, output_root, frozen=None):
    prepared, data, splits = _prepared(output_root)
    split = "test" if stage == "final" else "validation"
    rows = splits[split]["indices"]
    started = time.perf_counter()
    deadline = started + PROTOCOL["budget_seconds"]
    # Se materializan una sola vez y se comparten por los cinco candidatos.
    inputs = {key: evaluation_images(rows, key, data) for key in SCENARIOS}
    archive = Path(output_root) / stage / "evaluation-images.npz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as handle:
        np.savez_compressed(handle, row_ids=np.asarray(rows, dtype=np.int64), y_true=data.y[rows], **inputs)
    payload = {"schema_version": "shiftlab-results/1", "created_at": _now(), "stage": stage,
               "protocol": deepcopy(PROTOCOL), "splits": splits,
               "provenance": {"dataset": data.metadata, "split_digest": splits["digest"],
                              "code_digest": code_identity()["digest"], "versions": versions(),
                              "freeze_digest": digest_json(frozen) if frozen is not None else None,
                              "training_grid_sha256": prepared["training_grid"]["sha256"]},
               "evaluation": {"split": split, "row_ids": rows, "y_true": data.y[rows].tolist(),
                              "archive": {"path": str(archive), "format": "npz", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
                              "scenario_inputs": {key: {"sha256": array_digest(value), "shape": list(value.shape)} for key, value in inputs.items()}},
               "candidates": {}, "budget_seconds": 600}
    for key in CANDIDATES:
        if time.perf_counter() > deadline:
            raise TimeoutError("Se ha alcanzado el límite de 600 s antes de un ajuste.")
        model, candidate = fit_candidate(key, data, splits["train"]["indices"])
        candidate["scenarios"] = {}
        for scenario_id, images in inputs.items():
            if time.perf_counter() > deadline:
                raise TimeoutError("Se ha alcanzado el límite de 600 s durante la evaluación.")
            with threadpool_limits(limits=1):
                pred = model.predict(images.reshape(len(images), 64)).astype(int).tolist()
            clean_pred = pred if scenario_id == "clean" else candidate["scenarios"]["clean"]["y_pred"]
            candidate["scenarios"][scenario_id] = {"row_ids": rows, "y_true": data.y[rows].tolist(), "y_pred": pred,
                                                    **metrics_from_predictions(data.y[rows], pred),
                                                    "paired_vs_clean": paired_difference(data.y[rows], clean_pred, pred)}
        payload["candidates"][key] = candidate
    payload["elapsed_seconds"] = time.perf_counter() - started
    validate_results(payload)
    write_new(Path(output_root) / stage / "results.json", payload)
    return payload


def validation(output_root="artifacts"):
    prepared, _, _ = _prepared(output_root)
    cost_value = json.loads((Path(output_root) / "cost.json").read_text(encoding="utf-8"))
    if cost_value["identity"] != prepared["identity"] or not cost_value["within_budget"]:
        raise ValueError("Ejecuta y revisa una comprobación de coste compatible antes de validar.")
    return _evaluate("validation", output_root)


def freeze(output_root="artifacts", visual_reviewed=False, review_note=""):
    if visual_reviewed is not True or not isinstance(review_note, str) or not review_note.strip():
        raise ValueError("La congelación requiere inspección real de la cuadrícula de train y una nota explícita.")
    root = Path(output_root)
    prepared, data, splits = _prepared(root)
    validation_path = root / "validation" / "results.json"
    validation_value = json.loads(validation_path.read_text(encoding="utf-8"))
    validate_results(validation_value)
    if validation_value["stage"] != "validation":
        raise ValueError("Se requiere un resultado de validación, no test, antes de congelar.")
    snapshot = root / "frozen" / "source"
    if snapshot.exists() or (root / "frozen" / "config.json").exists():
        raise FileExistsError("Ya existe una congelación; se conserva íntegra.")
    code = code_identity()
    source_root = Path(__file__).resolve().parents[1]
    for name, digest in code["files"].items():
        source = source_root / name
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError("El código cambió durante la copia congelada.")
    value = {"schema_version": "shiftlab-frozen/1", "created_at": _now(), "identity": _identity(data, splits),
             "protocol": deepcopy(PROTOCOL), "dataset": data.metadata, "splits": splits,
             "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
             "visual_review": {"reviewed": True, "note": review_note, "training_grid": prepared["training_grid"]},
             "source_snapshot": str(snapshot),
             "note": "Congelación local después de desarrollar con validación; no es una preregistración independiente."}
    write_new(root / "frozen" / "config.json", value)
    return value


def final(output_root="artifacts"):
    root = Path(output_root)
    frozen = json.loads((root / "frozen" / "config.json").read_text(encoding="utf-8"))
    prepared, data, splits = _prepared(root)
    if frozen["identity"] != _identity(data, splits) or not frozen["visual_review"]["reviewed"]:
        raise ValueError("La congelación no corresponde al código, datos o particiones actuales.")
    for name, digest in frozen["identity"]["code_identity"]["files"].items():
        if hashlib.sha256((Path(frozen["source_snapshot"]) / name).read_bytes()).hexdigest() != digest:
            raise ValueError("La copia congelada no coincide con sus hashes.")
    return _evaluate("final", root, frozen)
