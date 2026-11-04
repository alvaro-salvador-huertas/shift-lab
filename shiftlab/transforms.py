"""Degradaciones puras y streams separados por imagen original."""

import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter

from .protocol import AUGMENTATION_SEED, EVALUATION_SEED, SCENARIOS


def array_digest(array):
    value = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def apply_transform(images, scenario_id, row_ids, stream="evaluation"):
    images = np.asarray(images, dtype=np.float64)
    row_ids = np.asarray(row_ids)
    if scenario_id not in SCENARIOS or stream not in ("evaluation", "augmentation"):
        raise ValueError("Transformación o stream desconocido.")
    if (images.ndim != 3 or images.shape[1:] != (8, 8) or len(images) != len(row_ids)
            or not np.isfinite(images).all() or (images < 0).any() or (images > 1).any()):
        raise ValueError("Imágenes normalizadas finitas (n,8,8) e IDs alineados requeridos.")
    if row_ids.dtype.kind not in "iu" or (row_ids < 0).any():
        raise ValueError("Los IDs de imagen deben ser enteros no negativos.")
    spec = SCENARIOS[scenario_id]
    if spec["kind"] == "clean":
        return images.copy()
    if spec["kind"] == "noise":
        seed = EVALUATION_SEED if stream == "evaluation" else AUGMENTATION_SEED
        base = np.array([np.random.default_rng(np.random.SeedSequence([seed, int(row)])).normal(size=(8, 8)) for row in row_ids])
        return np.clip(images + spec["level"] * base, 0.0, 1.0)
    if spec["kind"] == "blur":
        return gaussian_filter(images, sigma=(0, spec["level"], spec["level"]), mode="reflect", truncate=4.0)
    return spec["level"] * images + (1 - spec["level"]) * 0.5


def evaluation_images(indices, scenario_id, dataset=None):
    from .data import load_dataset
    data = load_dataset() if dataset is None else dataset
    rows = np.asarray(indices, dtype=int)
    if rows.ndim != 1 or (rows < 0).any() or (rows >= len(data.y)).any():
        raise ValueError("Índices de evaluación inválidos.")
    return apply_transform(data.images[rows], scenario_id, rows, stream="evaluation")


def training_batch(data, train_indices, augmented):
    rows = np.asarray(train_indices, dtype=int)
    if rows.ndim != 1 or len(set(rows.tolist())) != len(rows) or (rows < 0).any() or (rows >= len(data.y)).any():
        raise ValueError("Índices de entrenamiento inválidos.")
    versions = ["clean", "noise_005", "blur_045", "contrast_070"] if augmented else ["clean"]
    transformed = [apply_transform(data.images[rows], version, rows, "augmentation") for version in versions]
    images = np.stack(transformed, axis=1).reshape(-1, 8, 8)
    mapping = np.repeat(rows, len(versions))
    kinds = np.tile(np.array(versions), len(rows))
    weights = np.full(len(mapping), 1.0 / len(versions))
    return {"X": images.reshape(-1, 64), "y": data.y[mapping], "weights": weights,
            "row_ids": mapping, "versions": kinds, "original_indices": rows}
