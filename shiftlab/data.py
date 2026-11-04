"""Dataset local, duplicados exactos y particiones agrupadas reproducibles."""

from dataclasses import dataclass
import hashlib

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedGroupKFold

from .protocol import SPLIT_SEED, digest_json


@dataclass
class Dataset:
    images: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    metadata: dict

    @property
    def X(self):
        return self.images.reshape(len(self.images), 64)


def image_groups(original_images):
    original = np.asarray(original_images)
    if original.ndim != 3 or original.shape[1:] != (8, 8) or not np.isfinite(original).all():
        raise ValueError("Se requieren imágenes finitas de forma (n,8,8).")
    if (original < 0).any() or (original > 16).any() or not np.array_equal(original, original.astype(np.uint8)):
        raise ValueError("Los píxeles originales deben ser enteros entre 0 y 16.")
    return np.array([hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes(order="C")).hexdigest() for image in original])


def duplicate_report(groups, y):
    records = []
    for group in sorted(set(groups)):
        rows = np.flatnonzero(groups == group)
        if len(rows) > 1:
            labels = sorted(set(int(value) for value in np.asarray(y)[rows]))
            records.append({"group": str(group), "indices": rows.tolist(), "labels": labels, "contradictory_labels": len(labels) > 1})
    return {"duplicate_groups": records, "duplicate_group_count": len(records),
            "duplicate_extra_rows": sum(len(item["indices"]) - 1 for item in records),
            "contradictory_groups": [item for item in records if item["contradictory_labels"]],
            "policy": "Agrupar por imagen, conservar todas las observaciones y etiquetas; no eliminar selectivamente."}


def load_dataset():
    digits = load_digits()
    original = np.asarray(digits.images, dtype=np.uint8)
    y = np.asarray(digits.target, dtype=np.int64)
    groups = image_groups(original)
    digest = hashlib.sha256(original.tobytes(order="C") + y.astype(np.uint8).tobytes()).hexdigest()
    metadata = {"name": "sklearn.datasets.load_digits", "n_samples": len(y), "shape": [8, 8], "n_classes": 10,
                "normalization": "uint8 original /16.0; sin parámetros aprendidos", "sha256": digest,
                "source": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
                "loader": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
                "citation": "Alpaydin, E. y Kaynak, C. (1998). Optical Recognition of Handwritten Digits. DOI:10.24432/C50P49.",
                "license": "CC BY 4.0 según UCI; conservar atribución e indicar transformaciones.",
                "source_subset": "1797 filas del test original UCI, redistribuidas con scikit-learn. Las nuevas particiones son internas a ese subconjunto.",
                "writer_limit": "No se dispone de IDs de escritor utilizables; no se evalúa generalización a escritores nuevos.",
                "duplicates": duplicate_report(groups, y)}
    return Dataset(original.astype(np.float64) / 16.0, y, groups, metadata)


def make_splits(dataset=None):
    data = load_dataset() if dataset is None else dataset
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    folds = np.full(len(data.y), -1, dtype=int)
    for fold, (_, indices) in enumerate(splitter.split(data.X, data.y, data.groups)):
        folds[indices] = fold
    result = {"method": "StratifiedGroupKFold", "seed": SPLIT_SEED, "n_folds": 5, "fold_assignment": folds.tolist()}
    for name, membership in (("train", [2, 3, 4]), ("validation", [1]), ("test", [0])):
        indices = np.flatnonzero(np.isin(folds, membership))
        result[name] = {"indices": indices.tolist(), "size": len(indices),
                        "groups": sorted(set(data.groups[indices])),
                        "class_distribution": {str(label): int((data.y[indices] == label).sum()) for label in range(10)}}
    result["digest"] = digest_json(result)
    validate_splits(data, result)
    return result


def validate_splits(data, splits):
    all_rows, all_groups = set(), set()
    for name in ("train", "validation", "test"):
        rows = splits[name]["indices"]
        groups = set(data.groups[rows])
        if len(rows) != len(set(rows)) or set(rows) & all_rows or groups & all_groups:
            raise ValueError("Particiones con filas o grupos compartidos.")
        if splits[name]["size"] != len(rows) or splits[name]["groups"] != sorted(groups):
            raise ValueError("Metadatos de partición incompatibles.")
        all_rows.update(rows)
        all_groups.update(groups)
    if all_rows != set(range(len(data.y))):
        raise ValueError("Las particiones no cubren el dataset completo.")
