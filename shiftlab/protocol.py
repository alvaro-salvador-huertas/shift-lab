"""Decisiones fijadas antes del primer ajuste. No se buscan hiperparámetros."""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn


SPLIT_SEED = 20261029
AUGMENTATION_SEED = 20261030
EVALUATION_SEED = 20261031
SCENARIOS = {
    "clean": {"label": "Limpio", "kind": "clean", "level": 0.0},
    "noise_0025": {"label": "Ruido · σ 0,025", "kind": "noise", "level": 0.025},
    "noise_005": {"label": "Ruido · σ 0,05", "kind": "noise", "level": 0.05},
    "noise_010": {"label": "Ruido · σ 0,10", "kind": "noise", "level": 0.10},
    "blur_025": {"label": "Desenfoque · σ 0,25 px", "kind": "blur", "level": 0.25},
    "blur_045": {"label": "Desenfoque · σ 0,45 px", "kind": "blur", "level": 0.45},
    "blur_065": {"label": "Desenfoque · σ 0,65 px", "kind": "blur", "level": 0.65},
    "contrast_090": {"label": "Contraste · c 0,90", "kind": "contrast", "level": 0.90},
    "contrast_070": {"label": "Contraste · c 0,70", "kind": "contrast", "level": 0.70},
    "contrast_050": {"label": "Contraste · c 0,50", "kind": "contrast", "level": 0.50},
}
CANDIDATES = {
    "baseline": {"label": "Baseline · clase mayoritaria", "family": "dummy", "augmentation": False},
    "logistic_clean": {"label": "Logística · train limpio", "family": "logistic", "augmentation": False},
    "logistic_augmented": {"label": "Logística · train aumentado", "family": "logistic", "augmentation": True},
    "svm_clean": {"label": "SVM RBF · train limpio", "family": "svm", "augmentation": False},
    "svm_augmented": {"label": "SVM RBF · train aumentado", "family": "svm", "augmentation": True},
}
PROTOCOL = {
    "version": "shiftlab-v1", "question": "¿Cómo cambia el rendimiento al degradar las imágenes y al entrenar con aumentación?",
    "data": "sklearn.datasets.load_digits; 1797 imágenes reales de 8×8; intensidades /16.",
    "split": {"method": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True, "seed": SPLIT_SEED,
              "test_fold": 0, "validation_fold": 1, "train_folds": [2, 3, 4],
              "groups": "SHA256 de los 64 píxeles originales uint8, en orden C; no de la etiqueta."},
    "models": {"dummy": {"strategy": "most_frequent"},
               "logistic": {"scaler": "StandardScaler", "C": 1.0, "solver": "lbfgs", "max_iter": 2000},
               "svm": {"scaler": "StandardScaler", "kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": False}},
    "augmentation": {"seed": AUGMENTATION_SEED, "versions": ["clean", "noise_005", "blur_045", "contrast_070"],
                     "weight_each": 0.25, "total_weight_per_original": 1.0, "split": "train exclusivamente"},
    "evaluation": {"seed": EVALUATION_SEED, "scenarios": SCENARIOS,
                   "noise": "SeedSequence([stream_seed, original_row_id]); un patrón normal base por imagen, compartido entre intensidades.",
                   "blur": "scipy.ndimage.gaussian_filter sigma=(0,sigma,sigma), mode=reflect, truncate=4.0; nunca filtra entre imágenes.",
                   "contrast": "c*x+(1-c)*0.5", "clip": "Ruido recortado a [0,1]"},
    "metrics": ["accuracy", "macro_f1"], "class_labels": list(range(10)), "zero_division": 0,
    "comparisons": "Por candidato, diferencias de métricas sobre las mismas imágenes entre limpio y cada degradación; cada original es una unidad, sus versiones no son observaciones independientes.",
    "development": "Preparación y cuadrícula solo train; coste y desarrollo solo train/validación. Cinco candidatos fijos, sin búsqueda.",
    "freeze": "Configuración, dataset, particiones, versiones y hashes de código experimental antes de evaluar test. Medida de reproducibilidad, no preregistración independiente.",
    "budget_seconds": 600, "numeric_threads": 1,
    "limits": ["Degradaciones sintéticas sobre imágenes reales pequeñas.", "Sin identificadores de escritor: no evalúa escritores nuevos.",
               "No demuestra robustez en producción.", "No se exige monotonía ni ventaja de la aumentación."]}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def code_identity():
    root = Path(__file__).parent
    paths = list(root.glob("*.py")) + [root.parent / "requirements.txt", root.parent / "pyproject.toml", root.parent / "docs" / "protocolo.md"]
    files = {str(path.relative_to(root.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    return {"files": files, "digest": digest_json(files), "scope": "Todos los módulos del motor shiftlab; interfaz y documentos no alteran el entrenamiento."}


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
