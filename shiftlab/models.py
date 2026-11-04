"""Los cinco candidatos fijos; pesos explícitos en escalador y estimador."""

import time
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .data import make_splits
from .protocol import CANDIDATES
from .transforms import array_digest, training_batch


def fit_candidate(candidate_id, data, train_indices):
    if candidate_id not in CANDIDATES:
        raise ValueError("Candidato desconocido.")
    if not set(train_indices) <= set(make_splits(data)["train"]["indices"]):
        raise ValueError("El ajuste solo admite imágenes de la partición de entrenamiento.")
    spec = CANDIDATES[candidate_id]
    batch = training_batch(data, train_indices, spec["augmentation"])
    if spec["family"] == "dummy":
        model = DummyClassifier(strategy="most_frequent")
    else:
        estimator = (LogisticRegression(C=1, solver="lbfgs", max_iter=2000) if spec["family"] == "logistic"
                     else SVC(kernel="rbf", C=1, gamma="scale", probability=False))
        model = Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        if spec["family"] == "dummy":
            model.fit(batch["X"], batch["y"], sample_weight=batch["weights"])
        else:
            model.fit(batch["X"], batch["y"], scaler__sample_weight=batch["weights"], classifier__sample_weight=batch["weights"])
    elapsed = time.perf_counter() - started
    warning_records = [{"category": type(item.message).__name__, "message": str(item.message)} for item in captured]
    for item in captured:
        warnings.warn(str(item.message), item.category, stacklevel=2)
    estimator = model if spec["family"] == "dummy" else model.named_steps["classifier"]
    scope = {"n_rows": len(batch["X"]), "X_sha256": array_digest(batch["X"]), "y_sha256": array_digest(batch["y"]),
             "weights_sha256": array_digest(batch["weights"]), "original_indices": batch["original_indices"].tolist(),
             "weight_sum": float(batch["weights"].sum())}
    audit = {"estimator": scope.copy()}
    if spec["family"] != "dummy":
        scaler = model.named_steps["scaler"]
        audit["scaler"] = scope.copy()
        audit["scaler"]["mean"] = scaler.mean_.tolist()
        audit["scaler"]["variance"] = scaler.var_.tolist()
        audit["scaler"]["n_samples_seen"] = float(scaler.n_samples_seen_)
        # El estimador recibe la salida ya escalada, no X cruda.
        audit["estimator"]["X_sha256"] = array_digest(scaler.transform(batch["X"]))
    record = {"label": spec["label"], "family": spec["family"], "augmentation": spec["augmentation"],
              "model_params": estimator.get_params(deep=False),
              "training": {"original_indices": batch["original_indices"].tolist(), "row_ids": batch["row_ids"].tolist(),
                           "versions": batch["versions"].tolist(), "weights": batch["weights"].tolist(),
                           "fit_audit": audit, "warnings": warning_records, "seconds": elapsed}}
    return model, record
