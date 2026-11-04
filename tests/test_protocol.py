"""Guardas de protocolo y artefactos; no reentrena el estudio en cada prueba."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import numpy as np

from shiftlab.protocol import CANDIDATES, PROTOCOL, SCENARIOS, code_identity
from shiftlab.results import export_results, image_pair, load_results, validate_results
from shiftlab.runner import freeze, write_new


def test_fixed_protocol_has_five_candidates_ten_scenarios_and_separate_streams():
    assert len(CANDIDATES) == 5
    assert len(SCENARIOS) == 10
    assert PROTOCOL["split"]["seed"] != PROTOCOL["augmentation"]["seed"] != PROTOCOL["evaluation"]["seed"]
    assert PROTOCOL["budget_seconds"] == 600
    assert PROTOCOL["augmentation"]["weight_each"] == 0.25
    assert "requirements.txt" in code_identity()["files"]


def test_exclusive_write_preserves_existing_artifacts(tmp_path):
    path = tmp_path / "data.json"
    write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        write_new(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


def test_freeze_requires_explicit_visual_review_before_reading_test(tmp_path):
    with pytest.raises(ValueError, match="inspección"):
        freeze(tmp_path)


@pytest.fixture(scope="module")
def archived_results():
    # Validación archivada, sin ajustar modelos en la suite. Antes de su
    # creación estas pruebas quedan explícitamente pendientes, no simuladas.
    path = Path(__file__).resolve().parents[1] / "artifacts" / "validation" / "results.json"
    if not path.exists():
        pytest.skip("Aún no se ha ejecutado la validación explícita de CLI")
    return load_results(path)


def test_archived_predictions_export_and_reconstructed_images_agree(archived_results):
    payload = archived_results
    assert json.loads(export_results(payload)) == payload
    row = payload["evaluation"]["row_ids"][0]
    original, shifted = image_pair(payload, "noise_005", row)
    assert original.shape == shifted.shape == (8, 8)
    assert len(payload["candidates"]) * len(SCENARIOS) == 50


def test_changed_prediction_and_stale_split_are_detected(archived_results):
    bad = deepcopy(archived_results)
    # Asegurar un cambio efectivo, independientemente de la predicción original.
    prior = archived_results["candidates"]["logistic_clean"]["scenarios"]["clean"]["y_pred"][0]
    bad["candidates"]["logistic_clean"]["scenarios"]["clean"]["y_pred"][0] = (prior + 1) % 10
    with pytest.raises(ValueError):
        validate_results(bad)
    bad = deepcopy(archived_results)
    bad["evaluation"]["row_ids"] = list(reversed(bad["evaluation"]["row_ids"]))
    with pytest.raises(ValueError):
        validate_results(bad)


def test_all_candidates_receive_identical_evaluation_tensors_without_training(archived_results, tmp_path, monkeypatch):
    """50 llamadas predict de fixture; ningún ajuste ni artefacto de producción."""
    import shiftlab.runner as runner

    source_root = Path(__file__).resolve().parents[1]
    prepared = json.loads((source_root / "artifacts" / "prepared.json").read_text())
    (tmp_path / "prepared.json").write_text(json.dumps(prepared), encoding="utf-8")
    captured = {}
    fit_calls = []

    class PredictionRecorder:
        def __init__(self, candidate):
            self.candidate = candidate

        def predict(self, X):
            captured.setdefault(self.candidate, []).append(np.asarray(X).copy())
            return np.zeros(len(X), dtype=int)

    def fixture_fit(candidate, data, rows):
        fit_calls.append(candidate)
        assert rows == prepared["splits"]["train"]["indices"]
        assert set(rows).isdisjoint(archived_results["evaluation"]["row_ids"])
        return PredictionRecorder(candidate), {}

    monkeypatch.setattr(runner, "fit_candidate", fixture_fit)
    monkeypatch.setattr(runner, "validate_results", lambda payload: None)
    runner._evaluate("validation", tmp_path)
    assert fit_calls == list(CANDIDATES)
    assert all(len(value) == 10 for value in captured.values())
    for position in range(10):
        for candidate in CANDIDATES:
            np.testing.assert_array_equal(captured[candidate][position], captured["baseline"][position])
