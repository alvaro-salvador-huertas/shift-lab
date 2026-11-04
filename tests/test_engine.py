"""Propiedades de datos, aislamiento, transformaciones y pesos reales de fit."""

from copy import deepcopy

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from shiftlab.data import duplicate_report, image_groups, load_dataset, make_splits, validate_splits
from shiftlab.models import fit_candidate
from shiftlab.protocol import PROTOCOL, SCENARIOS
from shiftlab.results import metrics_from_predictions, paired_difference
from shiftlab.transforms import apply_transform, training_batch


@pytest.fixture(scope="module")
def data():
    return load_dataset()


@pytest.fixture(scope="module")
def splits(data):
    return make_splits(data)


def test_real_digits_normalization_and_reproducible_grouped_splits(data, splits):
    assert data.images.shape == (1797, 8, 8)
    assert data.X.shape == (1797, 64)
    assert np.isfinite(data.images).all()
    assert 0 <= data.images.min() <= data.images.max() <= 1
    assert np.array_equal(data.images * 16, np.rint(data.images * 16))
    assert make_splits(data) == splits
    rows, groups = set(), set()
    for name in ("train", "validation", "test"):
        chosen = splits[name]["indices"]
        assert set(chosen).isdisjoint(rows)
        assert set(data.groups[chosen]).isdisjoint(groups)
        assert sum(splits[name]["class_distribution"].values()) == len(chosen)
        assert set(data.y[chosen]) == set(range(10))
        rows.update(chosen)
        groups.update(data.groups[chosen])
    assert rows == set(range(1797))


def test_duplicate_groups_ignore_labels_and_preserve_contradictions():
    images = np.zeros((3, 8, 8), dtype=np.uint8)
    images[2, 3, 4] = 16
    groups = image_groups(images)
    assert groups[0] == groups[1] != groups[2]
    report = duplicate_report(groups, np.array([2, 7, 4]))
    assert report["duplicate_group_count"] == 1
    assert report["contradictory_groups"][0]["indices"] == [0, 1]
    assert report["contradictory_groups"][0]["labels"] == [2, 7]
    assert len(images) == 3  # No se elimina selectivamente ninguna fila.


def test_row_overlap_mutation_is_rejected(data, splits):
    broken = deepcopy(splits)
    broken["validation"]["indices"][0] = splits["train"]["indices"][0]
    with pytest.raises(ValueError):
        validate_splits(data, broken)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_transformations_finite_bounded_reproducible_and_do_not_mutate(data, scenario):
    rows = np.array([2, 14, 33])
    original = data.images[rows].copy()
    first = apply_transform(original, scenario, rows)
    assert first.shape == (3, 8, 8)
    assert np.isfinite(first).all()
    assert first.min() >= 0 and first.max() <= 1
    np.testing.assert_array_equal(first, apply_transform(original, scenario, rows))
    np.testing.assert_array_equal(original, data.images[rows])
    order = [2, 0, 1]
    np.testing.assert_array_equal(first[order], apply_transform(original[order], scenario, rows[order]))


def test_noise_base_is_shared_between_levels_but_separate_for_augmentation():
    images = np.full((2, 8, 8), 0.5)
    rows = np.array([10, 20])
    small = apply_transform(images, "noise_0025", rows)
    medium = apply_transform(images, "noise_005", rows)
    np.testing.assert_allclose((small - images) * 2, medium - images, atol=1e-15)
    assert not np.array_equal(medium, apply_transform(images, "noise_005", rows, "augmentation"))


def test_blur_never_mixes_images_and_contrast_formula_is_exact():
    images = np.stack([np.zeros((8, 8)), np.ones((8, 8))])
    np.testing.assert_allclose(apply_transform(images, "blur_065", [1, 2]), images, atol=1e-15)
    np.testing.assert_allclose(apply_transform(images, "contrast_070", [1, 2]), 0.7 * images + 0.3 * 0.5)


def test_augmented_versions_keep_train_mapping_and_unit_total_weight(data, splits):
    rows = splits["train"]["indices"][:5]
    batch = training_batch(data, rows, True)
    assert batch["X"].shape == (20, 64)
    assert set(batch["row_ids"]) == set(rows)
    assert set(batch["row_ids"]).isdisjoint(splits["validation"]["indices"] + splits["test"]["indices"])
    for row in rows:
        mask = batch["row_ids"] == row
        assert batch["weights"][mask].tolist() == [0.25] * 4
        assert batch["weights"][mask].sum() == 1
        assert batch["versions"][mask].tolist() == ["clean", "noise_005", "blur_045", "contrast_070"]
        assert batch["y"][mask].tolist() == [int(data.y[row])] * 4
        original = batch["X"][(batch["row_ids"] == row) & (batch["versions"] == "clean")][0]
        np.testing.assert_array_equal(original, data.X[row])


@pytest.mark.parametrize("candidate", ["logistic_clean", "logistic_augmented", "svm_clean", "svm_augmented"])
def test_actual_scaler_and_estimator_fit_receive_train_images_labels_and_weights(data, splits, monkeypatch, candidate):
    rows = splits["train"]["indices"][:60]
    expected = training_batch(data, rows, candidate.endswith("augmented"))
    captured = {}
    scaler_fit = StandardScaler.fit
    estimator_type = LogisticRegression if candidate.startswith("logistic") else SVC
    estimator_fit = estimator_type.fit

    def scaler_spy(self, X, y=None, sample_weight=None):
        captured["scaler"] = (np.asarray(X).copy(), np.asarray(sample_weight).copy())
        return scaler_fit(self, X, y, sample_weight=sample_weight)

    def estimator_spy(self, X, y, sample_weight=None):
        captured["estimator"] = (np.asarray(X).copy(), np.asarray(y).copy(), np.asarray(sample_weight).copy())
        return estimator_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", scaler_spy)
    monkeypatch.setattr(estimator_type, "fit", estimator_spy)
    model, record = fit_candidate(candidate, data, rows)
    np.testing.assert_array_equal(captured["scaler"][0], expected["X"])
    np.testing.assert_array_equal(captured["scaler"][1], expected["weights"])
    np.testing.assert_array_equal(captured["estimator"][1], expected["y"])
    np.testing.assert_array_equal(captured["estimator"][2], expected["weights"])
    mean = np.average(expected["X"], axis=0, weights=expected["weights"])
    variance = np.average((expected["X"] - mean) ** 2, axis=0, weights=expected["weights"])
    scale = np.sqrt(variance)
    scale[scale == 0] = 1
    np.testing.assert_allclose(captured["estimator"][0], (expected["X"] - mean) / scale, atol=1e-12)
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, mean, atol=1e-15)
    assert record["training"]["fit_audit"]["scaler"]["n_samples_seen"] == len(rows)
    assert record["training"]["row_ids"] == expected["row_ids"].tolist()


def test_attempting_to_train_on_validation_or_test_is_rejected(data, splits):
    for name in ("validation", "test"):
        with pytest.raises(ValueError, match="entrenamiento"):
            fit_candidate("logistic_clean", data, [splits[name]["indices"][0]])


def test_metrics_missing_classes_and_paired_units_have_explicit_semantics():
    y, clean, shifted = [0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]
    result = metrics_from_predictions(y, shifted)
    assert result["metrics"]["accuracy"] == 0.5
    assert result["metrics"]["macro_f1"] == 0.1  # Dos clases F1=.5; las otras ocho se definen como 0.
    assert sum(map(sum, result["confusion_matrix"])) == 4
    assert result["per_class"][9]["support"] == 0
    delta = paired_difference(y, clean, shifted)
    assert delta["correct_to_wrong"] == 2 and delta["wrong_to_correct"] == 0
    assert delta["per_image_correct_change"] == [0, -1, -1, 0]
    assert delta["accuracy_difference"] == -0.5


@pytest.mark.parametrize("images,scenario,rows", [
    (np.zeros((2, 8, 8)), "unknown", [1, 2]),
    (np.full((1, 8, 8), np.nan), "clean", [1]),
    (np.zeros((1, 8, 8)), "clean", [-1]),
    (np.zeros((2, 8, 8)), "clean", [1]),
])
def test_invalid_transform_parameters_fail_clearly(images, scenario, rows):
    with pytest.raises(ValueError):
        apply_transform(images, scenario, rows)
