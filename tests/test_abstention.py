"""Abstención exploratoria: grupos, orden de selección y recuentos auditables."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import abstention as experiment
from shiftlab.data import Dataset, image_groups, load_dataset, make_splits


def test_subsplit_preserves_duplicates_contradictions_and_all_versions():
    originals = np.zeros((100, 8, 8), dtype=np.uint8)
    originals[:, 0, 0] = np.arange(100) % 17
    originals[:, 0, 1] = np.arange(100) // 17
    images = np.repeat(originals, 2, axis=0)
    labels = np.repeat(np.arange(100) % 10, 2)
    labels[1] = 1  # Un duplicado contradictorio se conserva en el mismo grupo.
    data = Dataset(images / 16.0, labels, image_groups(images), {})
    rows = list(range(len(labels)))
    split = experiment.split_validation(data, rows)
    assert split == experiment.split_validation(data, rows[::-1])
    a, b = split["selection"], split["evaluation"]
    assert set(a["row_ids"]).isdisjoint(b["row_ids"])
    assert set(a["row_ids"]) | set(b["row_ids"]) == set(rows)
    assert set(a["groups"]).isdisjoint(b["groups"])
    for subset in (a, b):
        assert set(subset["class_distribution"]) == set(map(str, range(10)))
        assert all(count > 0 for count in subset["class_distribution"].values())
        assert sum(subset["class_distribution"].values()) == subset["total"]
        selected = set(subset["row_ids"])
        for row in selected:
            assert (row ^ 1) in selected  # Ambos duplicados, aun con etiquetas distintas.
        for condition in experiment.CONDITIONS:
            transformed = experiment.evaluation_images(subset["row_ids"], condition, data)
            assert len(transformed) == subset["total"]


def test_real_subsplit_stratified_and_isolated_from_original_train_and_test():
    data = load_dataset()
    splits = make_splits(data)
    divided = experiment.split_validation(data, splits["validation"]["indices"])
    validation_groups = set(data.groups[splits["validation"]["indices"]])
    forbidden = set(splits["train"]["indices"] + splits["test"]["indices"])
    for name in ("selection", "evaluation"):
        subset = divided[name]
        assert set(subset["row_ids"]).isdisjoint(forbidden)
        assert set(subset["groups"]) <= validation_groups
        assert set(data.y[subset["row_ids"]]) == set(range(10))
        assert abs(subset["total"] - 359 / 2) < 2
        for label, count in subset["class_distribution"].items():
            assert abs(count - splits["validation"]["class_distribution"][label] / 2) <= 4
    assert set(divided["selection"]["groups"]).isdisjoint(divided["evaluation"]["groups"])


@pytest.mark.parametrize("scores,target,expected", [
    ([0.8], 0.95, 0.8),
    ([0.4, 0.5, 0.8], 0.95, 0.4),
    (np.arange(1, 21) / 20, 0.95, 0.10),
    ([0.1] + [0.5] * 3 + [0.9] * 16, 0.95, 0.5),
    ([0.5] * 20, 0.95, 0.5),
    ([0.1, 0.2, 0.3, 0.4], 0.5, 0.3),
    ([0.0, 0.5, 1.0], 1.0, 0.0),
    (np.linspace(0, 1, 100), 0.07, 93 / 99),
])
def test_threshold_is_largest_feasible_cut_with_inclusive_ties(scores, target, expected):
    values = np.asarray(scores)
    threshold = experiment.choose_threshold(values, target)
    assert threshold == pytest.approx(expected)
    assert (values >= threshold).mean() >= target
    assert (values >= np.nextafter(threshold, np.inf)).mean() < target
    assert experiment.choose_threshold(values[::-1], target) == threshold


@pytest.mark.parametrize("scores,target", [([], .95), ([np.nan], .95), ([np.inf], .95),
                                          ([-.1], .95), ([1.1], .95), ([[.5]], .95),
                                          ([.5], 0), ([.5], 1.01)])
def test_invalid_threshold_inputs_rejected(scores, target):
    with pytest.raises(ValueError):
        experiment.choose_threshold(scores, target)


@pytest.mark.parametrize("threshold", [0, .3, .5, .8, 1.01])
def test_counts_match_independent_manual_arithmetic_and_undefined_rate(threshold):
    truth, predicted, scores = [0, 1, 2, 3, 4], [0, 0, 2, 0, 0], [.2, .3, .5, .8, .8]
    accepted = [i for i, score in enumerate(scores) if score >= threshold]
    rejected = [i for i, score in enumerate(scores) if score < threshold]
    errors_a = sum(truth[i] != predicted[i] for i in accepted)
    errors_r = sum(truth[i] != predicted[i] for i in rejected)
    result = experiment.policy_counts(truth, predicted, scores, threshold)
    assert result == {
        "total": 5, "accepted": len(accepted), "rejected": len(rejected),
        "errors_accepted": errors_a, "errors_rejected": errors_r,
        "coverage": len(accepted) / 5,
        "accepted_error_rate": errors_a / len(accepted) if accepted else None,
    }
    assert result["accepted"] + result["rejected"] == result["total"]
    assert result["errors_accepted"] + result["errors_rejected"] == 3
    assert result["errors_accepted"] <= result["accepted"]
    assert result["errors_rejected"] <= result["rejected"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_curve_uses_whole_ties_and_matches_policy_counts():
    truth, prediction, scores = [0, 1, 2, 3], [0, 0, 2, 0], [.9, .8, .8, .2]
    curve = experiment.risk_coverage_curve(truth, prediction, scores)
    assert [p["coverage"] for p in curve] == [0, .25, .75, 1]
    assert curve[0]["accepted_error_rate"] is None
    for point in curve:
        counts = experiment.policy_counts(truth, prediction, scores, point["threshold"])
        assert counts["coverage"] == point["coverage"]
        assert counts["accepted_error_rate"] == point["accepted_error_rate"]


@pytest.fixture
def synthetic_run(monkeypatch):
    """Recorre la orquestación con predictores espía, sin ajustar ni evaluar test."""
    data = load_dataset()
    splits = make_splits(data)
    halves = experiment.split_validation(data, splits["validation"]["indices"])
    selection = halves["selection"]["row_ids"]
    evaluation = halves["evaluation"]["row_ids"]
    events = []
    original_choose = experiment.choose_threshold
    original_images = experiment.evaluation_images

    class Predictor:
        def predict_proba(self, X):
            scores = .60 + .39 * X.mean(axis=1)
            return np.column_stack([scores, 1 - scores])

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    def fit(candidate, dataset, rows):
        assert candidate == "logistic_clean" and dataset.metadata == data.metadata
        assert rows == splits["train"]["indices"]
        assert not set(rows) & set(selection + evaluation)
        events.append("fit")
        return Predictor(), {"training": {"original_indices": rows}}

    def images(rows, condition, dataset):
        assert rows in (selection, evaluation)
        if rows == selection:
            assert events == ["fit"] and condition == "clean"
            events.append("selection_clean")
        else:
            assert "threshold_fixed" in events
            events.append(condition)
        return original_images(rows, condition, dataset)

    def choose(scores, min_coverage=experiment.MIN_COVERAGE):
        assert events == ["fit", "selection_clean"]
        events.append("threshold_fixed")
        return original_choose(scores, min_coverage)

    monkeypatch.setattr(experiment, "fit_candidate", fit)
    monkeypatch.setattr(experiment, "evaluation_images", images)
    monkeypatch.setattr(experiment, "choose_threshold", choose)
    root = Path(__file__).resolve().parents[1]
    result = experiment.run_experiment(root / "artifacts" / "prepared.json")
    return result, events


def test_orchestration_fits_only_original_train_and_fixes_threshold_before_measurement(synthetic_run):
    payload, events = synthetic_run
    assert events == ["fit", "selection_clean", "threshold_fixed", *experiment.CONDITIONS]
    selection = payload["selection"]
    evaluation = payload["evaluation"]
    assert evaluation["fixed_threshold"] == selection["threshold"]
    assert payload["analysis"] == "exploratory"
    assert payload["provenance"]["final_test_evaluated"] is False
    for result in evaluation["conditions"].values():
        assert result["row_ids"] == evaluation["row_ids"]
        scores = np.array(result["scores"])
        assert result["accepted_mask"] == (scores >= selection["threshold"]).tolist()
        for policy, threshold in (("mandatory", 0), ("abstention", selection["threshold"])):
            assert result[policy] == experiment.policy_counts(evaluation["y_true"], result["y_pred"], scores, threshold)
        a, b = result["mandatory"], result["abstention"]
        assert a["total"] == b["total"] == len(evaluation["row_ids"])
        assert a["accepted"] == a["total"] and a["rejected"] == a["errors_rejected"] == 0
        assert a["errors_accepted"] == b["errors_accepted"] + b["errors_rejected"]


def test_mismatched_original_preparation_rejected_before_fitting(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    prepared = json.loads((root / "artifacts" / "prepared.json").read_text())
    prepared["splits"]["validation"]["indices"].reverse()
    source = tmp_path / "prepared.json"
    source.write_text(json.dumps(prepared))
    monkeypatch.setattr(experiment, "fit_candidate", lambda *a: pytest.fail("No debe entrenar"))
    with pytest.raises(ValueError, match="original"):
        experiment.run_experiment(source)


def test_plot_highlights_fixed_threshold_and_mandatory_points(synthetic_run):
    payload, _ = synthetic_run
    figure = experiment.coverage_figure(payload)
    assert len(figure.data) == 9
    for index, result in enumerate(payload["evaluation"]["conditions"].values()):
        curve, mandatory, fixed = figure.data[3 * index:3 * index + 3]
        assert list(curve.x) == [p["coverage"] for p in result["descriptive_curve"]]
        assert list(curve.y) == [p["accepted_error_rate"] for p in result["descriptive_curve"]]
        assert list(mandatory.x) == [1.0]
        assert list(fixed.x) == [result["abstention"]["coverage"]]
        assert list(fixed.y) == [result["abstention"]["accepted_error_rate"]]
        assert fixed.marker.symbol == "star"
    empty = deepcopy(payload)
    empty["evaluation"]["conditions"]["clean"]["abstention"]["accepted_error_rate"] = None
    assert experiment.coverage_figure(empty).data[2].y[0] is None
    assert "sin definir" in experiment.brief_report(empty, "comando")


def test_cli_preserves_existing_destination_before_any_read_or_training(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    original = output / "results.json"
    original.write_bytes(b"historial intacto")
    monkeypatch.setattr(experiment, "run_experiment", lambda *a: pytest.fail("No debe leer ni entrenar"))
    with pytest.raises(SystemExit):
        experiment.main(["--output", str(output)])
    assert original.read_bytes() == b"historial intacto"


def test_cli_exports_strict_json_standalone_plot_and_report(tmp_path, synthetic_run, monkeypatch):
    payload, _ = synthetic_run
    monkeypatch.setattr(experiment, "run_experiment", lambda *a: payload)
    output = tmp_path / "new"
    assert experiment.main(["--output", str(output)]) == 0
    assert {p.name for p in output.iterdir()} == {"results.json", "cobertura-error.html", "informe.txt"}
    assert json.loads((output / "results.json").read_text()) == payload
    html = (output / "cobertura-error.html").read_text()
    assert "Plotly.newPlot" in html and "<html>" in html and "<script src=" not in html
    report = (output / "informe.txt").read_text()
    assert "UMBRAL FIJADO" in report and "exploratorio" in report
    assert "--output" in report and str(output) in report
