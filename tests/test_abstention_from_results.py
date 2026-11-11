"""Regeneración del JSON publicado: sin entrenar ni consultar otros datos."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import abstention as experiment


ARCHIVE = Path(__file__).resolve().parents[1] / "artifacts/abstention-20261111/results.json"


@pytest.fixture
def archived():
    return json.loads(ARCHIVE.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("La regeneración no debe ejecutar el experimento, entrenar, cargar datos ni elegir umbral")

    for name in ("run_experiment", "fit_candidate", "load_dataset", "_predict", "choose_threshold"):
        monkeypatch.setattr(experiment, name, forbidden)


def test_regeneration_preserves_published_payload_threshold_counts_and_source(tmp_path, monkeypatch, archived):
    source_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    output = tmp_path / "regenerated"
    captured = {}
    original_figure = experiment.coverage_figure

    def figure(payload):
        captured["payload"] = deepcopy(payload)
        captured["figure"] = original_figure(payload)
        return captured["figure"]

    monkeypatch.setattr(experiment, "coverage_figure", figure)
    # No hay prepared.json ni dataset en este directorio.
    monkeypatch.chdir(tmp_path)
    assert experiment.main(["--from-results", str(ARCHIVE), "--output", str(output)]) == 0
    assert captured["payload"] == archived
    assert {p.name for p in output.iterdir()} == {"cobertura-error.html", "informe.txt"}
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == source_hash
    figure = captured["figure"]
    assert f"{archived['selection']['threshold']:.8f}" in figure.layout.title.text
    for i, result in enumerate(archived["evaluation"]["conditions"].values()):
        assert list(figure.data[i * 3 + 2].x) == [result["abstention"]["coverage"]]
        assert list(figure.data[i * 3 + 2].y) == [result["abstention"]["accepted_error_rate"]]
    report = (output / "informe.txt").read_text()
    assert "Regeneración desde resultados archivados, sin reentrenamiento." in report
    assert f"JSON de origen: {ARCHIVE}" in report
    assert f"UMBRAL FIJADO: {archived['selection']['threshold']:.17g}" in report
    assert "--from-results" in report and "--prepared" not in report
    assert "Limpio | Umbral fijado | 179 | 166 | 13 | 2 | 6 | 92.74% | 1.20%" in report
    assert "Ruido · σ 0,10 | Umbral fijado | 179 | 161 | 18 | 23 | 6 | 89.94% | 14.29%" in report
    html = (output / "cobertura-error.html").read_text()
    assert "Plotly.newPlot" in html and "<script src=" not in html


@pytest.mark.parametrize("kind", ["directory", "file", "dangling_symlink"])
def test_existing_destination_rejected_before_loading(tmp_path, monkeypatch, capsys, kind):
    output = tmp_path / "existing"
    if kind == "directory":
        output.mkdir()
        sentinel = output / "informe.txt"
        sentinel.write_bytes(b"conservar")
    elif kind == "file":
        output.write_bytes(b"conservar")
    else:
        output.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(experiment, "load_abstention_results", lambda *a: pytest.fail("No debe leer el origen"))
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--from-results", str(ARCHIVE), "--output", str(output)])
    assert exc.value.code == 2 and "salida ya existe" in capsys.readouterr().err
    if kind == "directory":
        assert sentinel.read_bytes() == b"conservar" and len(list(output.iterdir())) == 1
    elif kind == "file":
        assert output.read_bytes() == b"conservar"
    else:
        assert output.is_symlink() and not output.exists()


@pytest.mark.parametrize("contents", ["{", "null", "[]", "{}", '{"x": NaN}', b"\xff"])
def test_invalid_json_has_clear_diagnostic_and_creates_no_output(tmp_path, capsys, contents):
    source = tmp_path / "bad.json"
    source.write_bytes(contents if isinstance(contents, bytes) else contents.encode())
    output = tmp_path / "new-parent" / "output"
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--from-results", str(source), "--output", str(output)])
    diagnostic = capsys.readouterr().err
    assert exc.value.code == 2 and "JSON de abstención inválido o ilegible" in diagnostic
    assert str(source) in diagnostic
    assert not output.parent.exists()


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema_version="otro/1"),
    lambda p: p["evaluation"].update(fixed_threshold=.99),
    lambda p: p["selection"].update(threshold="alto"),
    lambda p: p["selection"].update(accepted=0),
    lambda p: p["selection"].update(scores=[.5] * 180, accepted=0, coverage=0.0),
    lambda p: p["evaluation"]["conditions"].pop("clean"),
    lambda p: p["evaluation"]["conditions"]["clean"].update(label=None),
    lambda p: p["evaluation"]["conditions"]["clean"]["scores"].pop(),
    lambda p: p["evaluation"]["conditions"]["clean"]["y_pred"].__setitem__(0, 99),
    lambda p: p["evaluation"]["conditions"]["clean"]["accepted_mask"].__setitem__(0, "sí"),
    lambda p: p["evaluation"]["conditions"]["clean"]["abstention"].update(errors_accepted=0),
    lambda p: p["evaluation"]["conditions"]["clean"]["descriptive_curve"][-1].update(coverage=.5),
    lambda p: p["validation_subsets"]["selection"].update(row_ids=[]),
    lambda p: p["validation_subsets"].update(selection=None),
])
def test_inconsistent_archive_rejected_before_any_output(tmp_path, archived, capsys, mutation):
    mutation(archived)
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(archived), encoding="utf-8")
    output = tmp_path / "no-output"
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--from-results", str(source), "--output", str(output)])
    assert exc.value.code == 2 and "JSON de abstención inválido o ilegible" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_unreadable_source_diagnosed_without_output(tmp_path, capsys, kind):
    source = tmp_path / "source.json"
    if kind == "directory":
        source.mkdir()
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--from-results", str(source), "--output", str(output)])
    assert exc.value.code == 2 and "ilegible" in capsys.readouterr().err
    assert not output.exists()


def test_zero_accepted_rate_remains_undefined(tmp_path, archived):
    for result in archived["evaluation"]["conditions"].values():
        result["scores"] = [.5] * 179
        result["accepted_mask"] = [False] * 179
        result["abstention"] = {
            "total": 179, "accepted": 0, "rejected": 179, "errors_accepted": 0,
            "errors_rejected": result["mandatory"]["errors_accepted"],
            "coverage": 0.0, "accepted_error_rate": None,
        }
        result["descriptive_curve"] = [
            {"threshold": float(np.nextafter(.5, np.inf)), "coverage": 0.0, "accepted_error_rate": None},
            {"threshold": .5, "coverage": 1.0, "accepted_error_rate": result["mandatory"]["accepted_error_rate"]},
        ]
    source = tmp_path / "zero.json"
    source.write_text(json.dumps(archived), encoding="utf-8")
    assert experiment.load_abstention_results(source) == archived
    report = experiment.brief_report(archived, "comando", from_results=source)
    assert "tasa sin definir: ninguna imagen aceptada" in report
    assert experiment.coverage_figure(archived).data[2].y[0] is None


def test_source_options_are_mutually_exclusive(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--from-results", str(ARCHIVE), "--prepared", "prepared.json", "--output", str(output)])
    assert exc.value.code == 2 and not output.exists()


def test_previous_prepared_command_keeps_its_three_outputs(tmp_path, monkeypatch, archived):
    calls = []
    prepared = tmp_path / "prepared.json"

    def run(path):
        calls.append(path)
        return archived  # Fixture archivado; ningún ajuste real.

    monkeypatch.setattr(experiment, "run_experiment", run)
    output = tmp_path / "old-cli"
    assert experiment.main(["--prepared", str(prepared), "--output", str(output)]) == 0
    assert calls == [prepared]
    assert {p.name for p in output.iterdir()} == {"results.json", "cobertura-error.html", "informe.txt"}
    assert json.loads((output / "results.json").read_text()) == archived
    report = (output / "informe.txt").read_text()
    assert "--prepared" in report and "Regeneración" not in report


def test_render_failure_does_not_leave_partial_output(tmp_path, monkeypatch):
    class BrokenFigure:
        def to_html(self, **kwargs):
            raise ValueError("No se puede renderizar")

    monkeypatch.setattr(experiment, "coverage_figure", lambda *a: BrokenFigure())
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        experiment.main(["--from-results", str(ARCHIVE), "--output", str(output)])
    assert exc.value.code == 2 and not output.exists()
