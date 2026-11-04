"""Conservación de exportaciones; usa figuras sintéticas y no evalúa modelos."""
import pytest
import plotly.graph_objects as go
from ui import export_figures


def test_export_rejects_existing_directory_before_loading(tmp_path, monkeypatch):
    output = tmp_path / "figures"
    output.mkdir()
    original = output / "comparacion.html"
    original.write_bytes(b"original conservado")
    monkeypatch.setattr(export_figures, "load_results", lambda _: pytest.fail("No debe cargar ni escribir"))
    with pytest.raises(FileExistsError):
        export_figures.main(["--results", "fixture.json", "--output", str(output)])
    assert original.read_bytes() == b"original conservado"
    assert list(output.iterdir()) == [original]


def test_export_writes_three_standalone_figures_to_new_directory(tmp_path, monkeypatch):
    payload = {"stage": "validation", "candidates": {"logistic_clean": {"scenarios": {"noise_005": {"confusion_matrix": [[1]]}}}}}
    monkeypatch.setattr(export_figures, "load_results", lambda _: payload)
    for name in ("comparison_chart", "curve_chart", "confusion_chart"):
        monkeypatch.setattr(export_figures, name, lambda *args: go.Figure(go.Bar(y=[1])))
    output = tmp_path / "figures"
    assert export_figures.main(["--results", "fixture.json", "--output", str(output)]) == 0
    assert {p.name for p in output.iterdir()} == {"comparacion.html", "curva-ruido.html", "matriz-confusion.html"}
    for path in output.iterdir():
        html = path.read_text()
        assert "Plotly.newPlot" in html and "<html>" in html
        assert '<script src=' not in html
