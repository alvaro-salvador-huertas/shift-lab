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


def test_export_rejects_dangling_output_symlink_before_loading(tmp_path, monkeypatch):
    target = tmp_path / "absent"
    output = tmp_path / "figures"
    output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(export_figures, "load_results", lambda _: pytest.fail("No debe cargar ni escribir"))
    with pytest.raises(FileExistsError):
        export_figures.main(["--results", "fixture.json", "--output", str(output)])
    assert output.readlink() == target
    assert not target.exists()


def test_export_does_not_create_output_for_invalid_results(tmp_path):
    source = tmp_path / "results.json"
    source.write_text("{", encoding="utf-8")
    output = tmp_path / "new-parent" / "figures"
    with pytest.raises(ValueError, match="No se encuentran resultados válidos"):
        export_figures.main(["--results", str(source), "--output", str(output)])
    assert not output.parent.exists()


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
