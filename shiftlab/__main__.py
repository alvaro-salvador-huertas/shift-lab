"""CLI explícita; abrir Streamlit no ejecuta estas acciones."""

import argparse
import json
from pathlib import Path

from . import runner


def main(argv=None):
    parser = argparse.ArgumentParser(description="ShiftLab: preparar → coste → validación → congelar → test final.")
    parser.add_argument("action", choices=["prepare", "cost", "validation", "freeze", "final"])
    parser.add_argument("--artifacts", default="artifacts", type=Path, help="Carpeta nueva para cada protocolo; nunca sobreescribe resultados.")
    parser.add_argument("--visual-reviewed", action="store_true", help="Declara la inspección visual real de la cuadrícula de entrenamiento.")
    parser.add_argument("--review-note", default="", help="Nota de la revisión visual anterior a la congelación.")
    args = parser.parse_args(argv)
    try:
        operation = getattr(runner, args.action)
        result = operation(args.artifacts, args.visual_reviewed, args.review_note) if args.action == "freeze" else operation(args.artifacts)
    except (ValueError, OSError, KeyError, TimeoutError) as exc:
        parser.error(str(exc))
    if args.action == "cost":
        print(json.dumps({key: result[key] for key in ("elapsed_seconds", "conservative_estimate_seconds", "within_budget")}, indent=2))
    elif args.action in ("validation", "final"):
        print(f"{args.action}: {len(result['candidates'])} candidatos × 10 escenarios en {result['elapsed_seconds']:.3f} s")
        for key, value in result["candidates"].items():
            print(f"{key}: clean={value['scenarios']['clean']['metrics']} warnings={len(value['training']['warnings'])}")
    else:
        print(f"{args.action}: artefacto guardado en {args.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
