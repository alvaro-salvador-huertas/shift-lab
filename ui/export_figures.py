"""Exportación explícita de gráficos autónomos a partir del mismo resultado validado."""

import argparse
from pathlib import Path

from shiftlab.results import load_results
from ui.views import comparison_chart,confusion_chart,curve_chart


def main(argv=None):
    parser=argparse.ArgumentParser(description="Guarda tres gráficos HTML sin CDN ni entrenamiento.")
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    # Reservar una carpeta nueva antes de cargar o escribir ningún gráfico.
    args.output.mkdir(parents=True,exist_ok=False)
    results=load_results(args.results)
    # Decisiones visuales fijadas con validación antes de observar test.
    candidate="logistic_clean"
    scenario="noise_005"
    charts={"comparacion.html":comparison_chart(results,scenario,"accuracy"),"curva-ruido.html":curve_chart(results,candidate,"noise","accuracy"),"matriz-confusion.html":confusion_chart(results["candidates"][candidate]["scenarios"][scenario]["confusion_matrix"])}
    for name,figure in charts.items():
        figure.write_html(args.output/name,include_plotlyjs=True,full_html=True,config={"displayModeBar":False,"responsive":True})
    print(f"{len(charts)} gráficos HTML autónomos, fuente {args.results}, etapa {results['stage']}: {args.output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
