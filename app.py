"""ShiftLab: interfaz sobre evaluaciones archivadas, sin entrenamiento implícito."""

from html import escape
from pathlib import Path

import streamlit as st

from shiftlab.results import export_results, load_results, validate_results
from ui.views import render_errors, render_explorer, render_summary, scenario_label

PROJECT=Path(__file__).resolve().parent
VIEWS=["Resumen y comparación","Explorador","Análisis de errores"]


def main():
    st.set_page_config(page_title="ShiftLab · robustez de imágenes",page_icon="▦",layout="wide",initial_sidebar_state="expanded")
    st.markdown(f"<style>{(PROJECT/'ui/style.css').read_text(encoding='utf-8')}</style>",unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Evaluación archivada")
        path=st.text_input("Archivo de resultados",value="artifacts/final/results.json",key="results_path")
        load=st.button("Cargar resultados",type="primary",width="stretch",key="load_results")
        st.caption("Los controles seleccionan imágenes y predicciones ya calculadas. El entrenamiento se inicia desde la CLI.")
    if st.session_state.get("requested_results_path")!=path:
        st.session_state["requested_results_path"]=path
        st.session_state.pop("results",None)
        st.session_state.pop("loaded_results_path",None)
        for key in ("candidate_selector","scenario_selector","image_selector","explorer_class","explorer_status"):
            st.session_state.pop(key,None)
    st.title("ShiftLab")
    st.markdown('<p class="shift-intro">Explora cómo cambian las predicciones de un clasificador cuando se degradan las mismas imágenes.</p>',unsafe_allow_html=True)
    if load:
        st.session_state.pop("results",None)
        try:
            resolved=(PROJECT/path).resolve()
            if not resolved.is_relative_to(PROJECT):
                raise ValueError("El archivo debe encontrarse dentro del proyecto.")
            with st.spinner("Verificando particiones, predicciones y procedencia del archivo…"):
                results=load_results(resolved)
                st.session_state["results"]=results
                st.session_state["loaded_results_path"]=str(resolved.relative_to(PROJECT))
        except (ValueError,OSError) as error:
            st.error(f"No se pudieron cargar los resultados: {error}")
    results=st.session_state.get("results")
    if results is not None:
        try:
            validate_results(results)
        except ValueError as error:
            st.session_state.pop("results",None)
            results=None
            st.error(f"El archivo es incompatible o fue alterado: {error}. Vuelve a cargarlo.")
    if results is None:
        st.info("Carga un archivo para comparar resultados reales. Si todavía no existe, ejecuta primero el protocolo desde la terminal.")
        st.markdown("**Qué podrás inspeccionar**")
        st.write("Cinco candidatos, una evaluación limpia y nueve condiciones de ruido, desenfoque y contraste. Cada imagen conserva su etiqueta y su relación con la versión original.")
        st.caption("Durante el desarrollo utiliza el archivo de validación. El test final se evalúa tras congelar el protocolo.")
        with st.expander("Cómo generar resultados"):
            st.code(".venv/bin/python -m shiftlab --help",language="bash")
        return
    stage=results["stage"]
    label="Validación · archivo de desarrollo" if stage=="validation" else "Test final · archivo posterior a la congelación"
    st.markdown(f'<div class="shift-source"><strong>{escape(label)}</strong><br>{len(results["evaluation"]["row_ids"])} imágenes evaluadas · {len(results["candidates"])} candidatos · {len(next(iter(results["candidates"].values()))["scenarios"])} condiciones<br>Archivo: {escape(st.session_state["loaded_results_path"])}</div>',unsafe_allow_html=True)
    if stage=="validation":
        st.caption("Esta vista de validación sirve para desarrollar y revisar el experimento. Sus resultados no son una evaluación final independiente.")
    with st.sidebar:
        candidates=list(results["candidates"])
        default=candidates.index("logistic_clean") if "logistic_clean" in candidates else 0
        candidate=st.selectbox("Candidato",candidates,index=default,format_func=lambda key:results["candidates"][key]["label"],key="candidate_selector")
        scenarios=list(results["candidates"][candidate]["scenarios"])
        scenario=st.selectbox("Condición de evaluación",scenarios,format_func=scenario_label,key="scenario_selector")
        st.download_button("Descargar resultados · JSON",export_results(results),file_name=f"shiftlab-{stage}-resultados.json",mime="application/json",key="download_results",on_click="ignore",width="stretch")
        with st.expander("Protocolo y procedencia"):
            st.json({"stage":stage,"provenance":results["provenance"],"protocol":results["protocol"],"splits":results["splits"]},expanded=False)
    view=st.radio("Vista",VIEWS,horizontal=True,label_visibility="collapsed",key="view_selector")
    st.caption(f"Candidato seleccionado: {results['candidates'][candidate]['label']}. Condición: {scenario_label(scenario)}.")
    if view==VIEWS[0]:
        render_summary(results,candidate,scenario)
    elif view==VIEWS[1]:
        render_explorer(results,candidate,scenario)
    else:
        render_errors(results,candidate,scenario)
    st.markdown('<div class="shift-footer">Imágenes reales de load_digits; degradaciones sintéticas. Este conjunto de 8×8 no acredita robustez en producción ni generalización a escritores nuevos.</div>',unsafe_allow_html=True)


if __name__=="__main__":
    main()
