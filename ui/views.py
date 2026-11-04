"""Vistas sobre resultados archivados. No importa estimadores ni ejecuta fit."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shiftlab.protocol import SCENARIOS
from shiftlab.results import image_pair

METRICS = {"accuracy": "Accuracy", "macro_f1": "Macro-F1"}
SHORT_LABELS = {"baseline": "Baseline", "logistic_clean": "Logística<br>limpia", "logistic_augmented": "Logística<br>aumentada", "svm_clean": "SVM<br>limpia", "svm_augmented": "SVM<br>aumentada"}
COLORS = {"baseline": "#667892", "logistic_clean": "#3159c9", "logistic_augmented": "#087c8d", "svm_clean": "#794c92", "svm_augmented": "#a65e28"}


def scenario_label(scenario_id):
    return SCENARIOS[scenario_id]["label"]


def comparison_table(results, scenario):
    rows=[]
    for candidate_id, candidate in results["candidates"].items():
        clean=candidate["scenarios"]["clean"]
        shifted=candidate["scenarios"][scenario]
        paired=shifted["paired_vs_clean"]
        rows.append({"Candidato":candidate["label"], "Accuracy limpia":clean["metrics"]["accuracy"], "Accuracy condición":shifted["metrics"]["accuracy"], "Macro-F1 limpia":clean["metrics"]["macro_f1"], "Macro-F1 condición":shifted["metrics"]["macro_f1"], "Δ accuracy":paired["accuracy_difference"], "Δ macro-F1":paired["macro_f1_difference"]})
    return pd.DataFrame(rows)


def _style(figure, title, height=400):
    figure.update_layout(title={"text":title,"font":{"size":15}},template="plotly_white",height=height,paper_bgcolor="#f4f7fc",plot_bgcolor="#f4f7fc",font={"family":"Trebuchet MS, sans-serif","color":"#17243a","size":11},margin={"l":10,"r":10,"t":55,"b":110},legend={"orientation":"h","y":-.35,"yanchor":"top","x":0,"font":{"size":10}},yaxis={"gridcolor":"#dce3ef"})
    return figure


def comparison_chart(results, scenario, metric):
    figure=go.Figure()
    ids=list(results["candidates"])
    for name,key,color in (("Entrada limpia","clean","#3159c9"),("Condición elegida",scenario,"#a65e28")):
        if scenario=="clean" and name=="Condición elegida":
            continue
        figure.add_bar(name=name,x=[SHORT_LABELS.get(candidate,candidate) for candidate in ids],y=[results["candidates"][candidate]["scenarios"][key]["metrics"][metric] for candidate in ids],marker_color=color,hovertemplate="%{x}: %{y:.1%}<extra>%{fullData.name}</extra>")
    _style(figure,f"{METRICS[metric]} de los cinco candidatos")
    figure.update_layout(barmode="group",yaxis={"title":"Puntuación","range":[0,1.05],"tickformat":".0%","gridcolor":"#dce3ef"},xaxis={"title":"Entrenamiento del modelo","tickangle":0})
    return figure


def curve_chart(results, candidate_id, family, metric):
    chosen=results["candidates"][candidate_id]
    candidate_family=chosen["family"]
    ids=[key for key,value in results["candidates"].items() if value["family"]==candidate_family]
    scenarios=["clean"]+[key for key in SCENARIOS if key.startswith(family+"_")]
    figure=go.Figure()
    for key in ids:
        candidate=results["candidates"][key]
        figure.add_scatter(name=candidate["label"],x=[scenario_label(item).replace(" · ","<br>") for item in scenarios],customdata=[scenario_label(item) for item in scenarios],y=[candidate["scenarios"][item]["metrics"][metric] for item in scenarios],mode="lines+markers",line={"color":COLORS.get(key,"#667892"),"width":2},marker={"size":7},hovertemplate="%{customdata}: %{y:.1%}<extra>%{fullData.name}</extra>")
    _style(figure,f"{METRICS[metric]} al cambiar la condición",450)
    figure.update_layout(yaxis={"title":"Puntuación","range":[0,1.05],"tickformat":".0%","gridcolor":"#dce3ef"},xaxis={"title":"Condición de evaluación","tickangle":0})
    return figure


def render_summary(results,candidate_id,scenario):
    st.subheader("La misma evaluación para todos los candidatos")
    metric=st.selectbox("Métrica del gráfico",list(METRICS),format_func=METRICS.get,key="summary_metric")
    st.plotly_chart(comparison_chart(results,scenario,metric),width="stretch",config={"displayModeBar":False})
    st.dataframe(comparison_table(results,scenario),hide_index=True,width="stretch",column_config={key:st.column_config.NumberColumn(format="%.3f") for key in comparison_table(results,scenario).columns if key!="Candidato"})
    selected=results["candidates"][candidate_id]["scenarios"][scenario]
    paired=selected["paired_vs_clean"]
    st.markdown("**Comparación emparejada del candidato seleccionado**")
    st.caption("Cada comparación usa las mismas imágenes originales: las versiones degradadas no se cuentan como observaciones independientes. Los deltas son condición menos limpia; 0,01 equivale a un punto porcentual.")
    st.dataframe(pd.DataFrame([{"Transición":"Acierto → error","Imágenes":paired["correct_to_wrong"]},{"Transición":"Error → acierto","Imágenes":paired["wrong_to_correct"]},{"Transición":"Acierto en ambas","Imágenes":paired["both_correct"]},{"Transición":"Error en ambas","Imágenes":paired["both_wrong"]}]),hide_index=True,width="stretch")
    family=st.selectbox("Degradación de la curva",["noise","blur","contrast"],format_func={"noise":"Ruido gaussiano","blur":"Desenfoque gaussiano","contrast":"Contraste hacia gris"}.get,key="curve_family")
    st.plotly_chart(curve_chart(results,candidate_id,family,metric),width="stretch",config={"displayModeBar":False})
    st.caption("La curva reúne los candidatos de la familia seleccionada. Se conservan mejoras, empates y cambios no monótonos; la aumentación no tiene por qué ayudar en todas las condiciones.")


def prediction_rows(results,candidate_id,scenario):
    candidate=results["candidates"][candidate_id]
    clean=candidate["scenarios"]["clean"]
    shifted=candidate["scenarios"][scenario]
    return pd.DataFrame({"Índice original":shifted["row_ids"],"Clase real":shifted["y_true"],"Predicción limpia":clean["y_pred"],"Predicción condición":shifted["y_pred"]})


def render_explorer(results,candidate_id,scenario):
    st.subheader("Mira qué recibió el clasificador")
    rows=prediction_rows(results,candidate_id,scenario)
    true_class=st.selectbox("Clase real",["Todas"]+list(range(10)),key="explorer_class")
    status=st.selectbox("Predicciones que quieres inspeccionar",["Todas","Errores en esta condición","Acierto → error","Error → acierto"],key="explorer_status")
    filtered=rows if true_class=="Todas" else rows[rows["Clase real"]==true_class]
    if status=="Errores en esta condición":
        filtered=filtered[filtered["Clase real"]!=filtered["Predicción condición"]]
    elif status=="Acierto → error":
        filtered=filtered[(filtered["Clase real"]==filtered["Predicción limpia"])&(filtered["Clase real"]!=filtered["Predicción condición"])]
    elif status=="Error → acierto":
        filtered=filtered[(filtered["Clase real"]!=filtered["Predicción limpia"])&(filtered["Clase real"]==filtered["Predicción condición"])]
    st.caption(f"{len(filtered)} imágenes cumplen el filtro, de {len(rows)} evaluadas. El filtro cambia la vista y conserva las métricas del conjunto completo.")
    if filtered.empty:
        st.info("No hay imágenes con este filtro. Prueba otra clase o selecciona «Todas».")
        return
    ids=filtered["Índice original"].tolist()
    if st.session_state.get("image_selector") not in ids:
        st.session_state.pop("image_selector",None)
    row_id=st.selectbox("Imagen · índice original",ids,key="image_selector")
    row=filtered[filtered["Índice original"]==row_id].iloc[0]
    original,transformed=image_pair(results,scenario,int(row_id))
    left,right=st.columns(2,gap="large")
    for column,image,title,prediction in ((left,original,"Imagen original",row["Predicción limpia"]),(right,transformed,"Entrada en la condición elegida",row["Predicción condición"])):
        with column:
            st.markdown(f"**{title}**")
            st.image(np.asarray(image),clamp=True,width=240)
            st.metric("Predicción",int(prediction))
    st.write(f"**Etiqueta real: {int(row['Clase real'])}.** {scenario_label(scenario)}. Ampliación de los 8×8 píxeles, sin suavizado visual.")
    st.caption("Las dos predicciones pertenecen al mismo candidato. Los márgenes de SVM no se muestran como porcentajes de confianza.")
    with st.expander("Intensidades e identificación de la imagen"):
        st.write(f"Índice original: {row_id}; intensidades normalizadas en [0,1].")
        st.dataframe(pd.DataFrame(np.asarray(transformed)),width="stretch")
    st.dataframe(filtered.head(20),hide_index=True,width="stretch")


def confusion_chart(matrix):
    array=np.asarray(matrix)
    figure=go.Figure(go.Heatmap(z=array,x=list(range(10)),y=list(range(10)),colorscale=[[0,"#e8edf7"],[1,"#3159c9"]],text=array,texttemplate="%{text}",textfont={"size":10},hovertemplate="Real %{y} · predicho %{x}<br>%{z} imágenes<extra></extra>",colorbar={"title":"N"}))
    _style(figure,"Matriz de confusión · recuentos reales",440)
    figure.update_layout(margin={"l":10,"r":10,"t":55,"b":55},xaxis={"title":"Clase predicha","tickmode":"linear","dtick":1},yaxis={"title":"Clase real","tickmode":"linear","dtick":1,"autorange":"reversed"})
    return figure


def render_errors(results,candidate_id,scenario):
    st.subheader("Qué clases se confunden")
    result=results["candidates"][candidate_id]["scenarios"][scenario]
    st.plotly_chart(confusion_chart(result["confusion_matrix"]),width="stretch",config={"displayModeBar":False})
    st.dataframe(pd.DataFrame(result["per_class"]).rename(columns={"class":"Clase","precision":"Precisión","recall":"Recall","f1":"F1","support":"N"}),hide_index=True,width="stretch")
    st.caption("Recall es la proporción de imágenes de una clase que se reconocen correctamente. Macro-F1 da el mismo peso a cada una de las diez clases; accuracy cuenta aciertos entre todas las imágenes.")
    rows=prediction_rows(results,candidate_id,scenario)
    errors=rows[rows["Clase real"]!=rows["Predicción condición"]]
    st.markdown(f"**{len(errors)} errores de {len(rows)} imágenes**")
    if errors.empty:
        st.info("Este candidato no cometió errores en esta condición archivada. Puedes inspeccionar otra condición.")
    else:
        st.dataframe(errors.head(30),hide_index=True,width="stretch")
        st.caption("Se muestran hasta 30 errores. Abre el Explorador para examinar sus píxeles y filtrar por clase.")
