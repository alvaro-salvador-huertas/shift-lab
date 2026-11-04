"""UI sobre fixtures sintéticos aislados; ningún entrenamiento en estas pruebas."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from streamlit.testing.v1 import AppTest

from shiftlab.protocol import SCENARIOS

APP=Path(__file__).resolve().parents[1]/"app.py"


@pytest.fixture
def payload():
    # Fixture de presentación, no es una evaluación ni se distribuye como demo.
    true=np.array([0,1,2,1])
    candidates={}
    for candidate_index,candidate_id in enumerate(("baseline","logistic_clean","logistic_augmented","svm_clean","svm_augmented")):
        scenarios={}
        clean=np.array([0,0,0,0]) if candidate_id=="baseline" else true.copy()
        for scenario_index,scenario in enumerate(SCENARIOS):
            pred=clean.copy()
            if scenario!="clean" and candidate_id!="baseline":
                pred[scenario_index%4]=(pred[scenario_index%4]+1)%10
            correct_clean=clean==true
            correct=pred==true
            precision,recall,f1,support=precision_recall_fscore_support(true,pred,labels=range(10),zero_division=0)
            metrics={"accuracy":float(accuracy_score(true,pred)),"macro_f1":float(f1_score(true,pred,labels=range(10),average="macro",zero_division=0))}
            clean_f1=float(f1_score(true,clean,labels=range(10),average="macro",zero_division=0))
            scenarios[scenario]={"row_ids":[0,1,2,3],"y_true":true.tolist(),"y_pred":pred.tolist(),"metrics":metrics,"confusion_matrix":confusion_matrix(true,pred,labels=range(10)).tolist(),"per_class":[{"class":i,"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),"support":int(support[i])} for i in range(10)],"paired_vs_clean":{"accuracy_difference":metrics["accuracy"]-float(accuracy_score(true,clean)),"macro_f1_difference":metrics["macro_f1"]-clean_f1,"correct_to_wrong":int((correct_clean&~correct).sum()),"wrong_to_correct":int((~correct_clean&correct).sum()),"both_correct":int((correct_clean&correct).sum()),"both_wrong":int((~correct_clean&~correct).sum()),"per_image_correct_change":(correct.astype(int)-correct_clean.astype(int)).tolist()}}
        candidates[candidate_id]={"label":candidate_id.replace("_"," "),"family":"baseline" if candidate_id=="baseline" else "logistic" if candidate_id.startswith("logistic") else "svm","augmentation":candidate_id.endswith("augmented"),"model_params":{},"training":{},"scenarios":scenarios}
    return {"schema_version":"shiftlab-results/1","stage":"validation","protocol":{"fixture":True},"provenance":{"fixture":True},"splits":{"train":{"indices":[100,101]},"validation":{"indices":[0,1,2,3]},"test":{"indices":[200,201]}},"evaluation":{"split":"validation","row_ids":[0,1,2,3],"y_true":true.tolist(),"scenario_inputs":{}},"candidates":candidates}


@pytest.fixture
def ui_mocks(payload,monkeypatch):
    import shiftlab.results as results
    import ui.views as views

    calls={"load":[],"images":[],"downloads":{}}
    def load(path):
        calls["load"].append(str(path))
        return deepcopy(payload)
    def validate(value):
        if value.get("schema_version")!="shiftlab-results/1":
            raise ValueError("Archivo incompatible")
    def export(value):
        validate(value)
        return json.dumps(value)
    def pair(value,scenario,row_id):
        calls["images"].append((value,scenario,row_id))
        original=np.arange(64,dtype=float).reshape(8,8)/63
        return original,original.copy() if scenario=="clean" else .7*original+.15
    original_download=st.download_button
    def download(*args,**kwargs):
        calls["downloads"][kwargs["key"]]=kwargs.get("data",args[1] if len(args)>1 else None)
        return original_download(*args,**kwargs)
    monkeypatch.setattr(results,"load_results",load)
    monkeypatch.setattr(results,"validate_results",validate)
    monkeypatch.setattr(results,"export_results",export)
    monkeypatch.setattr(views,"image_pair",pair)
    monkeypatch.setattr(st,"download_button",download)
    return calls


def loaded_app():
    app=AppTest.from_file(str(APP),default_timeout=30).run()
    app.text_input(key="results_path").set_value("artifacts/validation/fixture.json").run()
    app.button(key="load_results").click().run()
    assert not app.exception
    return app


def test_initial_state_is_empty_and_does_not_load_or_train(ui_mocks,monkeypatch):
    from sklearn.pipeline import Pipeline
    monkeypatch.setattr(Pipeline,"fit",lambda *args,**kwargs:pytest.fail("La vista no debe entrenar"))
    app=AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert ui_mocks["load"]==[]
    assert len(app.metric)==0
    assert len(app.get("download_button"))==0
    assert any("Carga un archivo" in item.value for item in app.info)


def test_loaded_summary_and_download_match_archived_object(payload,ui_mocks):
    app=loaded_app()
    assert app.session_state["results"]==payload
    assert json.loads(ui_mocks["downloads"]["download_results"])==payload
    table=app.dataframe[0].value
    assert table.loc[1,"Accuracy condición"]==payload["candidates"]["logistic_clean"]["scenarios"]["clean"]["metrics"]["accuracy"]
    assert any("Validación" in item.value for item in app.markdown)
    assert len(app.get("plotly_chart"))==2


def test_navigation_and_images_use_the_selected_archived_ids(payload,ui_mocks):
    app=loaded_app()
    scenario=next(key for key in SCENARIOS if key.startswith("noise_"))
    app.selectbox(key="scenario_selector").set_value(scenario).run()
    app.radio(key="view_selector").set_value("Explorador").run()
    assert not app.exception
    app.selectbox(key="image_selector").set_value(2).run()
    source,used_scenario,row_id=ui_mocks["images"][-1]
    assert source==payload
    assert used_scenario==scenario and row_id==2
    expected=payload["candidates"]["logistic_clean"]["scenarios"][scenario]["y_pred"][2]
    assert app.metric[1].value==str(expected)
    app.selectbox(key="candidate_selector").set_value("svm_clean").run()
    app.radio(key="view_selector").set_value("Análisis de errores").run()
    assert not app.exception
    assert len(app.get("plotly_chart"))==1
    assert len(ui_mocks["load"])==1


def test_changing_file_clears_previous_results_and_download(ui_mocks):
    app=loaded_app()
    app.text_input(key="results_path").set_value("artifacts/otro.json").run()
    assert not app.exception
    assert "results" not in app.session_state
    assert len(app.get("download_button"))==0
    assert len(ui_mocks["load"])==1


def test_corrupted_session_object_is_rejected_before_display(ui_mocks):
    app=loaded_app()
    app.session_state["results"]={"schema_version":"otra-version"}
    app.run()
    assert not app.exception
    assert "results" not in app.session_state
    assert any("incompatible" in item.value for item in app.error)
    assert len(app.get("download_button"))==0


def test_invalid_path_is_explained_without_reading_outside_project(ui_mocks):
    app=AppTest.from_file(str(APP)).run()
    app.text_input(key="results_path").set_value("../otro-proyecto/resultados.json").run()
    app.button(key="load_results").click().run()
    assert not app.exception
    assert ui_mocks["load"]==[]
    assert any("dentro del proyecto" in item.value for item in app.error)


def test_graph_values_correspond_to_archived_metrics_and_matrix(payload):
    from ui.views import comparison_chart,confusion_chart
    scenario=next(key for key in SCENARIOS if key.startswith("noise_"))
    figure=comparison_chart(payload,scenario,"macro_f1")
    assert list(figure.data[1].y)==[candidate["scenarios"][scenario]["metrics"]["macro_f1"] for candidate in payload["candidates"].values()]
    matrix=payload["candidates"]["logistic_clean"]["scenarios"][scenario]["confusion_matrix"]
    assert np.array_equal(confusion_chart(matrix).data[0].z,matrix)


def test_curve_ticks_wrap_condition_and_keep_full_hover_labels(payload):
    from ui.views import curve_chart,scenario_label
    figure=curve_chart(payload,"logistic_clean","noise","accuracy")
    assert all("<br>" in tick for tick in figure.data[0].x[1:])
    ids=["clean"]+[key for key in SCENARIOS if key.startswith("noise_")]
    assert list(figure.data[0].customdata)==[scenario_label(key) for key in ids]
    assert list(figure.data[0].y)==[payload["candidates"]["logistic_clean"]["scenarios"][key]["metrics"]["accuracy"] for key in ids]
