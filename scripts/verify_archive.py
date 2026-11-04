"""Recompute an archived ShiftLab evaluation without importing its engine.

Only simple Python arithmetic certifies metrics and paired changes. NumPy is
used solely to read exact image arrays when the companion NPZ is present.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

CANDIDATES = {'baseline','logistic_clean','logistic_augmented','svm_clean','svm_augmented'}
SCENARIOS = {'clean','noise_0025','noise_005','noise_010','blur_025','blur_045','blur_065','contrast_090','contrast_070','contrast_050'}
TOL = 1e-12


def equal_number(actual, expected):
    assert type(actual) in (float, int) and math.isfinite(actual)
    assert abs(actual-expected) <= TOL, (actual,expected)


def metrics(truth,prediction):
    assert len(truth)==len(prediction)>0
    assert all(type(v) is int and 0<=v<10 for v in truth+prediction)
    matrix=[[0 for _ in range(10)] for _ in range(10)]
    for actual,predicted in zip(truth,prediction):
        matrix[actual][predicted]+=1
    per_class=[]
    for c in range(10):
        tp=matrix[c][c]
        support=sum(matrix[c])
        predicted=sum(row[c] for row in matrix)
        precision=tp/predicted if predicted else 0.0
        recall=tp/support if support else 0.0
        f1=2*tp/(support+predicted) if support+predicted else 0.0
        per_class.append({'class':c,'precision':precision,'recall':recall,'f1':f1,'support':support})
    return {'accuracy':sum(matrix[c][c] for c in range(10))/len(truth),'macro_f1':sum(row['f1'] for row in per_class)/10},matrix,per_class


def verify(payload):
    assert set(payload['candidates'])==CANDIDATES
    evaluation=payload['evaluation']
    indices,truth=evaluation['row_ids'],evaluation['y_true']
    assert len(indices)==len(set(indices))==len(truth)>0
    assert all(type(i) is int and 0<=i<1797 for i in indices)
    condition_ids=set(evaluation['scenario_inputs'])
    assert condition_ids==SCENARIOS
    splits=payload['splits']
    split_rows=[set(splits[key]['indices']) for key in ('train','validation','test')]
    split_groups=[set(splits[key]['groups']) for key in ('train','validation','test')]
    assert set.union(*split_rows)==set(range(1797))
    for i in range(3):
        for j in range(i):
            assert not split_rows[i]&split_rows[j] and not split_groups[i]&split_groups[j]
    assert indices==splits[evaluation['split']]['indices']
    rows=[]
    for candidate_id,candidate in payload['candidates'].items():
        assert set(candidate['scenarios'])==condition_ids
        clean=candidate['scenarios']['clean']
        clean_metrics,_,_=metrics(truth,clean['y_pred'])
        for scenario_id,value in candidate['scenarios'].items():
            assert value['row_ids']==indices and value['y_true']==truth
            expected,cm,perclass=metrics(truth,value['y_pred'])
            assert value['confusion_matrix']==cm
            for metric,number in expected.items(): equal_number(value['metrics'][metric],number)
            assert len(value['per_class'])==10
            for actual,desired in zip(value['per_class'],perclass):
                for key,number in desired.items(): equal_number(actual[key],number)
            paired=value['paired_vs_clean']
            assert paired['unit']=='same_original_image'
            transitions=Counter()
            changes=[]
            for label,before,after in zip(truth,clean['y_pred'],value['y_pred']):
                a,b=before==label,after==label
                transitions[{(True,True):'both_correct',(True,False):'correct_to_wrong',(False,True):'wrong_to_correct',(False,False):'both_wrong'}[(a,b)]]+=1
                changes.append(int(b)-int(a))
            assert paired['per_image_correct_change']==changes
            for key in ('both_correct','correct_to_wrong','wrong_to_correct','both_wrong'):
                assert paired[key]==transitions[key]
            equal_number(paired['accuracy_difference'],expected['accuracy']-clean_metrics['accuracy'])
            equal_number(paired['accuracy_difference'],sum(changes)/len(changes))
            equal_number(paired['macro_f1_difference'],expected['macro_f1']-clean_metrics['macro_f1'])
            rows.append({'candidate':candidate_id,'scenario':scenario_id,**expected,'accuracy_difference':paired['accuracy_difference']})
        training=candidate['training']
        train=training['original_indices']
        assert train==splits['train']['indices']
        assert len(train)==len(set(train)) and not set(train)&set(indices)
        assert set(training['row_ids'])==set(train)
        augmented=candidate_id.endswith('_augmented')
        repetitions=4 if augmented else 1
        counts=Counter(training['row_ids'])
        assert set(counts.values())=={repetitions}
        assert len(training['weights'])==len(training['row_ids'])==len(training['versions'])
        assert all(weight==1/repetitions for weight in training['weights'])
        equal_number(sum(training['weights']),len(train))
        if augmented:
            for row in train:
                versions=[version for origin,version in zip(training['row_ids'],training['versions']) if origin==row]
                assert len(set(versions))==4 and 'clean' in versions
        for stage in training['fit_audit'].values():
            assert stage['n_rows']==len(training['row_ids'])
            assert stage['original_indices']==train
            equal_number(stage['weight_sum'],len(train))
        if candidate_id!='baseline':
            equal_number(training['fit_audit']['scaler']['n_samples_seen'],len(train))
    return rows


def verify_images(payload, archive_path):
    import numpy as np
    path=archive_path.parent/'evaluation-images.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==payload['evaluation']['archive']['sha256']
    with np.load(path,allow_pickle=False) as images:
        assert set(images.files)==SCENARIOS|{'row_ids','y_true'}
        assert images['row_ids'].tolist()==payload['evaluation']['row_ids']
        assert images['y_true'].tolist()==payload['evaluation']['y_true']
        for key in SCENARIOS:
            value=images[key]
            descriptor=payload['evaluation']['scenario_inputs'][key]
            assert list(value.shape)==descriptor['shape']==[len(images['row_ids']),8,8]
            assert np.isfinite(value).all() and (value>=-TOL).all() and (value<=1+TOL).all()
            digest=hashlib.sha256(np.ascontiguousarray(value,dtype=np.float64).tobytes()).hexdigest()
            assert digest==descriptor['sha256']
        clean=images['clean']
        assert np.array_equal(clean*16,np.rint(clean*16))
        groups=set(payload['splits'][payload['evaluation']['split']]['groups'])
        assert all(hashlib.sha256(np.asarray(image*16,dtype=np.uint8).tobytes()).hexdigest() in groups for image in clean)
    return {'npz_sha256':payload['evaluation']['archive']['sha256'],'verified_tensors':10,'image_bounds_tolerance':TOL,'same_inputs_for_all_candidates':True}


def verify_freeze(payload,archive_path):
    if payload['stage']!='final': return None
    config_path=archive_path.parent.parent/'frozen/config.json'
    frozen=json.loads(config_path.read_text())
    canonical=json.dumps(frozen,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest()==payload['provenance']['freeze_digest']
    assert frozen['splits']==payload['splits'] and frozen['protocol']==payload['protocol']
    assert frozen['created_at']<payload['created_at']
    project=Path(__file__).resolve().parents[1]
    expected=frozen['identity']['code_identity']['files']
    for name,digest in expected.items():
        assert hashlib.sha256((config_path.parent/'source'/name).read_bytes()).hexdigest()==digest
        assert hashlib.sha256((project/name).read_bytes()).hexdigest()==digest
    return {'files_verified':len(expected),'freeze_precedes_test':True,'snapshot_matches_current_experiment':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    payload=json.loads(args.archive.read_text(),parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    rows=verify(payload)
    images=verify_images(payload,args.archive)
    freeze=verify_freeze(payload,args.archive)
    mutant=deepcopy(payload)
    mutant['candidates']['baseline']['scenarios']['clean']['metrics']['accuracy']+=0.01
    try: verify(mutant)
    except AssertionError: mutation_detected=True
    else: raise AssertionError('Checker accepted deliberately altered metric')
    report={'at':datetime.now(timezone.utc).isoformat(),'archive':str(args.archive),'archive_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),'implementation':'Simple Python arithmetic; no shiftlab or sklearn imports. NumPy only reads archived image tensors.','tolerance':TOL,'passed':True,'evaluation_rows':len(payload['evaluation']['row_ids']),'candidate_scenario_checks':len(rows),'images':images,'freeze':freeze,'stale_metric_mutation_detected':mutation_detected,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Preserve existing verification evidence; use a new output path.')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
