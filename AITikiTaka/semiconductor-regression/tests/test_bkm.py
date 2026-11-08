import json
import pandas as pd
import pytest
from lab import workflow as flow
from lab.store import Store


def test_bkm_joins_by_identity_and_excludes_other_recipes():
    frame = pd.DataFrame(dict(lot=['A','A','B'], wafer=[1,2,1],
                              thickness=[3000.,3200.,9000.], nonuniformity=[8.,10.,2.]))
    provenance = pd.DataFrame(dict(lot=['B','A','A'], wafer=[1,2,1], recipe=['OTHER',' BKM ','BKM']))
    ref = flow.summarize_bkm(frame, provenance)
    assert ref == dict(available=True, count=2, lots=1, thickness=3100., nonuniformity=9.)
    result = flow.compare_bkm(pd.DataFrame(dict(thickness=[3255.,10000.,3000.],
                                               nonuniformity=[5.,5.01,-1.])), dict(ref,name='source'), 5.)
    assert result['실험 목표 도달 예상'].tolist() == [True,False,False]
    assert result.iloc[0]['BKM 대비 두께 변화 (%)'] == pytest.approx(5.)
    assert result.iloc[0]['BKM 대비 불균일도 변화 (%p)'] == -4.
    assert '목표 충족 예상' not in result


def test_missing_bkm_does_not_fabricate_reference():
    frame = pd.DataFrame(dict(lot=['A'], wafer=[1], thickness=[3000.], nonuniformity=[8.]))
    for provenance in [frame[['lot','wafer']], frame[['lot','wafer']].assign(recipe='Dummy')]:
        ref = flow.summarize_bkm(frame, provenance)
        assert not ref['available']
        result = flow.compare_bkm(frame, ref, 5.)
        assert result['BKM 대비 두께 변화 (%)'].iloc[0] is None
        json.dumps(result.to_dict('records'),allow_nan=False)
    with pytest.raises(ValueError,match='중복'):
        flow.summarize_bkm(frame, pd.concat([frame[['lot','wafer']].assign(recipe='BKM')]*2))


def test_legacy_settings_default_to_experiment_five():
    assert flow.experiment_target({'nonuniformity_max':10}) == 5.
    assert flow.experiment_target({'experiment_nonuniformity_target':4.}) == 4.
    with pytest.raises(ValueError):
        flow.validate_settings(dict(flow.DEFAULTS,experiment_nonuniformity_target=float('nan')))


def test_bkm_reference_stays_in_model_dataset(tmp_path):
    db = Store(tmp_path)
    pid = db.create_project('A',flow.DEFAULTS.copy())
    f=pd.DataFrame([['L',1,400.,2250.,400.,200.,30.,3150.,9.]],columns=flow.CANON)
    pending=dict(frame=f,provenance=f[['lot','wafer']].assign(recipe='BKM'),data=b'a',filename='a.csv',plan={},diagnostics=[],log='test')
    first=flow.approve(db,pid,pending,'원본 단위','원본 단위', scope_confirmed=True)
    second=flow.approve(db,pid,dict(pending,data=b'b',frame=f.assign(thickness=6000.)),'원본 단위','원본 단위', scope_confirmed=True)
    assert flow.dataset_bkm(db,pid,first)['thickness']==3150.
    assert flow.dataset_bkm(db,pid,second)['thickness']==6000.
    other=db.create_project('B',flow.DEFAULTS.copy())
    with pytest.raises(ValueError):flow.dataset_bkm(db,other,first)


def test_settings_ui_shows_experiment_target_and_no_acceptance_limits(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    monkeypatch.setenv('TIKITAKA_V2_DATA_DIR',str(tmp_path/'ui'))
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app_complete.py')).run()
    assert not app.exception
    labels={n.label:n.value for n in app.number_input}
    assert labels['불균일도 실험 목표 (%) · 낮을수록 좋음']==5.
    assert '목표 두께 하한' not in labels and '목표 두께 상한' not in labels
