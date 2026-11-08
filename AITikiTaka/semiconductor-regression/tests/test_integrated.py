import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from lab import workflow as flow, extraction
from lab.store import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    db = Store(tmp_path)
    pid = db.create_project('five input test', flow.DEFAULTS.copy())
    frame = pd.read_csv(ROOT/'demo_validation/K_expected.csv')
    pending = dict(frame=frame, provenance=frame[['lot', 'wafer']], filename='test.csv',
                   data=frame.to_csv(index=False).encode(), plan={}, diagnostics=[], log='test')
    did = flow.approve(db, pid, pending, '°C', 'nm', True, scope_confirmed=True)
    return db, pid, did, pending


def test_five_inputs_model_and_direction(project):
    db, pid, did, _ = project
    mid = flow.train(db, pid, did)
    meta = db.record(pid, mid, 'model')['payload']
    assert meta['source_kind'] == 'synthetic'
    assert meta['process_scope'] == flow.PROCESS_SCOPE
    assert meta['eligible'] and 'time' in meta['ranges']
    parts = list(meta['split'].values())
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        assert not set(parts[a]['lots']) & set(parts[b]['lots'])
    db.activate(pid, mid)
    defaults = meta['defaults']
    predictions = flow.predict(db, pid, mid, [defaults, dict(defaults, time=defaults['time']+5)])
    assert predictions.iloc[1].thickness > predictions.iloc[0].thickness
    settings = db.project(pid)['settings']
    settings['thickness_min'], settings['thickness_max'] = 1., 100000.
    settings['experiment_nonuniformity_target'] = float(predictions.iloc[0].nonuniformity)+.1
    db.settings(pid, settings)
    assert flow.predict(db, pid, mid, [defaults]).iloc[0]['실험 목표 도달 예상']
    settings['experiment_nonuniformity_target'] = 0.
    db.settings(pid, settings)
    assert not flow.predict(db, pid, mid, [defaults]).iloc[0]['실험 목표 도달 예상']
    other = db.create_project('other', flow.DEFAULTS.copy())
    with pytest.raises(ValueError):
        flow.predict(db, other, mid, [defaults])
    with pytest.raises(ValueError):
        flow.predict(db, pid, mid, [dict(defaults, time=-1)])
    with pytest.raises(ValueError):
        flow.predict(db, pid, mid, [{k:v for k,v in defaults.items() if k!='time'}])


def test_import_units_duplicate_and_small_lots(project):
    db, pid, did, pending = project
    with pytest.raises(ValueError, match='이미 승인'):
        flow.approve(db, pid, pending, '°C', 'nm', scope_confirmed=True)
    with pytest.raises(ValueError, match='단위가 다릅니다'):
        flow.approve(db, pid, dict(pending, data=b'other'), 'K', 'nm', scope_confirmed=True)
    frame, _ = flow.dataset_frame(db, pid, did)
    frame['Lot_ID'] = [f'L{i%4}' for i in range(len(frame))]
    with pytest.raises(ValueError, match='최소 7개'):
        flow.split_data(frame)


def test_missing_and_duplicate_values_fail(project):
    _, _, _, pending = project
    frame = pending['frame'].copy()
    frame.loc[0, 'thickness'] = np.nan
    with pytest.raises(ValueError, match='누락'):
        extraction.validate_values(frame)
    frame = pd.concat([pending['frame'], pending['frame'].iloc[:1]])
    with pytest.raises(ValueError, match='중복'):
        extraction.validate_values(frame)
    frame = pending['frame'].copy()
    frame.loc[0, 'lot'] = '  '
    with pytest.raises(ValueError, match='Lot'):
        extraction.validate_values(frame)


def test_app_pages_and_approval_gate(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('TIKITAKA_V2_DATA_DIR', str(tmp_path/'ui'))
    app = AppTest.from_file(str(ROOT/'app_complete.py'), default_timeout=30).run()
    assert not app.exception
    assert not any(r.label == '시작 방법' for r in app.radio)
    import io
    import streamlit as st
    from lab import condition_files
    class Upload(io.BytesIO):
        def __init__(self, data, name):
            super().__init__(data)
            self.name = name
    training_file = Upload((ROOT/'demo_uploads/K회사_새공정_업로드데모.xlsx').read_bytes(), 'K.xlsx')
    current = {'file': training_file}
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **kw: current['file'])
    app.run()
    assert not app.exception
    # Replay only a previously verified extraction at the UI boundary, without LLM/network.
    if not (ROOT/'demo_validation/K_extracted.csv').exists():
        pytest.skip('Run real local integration test before UI extraction replay')
    def replay(*args, **kwargs):
        plan = json.loads((ROOT/'demo_validation/K_plan.json').read_text())
        if (ROOT/'demo_validation/K_units.json').exists():
            plan['unit_suggestions'] = json.loads((ROOT/'demo_validation/K_units.json').read_text())
        return (pd.read_csv(ROOT/'demo_validation/K_extracted.csv'),
                plan,
                json.loads((ROOT/'demo_validation/K_diagnostics.json').read_text()),
                pd.read_csv(ROOT/'demo_validation/K_provenance.csv'))
    monkeypatch.setattr(extraction, 'extract', replay)
    next(b for b in app.button if b.label=='파일 분석하고 다음 →').click().run()
    assert not app.exception and not app.error
    if (ROOT/'demo_validation/K_units.json').exists():
        assert next(s for s in app.selectbox if s.label=='원본 온도 단위').value == '°C'
        assert next(s for s in app.selectbox if s.label=='원본 두께 단위').value == 'nm'
    next(b for b in app.button if b.label=='확인하고 모델 준비로 →').click().run()
    assert app.error  # Explicit approval is required.
    next(c for c in app.checkbox if c.label.startswith('항목의 의미')).check()
    next(b for b in app.button if b.label=='확인하고 모델 준비로 →').click().run()
    assert any('고정 보조가스' in e.value for e in app.error)
    next(c for c in app.checkbox if c.label.startswith('Si₂H₆·N₂O·N₂')).check()
    next(s for s in app.selectbox if s.label=='원본 온도 단위').set_value('°C')
    next(s for s in app.selectbox if s.label=='원본 두께 단위').set_value('nm')
    next(b for b in app.button if b.label=='확인하고 모델 준비로 →').click().run()
    assert not app.exception and not app.error
    assert any(b.label=='모델 학습 시작' for b in app.button)
    next(b for b in app.button if b.label=='모델 학습 시작').click().run(timeout=60)
    assert not app.exception and not app.error
    next(b for b in app.button if b.label=='이 모델로 가상 실험 →').click().run()
    assert not app.exception and not app.error
    assert not any(u.label == '가상 실험 조건 파일' for u in app.get('file_uploader'))
    labels = ['RF 전력 (W)', '압력 (mTorr)', '온도 (°C)', 'Si₂H₆ 유량 (sccm)', '증착 시간 (s)']
    assert all(any(n.label == label for n in app.number_input) for label in labels)
    time_input = next(n for n in app.number_input if n.label=='증착 시간 (s)')
    requested_time = time_input.value + 5
    time_input.set_value(requested_time).run()
    next(b for b in app.button if b.label=='결과 예측하기').click().run()
    assert not app.exception and not app.error
    assert any(m.label == '예상 불균일도 (%)' for m in app.metric)
    # Editing an input must invalidate the old prediction before it can be saved.
    power = next(n for n in app.number_input if n.label=='RF 전력 (W)')
    power.set_value(power.value + 1).run()
    assert not any(b.label=='후보 저장하고 결과 등록으로 →' for b in app.button)
    next(n for n in app.number_input if n.label=='압력 (mTorr)').set_value(-1).run()
    next(b for b in app.button if b.label=='결과 예측하기').click().run()
    assert app.error
    next(n for n in app.number_input if n.label=='압력 (mTorr)').set_value(400).run()
    next(b for b in app.button if b.label=='결과 예측하기').click().run()
    assert not app.exception and not app.error
    store = Store(tmp_path/'ui')
    project = store.projects()[0]
    model = store.record(project['id'], project['active_model'], 'model')['payload']
    saved = app.session_state['prediction_'+project['id']]
    assert saved['inputs'][0]['time'] == requested_time
    next(b for b in app.button if b.label=='후보 저장하고 결과 등록으로 →').click().run()
    next(b for b in app.button if b.label=='데모 결과 만들고 저장').click().run()
    assert any('체험 완료' in m.value for m in app.success)
    next(r for r in app.radio if r.label=='어떤 결과를 등록하나요?').set_value('실제 측정').run()
    assert next(n for n in app.number_input if n.label=='실제 평균 두께').value is None
    next(b for b in app.button if b.label=='실제 측정 저장').click().run()
    assert any('Lot ID' in e.value for e in app.error)
    next(r for r in app.radio if r.label=='어떤 결과를 등록하나요?').set_value('데모 체험').run()
    next(b for b in app.button if b.label=='다른 조건 비교하기 →').click().run()
    assert any(b.label=='결과 예측하기' for b in app.button)
    assert not app.exception


def test_demo_result_is_separate_and_idempotent(project):
    db, pid, did, _ = project
    mid = flow.train(db, pid, did)
    row = db.record(pid, mid, 'model')['payload']['defaults']
    rid = flow.save_prediction(db, pid, mid, [row])
    before, _ = flow.dataset_frame(db, pid, did, include_measurements=True)
    demo = flow.save_demo_measurement(db, pid, rid, 0)
    assert flow.save_demo_measurement(db, pid, rid, 0) == demo
    assert not db.records(pid, 'measurement')
    after, _ = flow.dataset_frame(db, pid, did, include_measurements=True)
    assert len(before) == len(after)
    assert db.record(pid, demo, 'demo_measurement')['payload']['training_eligible'] is False
