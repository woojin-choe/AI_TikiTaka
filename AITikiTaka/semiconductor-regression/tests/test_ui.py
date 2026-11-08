from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_demo_to_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv('TIKITAKA_DATA_DIR', str(tmp_path))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    button(app, '샘플 데이터로 시작하기').click().run()
    assert not app.exception
    app.sidebar.radio[0].set_value('모델 학습').run()
    button(app, '학습 시작').click().run(timeout=60)
    assert not app.exception
    assert not app.error
    button(app, '이 모델 활성화 / 이전 버전으로 복원').click().run()
    assert not app.error
    app.sidebar.radio[0].set_value('가상 실험').run()
    button(app, '예상 품질 비교').click().run()
    assert not app.exception
    assert not app.error
    button(app, '이 후보와 예측 결과 저장').click().run()
    app.sidebar.radio[0].set_value('실제 결과').run()
    assert not app.exception
    next(t for t in app.text_input if t.label == 'Lot / 배치 ID').set_value('UI-LOT')
    next(c for c in app.checkbox if c.label == '예측값이 아닌 실제로 측정한 결과를 입력했습니다.').check()
    button(app, '실제 결과 저장').click().run()
    assert not app.error
    assert not app.exception
    app.sidebar.radio[0].set_value('설정').run()
    button(app, '설정 저장').click().run()
    assert not app.exception
    assert not app.error


def test_upload_ai_confirmation_and_save(tmp_path, monkeypatch):
    import io
    import streamlit as st
    from lab.constants import ROOT, FIELDS, UNITS
    from lab.store import Store
    from lab import ingest, llm
    monkeypatch.setenv('TIKITAKA_DATA_DIR', str(tmp_path))
    db = Store(tmp_path)
    db.create_project('업로드 시험', dict(equipment='A', process='증착', definition='설정온도와 질소 가스 유량',
        thickness_min=98., thickness_max=102., uniformity_min=98., thickness_mae=5., uniformity_mae=1.,
        retrain_days=30, retrain_count=20))
    class Upload(io.BytesIO):
        name = '02_한글_보고서.xlsx'
    file = Upload((ROOT / 'demo_data' / Upload.name).read_bytes())
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **kw: file)
    raw = ingest.read_workbook(file.getvalue(), file.name)['공정기록']
    answer = dict(sheet='공정기록', header_row=4, data_start_row=5, issues=[],
                  fields={f: dict(column=c, source_unit=UNITS[f], reason='표 제목 확인', needs_confirmation=True)
                          for f, c in ingest.suggest(raw, 4).items()})
    monkeypatch.setattr(llm, 'analyze', lambda *a, **kw: answer)
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.radio[0].set_value('데이터 등록').run()
    assert not app.exception
    button(app, 'AI로 항목 찾기').click().run()
    assert not app.error
    assert not app.exception
    next(c for c in app.checkbox if c.label.startswith('항목의 의미')).check().run()
    button(app, '확인한 데이터 저장').click().run()
    assert not app.error
    pid = db.projects()[0]['id']
    assert len(db.records(pid, 'dataset')) == 1
