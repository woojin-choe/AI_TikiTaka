from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unicodedata

import pytest

from lab.store import Store


def test_duplicate_names_and_normalization(tmp_path):
    db = Store(tmp_path)
    pid = db.create_project('  두번째공정  ', {})
    for name in ['두번째공정', ' 두번째공정 ', unicodedata.normalize('NFD', '두번째공정')]:
        with pytest.raises(ValueError, match='같은 이름'):
            db.create_project(name, {})
    assert db.projects()[0]['id'] == pid
    assert db.projects()[0]['name'] == '두번째공정'
    assert len(db.projects()) == 1
    db.create_project('세번째공정', {})
    with pytest.raises(ValueError, match='입력'):
        db.create_project('  ', {})


def test_concurrent_creation_does_not_duplicate(tmp_path):
    db = Store(tmp_path)
    def create(_):
        try:
            return db.create_project('공정 A', {})
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(create, range(2)))
    assert sum(value is not None for value in result) == 1
    assert len(db.projects()) == 1


def test_existing_duplicates_remain_accessible(tmp_path):
    db = Store(tmp_path)
    pid = db.create_project('공정 A', {})
    with db.connect() as conn:
        conn.execute("INSERT INTO projects SELECT 'legacy', name, created, settings, NULL FROM projects WHERE id=?", (pid,))
    with pytest.raises(ValueError, match='같은 이름'):
        db.create_project('공정 A', {})
    assert len(db.projects()) == 2
    assert db.project('legacy')['name'] == '공정 A'


def test_duplicate_name_shows_message_in_app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('TIKITAKA_V2_DATA_DIR', str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app_complete.py')).run()
    next(t for t in app.text_input if t.label == '프로젝트 이름').set_value(' 첫 공정 ')
    next(b for b in app.button if b.label == '프로젝트 만들기').click().run()
    assert not app.exception
    assert any('이미 같은 이름의 공정' in error.value for error in app.error)
    assert len(Store(tmp_path).projects()) == 1
