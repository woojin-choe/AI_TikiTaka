import pandas as pd
import pytest
from lab import workflow as flow
from lab.store import Store

@pytest.fixture
def source(tmp_path):
    db = Store(tmp_path)
    pid = db.create_project('scope', flow.DEFAULTS.copy())
    frame = pd.DataFrame([['L1',1,400,400,400,200,30,3100,8]], columns=flow.CANON)
    pending = dict(frame=frame, provenance=frame[['lot','wafer']], data=b'csv', filename='field.csv', plan={}, diagnostics=[], log='test')
    return db,pid,pending

def test_scope_required_before_any_save(source):
    db,pid,pending=source
    with pytest.raises(ValueError, match='고정 보조가스'):
        flow.approve(db,pid,pending,'°C','nm')
    assert db.records(pid,'dataset') == []

def test_modified_source_saved(source):
    db,pid,pending=source
    did=flow.approve(db,pid,pending,'°C','nm',source_kind='modified_field',scope_confirmed=True)
    meta=db.record(pid,did,'dataset')['payload']
    assert meta['source_kind']=='modified_field' and not meta['demo']
    assert meta['scope_confirmed'] and meta['process_scope']==flow.PROCESS_SCOPE

def test_invalid_source_rejected(source):
    db,pid,pending=source
    with pytest.raises(ValueError,match='출처'):
        flow.approve(db,pid,pending,'°C','nm',source_kind='invalid',scope_confirmed=True)
    assert not db.records(pid,'dataset')

def test_demo_cannot_be_relabeled_measured(source):
    db,pid,pending=source
    did=flow.approve(db,pid,pending,'°C','nm',True,source_kind='measured',scope_confirmed=True)
    assert flow.source_kind(db.record(pid,did,'dataset')['payload'])=='synthetic'

def test_legacy_source_is_not_assumed_measured():
    assert flow.source_kind({'demo':False})=='unknown'
    assert flow.source_kind({'demo':True})=='synthetic'
