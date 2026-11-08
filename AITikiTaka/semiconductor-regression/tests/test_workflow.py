import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from lab.constants import ROOT, FIELDS, FEATURES, TARGETS, UNITS
from lab.store import Store
from lab import ingest, engine, llm


def settings():
    return dict(equipment='A', process='test', definition='test', thickness_min=90., thickness_max=110.,
                uniformity_min=90., thickness_mae=10., uniformity_mae=5., retrain_days=30, retrain_count=1)


@pytest.fixture
def project(tmp_path):
    db = Store(tmp_path)
    pid = db.create_project('test', settings())
    data = (ROOT / 'data/process_data.csv').read_bytes()
    raw = ingest.read_workbook(data, 'sample.csv')['CSV']
    mapping = ingest.suggest(raw, 1)
    frame = ingest.extract(raw, 'CSV', mapping, UNITS, 2, len(raw), group='A')
    did = ingest.save_import(db, pid, data, 'sample.csv', frame, {'mapping': mapping}, demo=True)
    return db, pid, did


def test_excel_positions_units_and_formulas():
    wb = Workbook()
    sheet = wb.active
    sheet.title = '실험'
    sheet.append(['보고서'])
    sheet.merge_cells('A1:F1')
    sheet.append(FIELDS)
    sheet.append([.3, .01, 523.15, .05, .1, .98])
    sheet.append(['=1+2', .01, 523.15, .05, .1, .98])
    buffer = io.BytesIO()
    wb.save(buffer)
    raw = ingest.read_workbook(buffer.getvalue(), 'test.xlsx')['실험']
    assert ingest.best_header(raw) == 2
    mapping = ingest.suggest(raw, 2)
    units = dict(zip(FIELDS, ['kW', 'Torr', 'K', 'slm', 'µm', 'fraction']))
    result = ingest.extract(raw, '실험', mapping, units, 3, 4)
    np.testing.assert_allclose(result.loc[0, FIELDS].to_numpy(dtype=float), [300, 10, 250, 50, 100, 98])
    assert result.loc[0, 'source_row'] == 3
    assert not result.loc[1, 'valid']
    assert 'RF_Power' in result.loc[1, 'issue']
    with pytest.raises(ValueError):
        ingest.extract(raw, '실험', mapping, {**units, 'Pressure': 'unknown'}, 3, 4)


def test_group_and_duplicate_isolation(project):
    db, pid, did = project
    frame, _ = engine.load_dataset(db, pid, did)
    parts, mode = engine.split_data(frame)
    assert mode == 'Lot 분리'
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert not set(parts[a].Lot_ID) & set(parts[b].Lot_ID)
        assert not set(parts[a].content_id) & set(parts[b].content_id)
    assert sum(map(len, parts)) == len(frame)


def test_end_to_end_and_project_isolation(project):
    db, pid, did = project
    mid = engine.train(db, pid, did)
    model = db.record(pid, mid, 'model')['payload']
    assert set(model['validation']) == {'LinearRegression', 'RandomForest', 'XGBoost', 'Baseline'}
    assert model['eligible']
    db.activate(pid, mid)
    row = model['defaults']
    result = engine.predict(db, pid, mid, [row])
    assert np.isfinite(result[TARGETS].to_numpy()).all()
    np.testing.assert_allclose(result[TARGETS], engine.predict(Store(db.root), pid, mid, [row])[TARGETS])
    prediction_id = engine.save_prediction(db, pid, mid, [row])
    actual_inputs = dict(row, Temperature=row['Temperature'] + 1)
    measurement_id = engine.add_measurement(db, pid, prediction_id, 0, actual_inputs,
                                            {'Thickness': 100., 'Uniformity': 95.}, 'new-lot',
                                            datetime.now(timezone.utc).isoformat())
    record = db.record(pid, measurement_id, 'measurement')['payload']
    assert record['conditions_changed']
    expected = engine.predict(db, pid, mid, [actual_inputs]).iloc[0]
    assert record['expected_at_actual_inputs']['Thickness'] == expected.Thickness
    assert engine.retraining_status(db, pid)
    with pytest.raises(ValueError, match='이미 등록'):
        engine.add_measurement(db, pid, prediction_id, 0, row, {'Thickness': 100., 'Uniformity': 95.},
                               'new', datetime.now(timezone.utc).isoformat())
    second = engine.train(db, pid, did)
    assert measurement_id in db.record(pid, second, 'model')['payload']['measurements_used']
    assert db.project(pid)['active_model'] == mid
    other = db.create_project('other', settings())
    with pytest.raises(ValueError):
        engine.predict(db, other, mid, [row])
    with pytest.raises(ValueError):
        db.write(pid, '../escape', b'bad')
    with pytest.raises(ValueError):
        engine.predict(db, pid, mid, [dict(row, RF_Power=float('nan'))])


def test_llm_schema_and_untrusted_location(monkeypatch):
    raw = ingest.read_workbook((ROOT / 'data/process_data.csv').read_bytes(), 'x.csv')['CSV']
    reply = {'sheet': 'CSV', 'header_row': 1, 'data_start_row': 2,
             'fields': {f: {'column': c, 'source_unit': UNITS[f], 'reason': '테스트', 'needs_confirmation': True}
                        for f, c in ingest.suggest(raw, 1).items()}, 'issues': []}
    captured = {}
    def fake_request(path, body, timeout):
        if path == '/api/show':
            return {'details': {'family': 'qwen3'}}
        captured.update(body)
        return {'message': {'content': json.dumps(reply)}}
    monkeypatch.setattr(llm, 'request', fake_request)
    result = llm.analyze(raw, 'CSV', 'test')
    assert result['fields']['RF_Power']['column'] == 'B'
    assert result['fields']['RF_Power']['source_unit'] is None  # no unit written in CSV header
    assert len(json.loads(captured['messages'][1]['content'])['sample']) <= 20
    reply['fields']['RF_Power']['column'] = 'ZZ'
    with pytest.raises(ValueError):
        llm.analyze(raw, 'CSV', 'test')
    with pytest.raises(ValueError, match='클라우드'):
        llm.analyze(raw, 'CSV', 'test', model='qwen3:cloud')


def test_time_split_is_forward(project):
    db, pid, did = project
    frame, _ = engine.load_dataset(db, pid, did)
    # No groups; independent records with chronological order.
    frame = frame.drop_duplicates('content_id').drop(columns='Lot_ID').reset_index(drop=True)
    frame['measured_at'] = pd.date_range('2025-01-01', periods=len(frame), freq='h').astype(str)
    parts, mode = engine.split_data(frame)
    assert '시간순' in mode
    assert parts[0].measured_at.max() < parts[1].measured_at.min()
    assert parts[1].measured_at.max() < parts[2].measured_at.min()


def test_ambiguous_headers_not_guessed():
    raw = pd.DataFrame([['설정온도', 'Temperature', 'RF_Power']], columns=['A', 'B', 'C'], index=[1])
    assert ingest.suggest(raw, 1)['Temperature'] is None


def test_missing_output_not_training(project):
    raw = ingest.read_workbook((ROOT / 'data/process_data.csv').read_bytes(), 'x.csv')['CSV']
    mapping = ingest.suggest(raw, 1)
    raw.loc[2, mapping['Thickness']] = ''
    result = ingest.extract(raw, 'CSV', mapping, UNITS, 2, len(raw))
    assert not result.iloc[0].valid


def test_small_dataset_rejected(project):
    db, pid, did = project
    frame, _ = engine.load_dataset(db, pid, did)
    with pytest.raises(ValueError, match='40행'):
        engine.split_data(frame.head(10))


def test_demo_layouts_extract_identical_data():
    truth = json.loads((ROOT / 'demo_data/expected_mapping.json').read_text())
    source = pd.read_csv(ROOT / 'data/process_data.csv')
    for filename, expected in truth.items():
        raw = ingest.read_workbook((ROOT / 'demo_data' / filename).read_bytes(), filename)[expected['sheet']]
        units = dict(UNITS)
        if filename.startswith('03'):
            units.update(Temperature='K', RF_Power='kW', Thickness='µm', Gas_Flow_Rate='slm', Pressure='Torr')
        mapping = {f: expected['mapping'][f] for f in FIELDS}
        extracted = ingest.extract(raw, expected['sheet'], mapping, units, expected['data_start_row'], len(raw))
        np.testing.assert_allclose(extracted[FIELDS], source[FIELDS], equal_nan=True)


def test_excel_percentage_storage_is_explicit():
    book = Workbook()
    sheet = book.active
    sheet.append(FIELDS)
    sheet.append([300, 10, 250, 50, 100, .98])
    sheet['F2'].number_format = '0.0%'
    buffer = io.BytesIO()
    book.save(buffer)
    raw = ingest.read_workbook(buffer.getvalue(), 'percent.xlsx')['Sheet']
    mapping = ingest.suggest(raw, 1)
    with pytest.raises(ValueError, match='fraction'):
        ingest.extract(raw, 'Sheet', mapping, UNITS, 2, 2)
    result = ingest.extract(raw, 'Sheet', mapping, {**UNITS, 'Uniformity': 'fraction'}, 2, 2)
    assert result.iloc[0].Uniformity == 98


def test_excel_without_dimension_metadata():
    import zipfile
    import re
    book = Workbook()
    book.active.append(FIELDS)
    book.active.append([300, 10, 250, 50, 100, 98])
    original = io.BytesIO()
    book.save(original)
    rewritten = io.BytesIO()
    with zipfile.ZipFile(original) as archive, zipfile.ZipFile(rewritten, 'w') as output:
        for item in archive.infolist():
            data = archive.read(item.filename)
            if item.filename == 'xl/worksheets/sheet1.xml':
                data = re.sub(rb'<dimension[^>]*/>', b'', data)
            output.writestr(item, data)
    raw = ingest.read_workbook(rewritten.getvalue(), 'no_dimensions.xlsx')['Sheet']
    assert raw.shape == (2, 6)
    assert raw.loc[2, 'F'] == 98
