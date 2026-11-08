from __future__ import annotations

import hashlib
import io
import json
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.models import get_models
from src.evaluation import evaluate_model
from lab.constants import FEATURES, TARGETS, FIELDS


def load_dataset(store, pid, dataset_id, include_measurements=True):
    dataset = store.record(pid, dataset_id, 'dataset')['payload']
    frame = pd.read_csv(store.path(pid, dataset['path']), dtype={'Lot_ID': str, 'record_id': str})
    frame = frame[frame.valid == True].copy()
    if include_measurements:
        additions = []
        for record in store.records(pid, 'measurement'):
            p = record['payload']
            additions.append(dict(p['inputs'], **p['actual'], record_id=record['id'],
                                  Lot_ID=p.get('lot') or None, measured_at=p['measured_at']))
        if additions:
            frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    if not np.isfinite(frame[FIELDS].to_numpy(dtype=float)).all():
        raise ValueError('학습 데이터에 유효하지 않은 숫자가 있습니다.')
    # Identical values cannot cross partitions, even if the source file was reformatted.
    frame['content_id'] = [hashlib.sha256(json.dumps([round(float(v), 10) for v in row]).encode()).hexdigest()
                           for row in frame[FIELDS].to_numpy()]
    return frame.reset_index(drop=True), dataset


def split_data(frame):
    if len(frame) < 40:
        raise ValueError('학습에는 유효한 측정 데이터가 최소 40행 필요합니다. 충분한 정확도를 보장하는 수는 아닙니다.')
    has_group = 'Lot_ID' in frame and frame.Lot_ID.notna().any()
    has_time = 'measured_at' in frame and frame.measured_at.notna().all()
    if has_group and (frame.Lot_ID.isna().any() or frame.Lot_ID.astype(str).str.strip().eq('').any()):
        raise ValueError('그룹 기반 데이터에 Lot ID가 없는 기록이 있습니다. 실제 결과 등록 시 Lot ID를 입력해주세요.')

    # Union identical content and same lot into one indivisible component.
    parent = list(range(len(frame)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(a)] = find(b)
    for column in (['content_id', 'Lot_ID'] if has_group else ['content_id']):
        first = {}
        for i, value in enumerate(frame[column]):
            if value in first:
                union(i, first[value])
            else:
                first[value] = i
    component = pd.Series([find(i) for i in range(len(frame))], index=frame.index)
    groups = component.unique()
    if len(groups) < 7:
        raise ValueError('독립적인 Lot/실험 묶음이 최소 7개 필요합니다.')
    if has_time:
        times = pd.to_datetime(frame.measured_at, utc=True, errors='raise')
        table = pd.DataFrame({'group': component, 'time': times}).groupby('group').time.agg(['min', 'max'])
        groups = table.sort_values('min', kind='stable').index.to_numpy()
        mode = '시간순 + Lot/동일 기록 분리'
    else:
        groups = np.random.default_rng(42).permutation(groups)
        mode = 'Lot 분리' if has_group else '무작위 분할 (시간·Lot 정보 없음)'
    a, b = int(len(groups) * .70), int(len(groups) * .85)
    parts = [frame[component.isin(g)].copy() for g in [groups[:a], groups[a:b], groups[b:]]]
    if has_time:
        for before, after in zip(parts, parts[1:]):
            if pd.to_datetime(before.measured_at, utc=True).max() >= pd.to_datetime(after.measured_at, utc=True).min():
                raise ValueError('Lot의 시간 범위가 겹쳐 시간순 분할이 불가능합니다. 공정별 Lot·시간 정보를 확인해주세요.')
    if len(parts[0]) < 20 or min(len(parts[1]), len(parts[2])) < 5:
        raise ValueError('분할 후 학습 20행, 검증·테스트 각각 5행 이상이 필요합니다.')
    return parts, mode


def metrics(actual, predicted):
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = evaluate_model(np.asarray(actual), np.asarray(predicted), TARGETS, '')
    return {k: float(v) if np.isfinite(v) else None for k, v in result.items()}


def train(store, pid, dataset_id, include_measurements=True, progress=None):
    project = store.project(pid)
    settings = project['settings']
    tolerances = np.array([settings['thickness_mae'], settings['uniformity_mae']], dtype=float)
    if not np.isfinite(tolerances).all() or (tolerances <= 0).any():
        raise ValueError('허용 MAE는 0보다 큰 유한한 값이어야 합니다.')
    frame, dataset = load_dataset(store, pid, dataset_id, include_measurements)
    (training, validation, testing), mode = split_data(frame)
    candidates = get_models()
    candidates['Baseline'] = DummyRegressor(strategy='mean')
    results, fitted = {}, {}
    for i, (name, estimator) in enumerate(candidates.items()):
        if progress:
            progress(i / len(candidates), f'{name} 학습 중')
        pipeline = Pipeline([('scale', StandardScaler()), ('model', estimator)])
        with threadpool_limits(limits=2):
            pipeline.fit(training[FEATURES].to_numpy(), training[TARGETS].to_numpy())
            predicted = pipeline.predict(validation[FEATURES].to_numpy())
        result = metrics(validation[TARGETS], predicted)
        result['normalized_MAE'] = float(np.mean([result[f'{t}_MAE'] / tolerances[i]
                                                  for i, t in enumerate(TARGETS)]))
        fitted[name], results[name] = pipeline, result
    best = min((name for name in results if name != 'Baseline'), key=lambda n: results[n]['normalized_MAE'])
    model = fitted[best]
    test_pred = model.predict(testing[FEATURES].to_numpy())
    test_metrics = metrics(testing[TARGETS], test_pred)
    eligible = (all(results[best][f'{t}_MAE'] <= tolerances[i] for i, t in enumerate(TARGETS))
                and results[best]['normalized_MAE'] < results['Baseline']['normalized_MAE'])
    comparisons = None
    comparison_note = '기존 활성 모델이 없습니다.'
    if project['active_model']:
        old = store.record(pid, project['active_model'], 'model')['payload']
        seen = set(old['seen_content'])
        seen_groups = set(old.get('seen_groups', []))
        overlap = bool(set(testing.content_id) & seen)
        if 'Lot_ID' in testing:
            overlap = overlap or bool(set(testing.Lot_ID.dropna().astype(str)) & seen_groups)
        if not overlap:
            incumbent = joblib.load(store.path(pid, old['artifact']))
            comparisons = metrics(testing[TARGETS], incumbent.predict(testing[FEATURES].to_numpy()))
            comparison_note = '동일 테스트 데이터 비교. 기존 모델의 학습·검증 기록/그룹과 겹치지 않습니다.'
        else:
            comparison_note = '기존 모델이 본 데이터/그룹과 겹쳐 공정한 직접 비교를 생략했습니다.'
    rid = uuid.uuid4().hex
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    artifact = store.write(pid, f'models/{rid}/model.joblib', buffer.getvalue())
    snapshot = store.write(pid, f'models/{rid}/training_snapshot.csv', frame.to_csv(index=False).encode())
    test_table = testing[FEATURES + TARGETS + ['record_id']].copy()
    for i, target in enumerate(TARGETS):
        test_table['predicted_' + target] = test_pred[:, i]
    test_path = store.write(pid, f'models/{rid}/test_predictions.csv', test_table.to_csv(index=False).encode())
    seen_data = pd.concat([training, validation])
    payload = {
        'name': best, 'dataset_id': dataset_id, 'demo': dataset.get('demo', False),
        'artifact': artifact, 'snapshot': snapshot, 'test_predictions': test_path,
        'validation': results, 'test': test_metrics, 'eligible': bool(eligible),
        'tolerances': tolerances.tolist(), 'split_mode': mode,
        'split_ids': {name: part.record_id.tolist() for name, part in zip(['train', 'validation', 'test'], [training, validation, testing])},
        'seen_content': seen_data.content_id.tolist(),
        'seen_groups': seen_data.Lot_ID.dropna().astype(str).unique().tolist() if 'Lot_ID' in seen_data else [],
        'ranges': {f: [float(training[f].min()), float(training[f].max())] for f in FEATURES},
        'defaults': {f: float(training[f].median()) for f in FEATURES},
        'incumbent_test': comparisons, 'comparison_note': comparison_note,
        'measurements_used': [r['id'] for r in store.records(pid, 'measurement')] if include_measurements else [],
    }
    store.write(pid, f'models/{rid}/report.json', json.dumps(payload, ensure_ascii=False, indent=2).encode())
    store.add(pid, 'model', payload, rid)
    if progress:
        progress(1.0, '학습 완료')
    return rid


def predict(store, pid, model_id, inputs):
    record = store.record(pid, model_id, 'model')
    metadata = record['payload']
    frame = pd.DataFrame(inputs)[FEATURES].apply(pd.to_numeric, errors='coerce')
    if frame.empty or len(frame) > 100:
        raise ValueError('조건 후보는 1~100개까지 입력해주세요.')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('입력은 모두 유한한 숫자여야 합니다.')
    if (frame[['RF_Power', 'Gas_Flow_Rate']] < 0).any().any() or (frame.Pressure <= 0).any() or (frame.Temperature < -273.15).any():
        raise ValueError('전력·유량은 0 이상, 압력은 0 초과, 온도는 절대영도 이상이어야 합니다.')
    model = joblib.load(store.path(pid, metadata['artifact']))
    result = frame.copy()
    result[TARGETS] = model.predict(frame.to_numpy())
    result['범위 확인'] = [' / '.join(f for f in FEATURES if not metadata['ranges'][f][0] <= row[f] <= metadata['ranges'][f][1])
                           for row in frame.to_dict('records')]
    settings = store.project(pid)['settings']
    result['목표 충족 예상'] = result.Thickness.between(settings['thickness_min'], settings['thickness_max']) & result.Uniformity.between(settings['uniformity_min'], 100)
    return result


def save_prediction(store, pid, model_id, rows):
    # Recompute server-side; never persist client-supplied predictions as truth.
    result = predict(store, pid, model_id, rows)
    return store.add(pid, 'prediction', {'model_id': model_id, 'rows': result.to_dict('records')})


def add_measurement(store, pid, prediction_id, index, inputs, actual, lot, measured_at, note=''):
    prediction = store.record(pid, prediction_id, 'prediction')['payload']
    if not 0 <= index < len(prediction['rows']):
        raise ValueError('선택한 후보가 없습니다.')
    if any(r['payload']['prediction_id'] == prediction_id and r['payload']['index'] == index
           for r in store.records(pid, 'measurement')):
        raise ValueError('이 후보의 실제 결과가 이미 등록되어 있습니다. 반복 실험은 후보를 새로 저장해주세요.')
    timestamp = pd.to_datetime(measured_at, utc=True, errors='raise')
    if pd.isna(timestamp) or timestamp > pd.Timestamp.now(tz='UTC'):
        raise ValueError('실제 실험 시각은 현재 또는 과거여야 합니다.')
    row = predict(store, pid, prediction['model_id'], [inputs]).iloc[0]
    values = np.array([actual[t] for t in TARGETS], dtype=float)
    if not np.isfinite(values).all() or values[0] <= 0 or not 0 <= values[1] <= 100:
        raise ValueError('실제 두께는 양수, 균일도는 0~100%여야 합니다.')
    payload = {'prediction_id': prediction_id, 'index': index, 'model_id': prediction['model_id'],
               'inputs': {f: float(inputs[f]) for f in FEATURES}, 'actual': {t: float(actual[t]) for t in TARGETS},
               'expected_at_actual_inputs': {t: float(row[t]) for t in TARGETS},
               'lot': lot.strip(), 'measured_at': timestamp.isoformat(), 'note': note,
               'conditions_changed': any(float(inputs[f]) != float(prediction['rows'][index][f]) for f in FEATURES)}
    return store.add(pid, 'measurement', payload)


def retraining_status(store, pid):
    project = store.project(pid)
    if not project['active_model']:
        return []
    active = store.record(pid, project['active_model'], 'model')
    settings = project['settings']
    used = set(active['payload']['measurements_used'])
    measurements = store.records(pid, 'measurement')
    fresh = [r for r in measurements if r['id'] not in used]
    reasons = []
    if len(fresh) >= settings['retrain_count']:
        reasons.append(f'신규 실제 측정 {len(fresh)}건이 쌓였습니다.')
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(active['created'])).days
    if age >= settings['retrain_days']:
        reasons.append(f'활성 모델 학습 후 {age}일이 지났습니다.')
    recent = [r['payload'] for r in measurements if r['payload']['model_id'] == active['id']][-20:]
    if len(recent) >= 10:
        for t, threshold in zip(TARGETS, [settings['thickness_mae'], settings['uniformity_mae']]):
            mae = np.mean([abs(r['actual'][t] - r['expected_at_actual_inputs'][t]) for r in recent])
            if mae > threshold:
                reasons.append(f'최근 {len(recent)}건 {t} MAE {mae:.3f}가 설정 기준을 초과했습니다.')
    return reasons
