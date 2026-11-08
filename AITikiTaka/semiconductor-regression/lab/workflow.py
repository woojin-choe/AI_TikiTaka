"""Five-input process workflow; isolated from legacy four-input projects."""
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
from sklearn.metrics import mean_absolute_error, r2_score
from threadpoolctl import threadpool_limits
from src.models import get_models
from lab.engine import split_data
from lab.extraction import CANON, NUMBERS, validate_values

INPUTS = NUMBERS[:5]
OUTPUTS = NUMBERS[5:]
LABELS = dict(zip(CANON, ['Lot', '웨이퍼', 'RF 전력', '압력', '온도', 'Si₂H₆ 유량', '증착 시간', '평균 두께', '반범위 불균일도']))
DEFAULTS = dict(thickness_mae=50., nonuniformity_mae=.5, thickness_min=2000., thickness_max=4000.,
                nonuniformity_max=10., temperature_unit='원본 단위', thickness_unit='원본 단위',
                retrain_count=20, retrain_days=30, experiment_nonuniformity_target=5.)


def units(settings):
    return dict(power='W', pressure='mTorr', temperature=settings['temperature_unit'], gas='sccm',
                time='s', thickness=settings['thickness_unit'], nonuniformity='%')


def validate_settings(settings):
    names = ['thickness_mae', 'nonuniformity_mae']
    if not np.isfinite([settings[n] for n in names]).all():
        raise ValueError('설정은 유한한 숫자여야 합니다.')
    if settings['thickness_mae'] <= 0 or settings['nonuniformity_mae'] <= 0:
        raise ValueError('허용 MAE는 양수여야 합니다.')
    target = experiment_target(settings)
    if not np.isfinite(target) or not 0 <= target <= 100:
        raise ValueError('불균일도 실험 목표는 0~100%여야 합니다.')


def experiment_target(settings):
    # Legacy 10% acceptance settings are not experimental goals.
    return float(settings.get('experiment_nonuniformity_target', 5.))


def summarize_bkm(frame, provenance):
    """Join by wafer identity, never row order or approximate process settings."""
    if 'recipe' not in provenance:
        return dict(available=False, reason='레시피 식별 정보가 없어 BKM을 구분할 수 없습니다.')
    keys = ['lot', 'wafer']
    if provenance[keys].isna().any().any() or provenance.duplicated(keys).any():
        raise ValueError('BKM 비교용 Lot·웨이퍼 식별자가 누락되거나 중복됐습니다.')
    joined = frame.merge(provenance[keys+['recipe']], on=keys, how='left', validate='one_to_one')
    bkm = joined[joined.recipe.astype(str).str.strip().str.casefold().eq('bkm')]
    if bkm.empty:
        return dict(available=False, reason='이 데이터에 측정값이 있는 BKM 기록이 없습니다.')
    values = bkm[OUTPUTS].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 0] <= 0).any() or (values[:, 1] < 0).any():
        raise ValueError('BKM 측정값을 확인해주세요.')
    return dict(available=True, count=len(bkm), lots=int(bkm.lot.nunique()),
                thickness=float(bkm.thickness.mean()), nonuniformity=float(bkm.nonuniformity.mean()))


def dataset_bkm(db, pid, did):
    meta = db.record(pid, did, 'dataset')['payload']
    frame = pd.read_csv(db.path(pid, meta['path']), dtype={'lot': str})
    provenance = pd.read_csv(db.path(pid, meta['provenance']), dtype={'lot': str})
    return dict(summarize_bkm(frame, provenance), dataset_id=did, name=meta['name'], units=meta['units'])


def compare_bkm(result, reference, target):
    result = result.drop(columns=['목표 충족 예상'], errors='ignore').copy()
    available = reference['available']
    result['BKM 대비 두께 변화 (%)'] = ((result.thickness/reference['thickness']-1)*100
                                       if available else None)
    result['BKM 대비 불균일도 변화 (%p)'] = (result.nonuniformity-reference['nonuniformity']
                                           if available else None)
    result['불균일도 실험 목표 (%)'] = target
    result['실험 목표 도달 예상'] = result.nonuniformity.between(0, target)
    result['BKM 비교 기준'] = (f"{reference['name']} · BKM {reference['count']}개"
                              if available else reference['reason'])
    return result


SOURCE_KINDS = {
    'unknown': '출처·수정 여부 미확인',
    'modified_field': '현장 자료를 일부 수정한 데이터',
    'measured': '수정하지 않은 실측 데이터',
    'synthetic': '합성·시연 데이터',
}
PROCESS_SCOPE = 'Si₂H₆ / N₂O / N₂ · N₂O 4,000 sccm · N₂ 3,000 sccm 고정'


def source_kind(meta):
    return 'synthetic' if meta.get('demo') else meta.get('source_kind', 'unknown')


def approve(db, pid, pending, temperature_unit, thickness_unit, demo=False,
            *, source_kind='unknown', scope_confirmed=False):
    if scope_confirmed is not True:
        raise ValueError('Si₂H₆·N₂O·N₂ 조합과 고정 보조가스 유량이 지원 범위와 같은지 확인해주세요.')
    if source_kind not in SOURCE_KINDS:
        raise ValueError('데이터 출처 구분을 확인해주세요.')
    if demo:
        source_kind = 'synthetic'
    demo = source_kind == 'synthetic'

    frame = pending['frame'].copy()
    validate_values(frame)
    settings = db.project(pid)['settings']
    if db.records(pid, 'dataset') and (settings['temperature_unit'], settings['thickness_unit']) != (temperature_unit, thickness_unit):
        raise ValueError('기존 데이터와 단위가 다릅니다. 단위가 다른 공정은 새 프로젝트로 등록해주세요.')
    if temperature_unit not in ['원본 단위', '°C', 'K'] or thickness_unit not in ['원본 단위', 'nm', 'µm']:
        raise ValueError('지원하지 않는 단위입니다.')
    digest = hashlib.sha256(pending['data']).hexdigest()
    if any(r['payload']['source_hash'] == digest for r in db.records(pid, 'dataset')):
        raise ValueError('이미 승인한 파일입니다. 기존 데이터셋을 선택해주세요.')
    rid = uuid.uuid4().hex
    path = db.write(pid, f'datasets/{rid}/training.csv', frame.to_csv(index=False).encode('utf-8-sig'))
    provenance = db.write(pid, f'datasets/{rid}/provenance.csv', pending['provenance'].to_csv(index=False).encode('utf-8-sig'))
    original = db.write(pid, f'datasets/{rid}/original'+('.xlsx' if pending['filename'].lower().endswith('.xlsx') else '.csv'), pending['data'])
    settings.update(temperature_unit=temperature_unit, thickness_unit=thickness_unit)
    db.settings(pid, settings)
    return db.add(pid, 'dataset', dict(path=path, provenance=provenance, original=original,
        name=pending['filename'], rows=len(frame), lots=frame.lot.nunique(), source_hash=digest,
        plan=pending['plan'], diagnostics=pending['diagnostics'], units=units(settings),
        demo=bool(demo), source_kind=source_kind, process_scope=PROCESS_SCOPE,
        scope_confirmed=True, extraction_log=pending['log']), rid)


def dataset_frame(db, pid, did, include_measurements=False):
    meta = db.record(pid, did, 'dataset')['payload']
    frame = pd.read_csv(db.path(pid, meta['path']), dtype={'lot': str})
    frame['record_id'] = [f'{did}:{i}' for i in range(len(frame))]
    if include_measurements:
        additions = [dict(r['payload']['inputs'], **r['payload']['actual'], lot=r['payload']['lot'],
                          wafer=r['payload']['wafer'], record_id=r['id']) for r in db.records(pid, 'measurement')]
        if additions:
            frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    validate_values(frame)
    frame['Lot_ID'] = frame.lot
    frame['content_id'] = [hashlib.sha256(json.dumps([round(float(v), 10) for v in row]).encode()).hexdigest()
                           for row in frame[NUMBERS].to_numpy()]
    return frame, meta


def score(actual, predicted):
    return {t: dict(MAE=float(mean_absolute_error(actual[:, i], predicted[:, i])),
                    R2=float(r2_score(actual[:, i], predicted[:, i]))) for i, t in enumerate(OUTPUTS)}


def train(db, pid, did, include_measurements=False, progress=None):
    settings = db.project(pid)['settings']
    validate_settings(settings)
    frame, dataset = dataset_frame(db, pid, did, include_measurements)
    parts, mode = split_data(frame)
    training, validation, testing = parts
    tolerances = np.array([settings['thickness_mae'], settings['nonuniformity_mae']])
    candidates = get_models()
    candidates['Baseline'] = DummyRegressor(strategy='mean')
    scores, models = {}, {}
    for name, estimator in candidates.items():
        if progress:
            progress(name+' 학습·검증 중')
        model = Pipeline([('scale', StandardScaler()), ('model', estimator)])
        with threadpool_limits(limits=2):
            model.fit(training[INPUTS].to_numpy(), training[OUTPUTS].to_numpy())
            pred = model.predict(validation[INPUTS].to_numpy())
        scores[name] = score(validation[OUTPUTS].to_numpy(), pred)
        scores[name]['normalized_MAE'] = float(np.mean([scores[name][t]['MAE']/tolerances[i] for i, t in enumerate(OUTPUTS)]))
        models[name] = model
    best = min((n for n in scores if n != 'Baseline'), key=lambda n: scores[n]['normalized_MAE'])
    eligible = all(scores[best][t]['MAE'] <= tolerances[i] for i, t in enumerate(OUTPUTS)) and scores[best]['normalized_MAE'] < scores['Baseline']['normalized_MAE']
    test_pred = models[best].predict(testing[INPUTS].to_numpy())
    rid = uuid.uuid4().hex
    buffer = io.BytesIO()
    joblib.dump(models[best], buffer)
    path = db.write(pid, f'models/{rid}/model.joblib', buffer.getvalue())
    table = testing[CANON].copy()
    for i, target in enumerate(OUTPUTS):
        table['predicted_'+target] = test_pred[:, i]
    test_path = db.write(pid, f'models/{rid}/test.csv', table.to_csv(index=False).encode('utf-8-sig'))
    db.write(pid, f'models/{rid}/snapshot.csv', frame.to_csv(index=False).encode('utf-8-sig'))
    return db.add(pid, 'model', dict(name=best, artifact=path, test_path=test_path, dataset_id=did,
        schema='five-input-half-range-v1', validation=scores, test=score(testing[OUTPUTS].to_numpy(), test_pred),
        eligible=bool(eligible), tolerances=tolerances.tolist(), units=dataset['units'], demo=dataset['demo'],
        source_kind=source_kind(dataset), process_scope=dataset.get('process_scope'),
        split_mode=mode, split={name: dict(rows=len(part), lots=part.lot.unique().tolist())
                               for name, part in zip(['학습', '검증', '테스트'], parts)},
        split_ids={name: part.record_id.tolist() for name, part in zip(['train', 'validation', 'test'], parts)},
        ranges={f: [float(training[f].min()), float(training[f].max())] for f in INPUTS},
        defaults={f: float(training[f].median()) for f in INPUTS},
        measurements_used=[r['id'] for r in db.records(pid, 'measurement')] if include_measurements else []), rid)


def predict(db, pid, mid, rows):
    meta = db.record(pid, mid, 'model')['payload']
    if meta.get('schema') != 'five-input-half-range-v1':
        raise ValueError('다른 입력·출력 정의의 모델입니다.')
    frame = pd.DataFrame(rows).reindex(columns=INPUTS).apply(pd.to_numeric, errors='coerce')
    if not 1 <= len(frame) <= 100 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('후보는 1~100개, 모든 입력은 유한한 숫자여야 합니다.')
    if (frame[['power', 'gas']] < 0).any().any() or (frame[['pressure', 'time']] <= 0).any().any():
        raise ValueError('전력·유량은 0 이상, 압력·시간은 양수여야 합니다.')
    unit = meta['units']['temperature']
    if (unit == '°C' and (frame.temperature < -273.15).any()) or (unit == 'K' and (frame.temperature < 0).any()):
        raise ValueError('온도가 절대영도보다 낮습니다.')
    model = joblib.load(db.path(pid, meta['artifact']))
    result = frame.copy()
    result[OUTPUTS] = model.predict(frame.to_numpy())
    settings = db.project(pid)['settings']
    result['학습 범위 밖'] = [', '.join(LABELS[f] for f in INPUTS if not meta['ranges'][f][0] <= row[f] <= meta['ranges'][f][1]) for row in frame.to_dict('records')]
    return compare_bkm(result, dataset_bkm(db, pid, meta['dataset_id']), experiment_target(settings))


def save_prediction(db, pid, mid, rows):
    result = predict(db, pid, mid, rows)
    meta = db.record(pid, mid, 'model')['payload']
    return db.add(pid, 'prediction', dict(model_id=mid, rows=result.to_dict('records'),
        bkm_reference=dataset_bkm(db, pid, meta['dataset_id']),
        experiment_nonuniformity_target=experiment_target(db.project(pid)['settings'])))


def add_measurement(db, pid, prediction_id, index, inputs, actual, lot, wafer, measured_at):
    saved = db.record(pid, prediction_id, 'prediction')['payload']
    if not 0 <= index < len(saved['rows']):
        raise ValueError('유효하지 않은 후보입니다.')
    if any(r['payload']['prediction_id'] == prediction_id and r['payload']['index'] == index for r in db.records(pid, 'measurement')):
        raise ValueError('이미 실제 결과를 등록한 후보입니다.')
    date = pd.to_datetime(measured_at, utc=True, errors='raise')
    if pd.isna(date) or date > pd.Timestamp.now(tz='UTC') or not lot.strip():
        raise ValueError('Lot와 현재/과거 실험 시각을 입력해주세요.')
    row = dict(inputs, **actual, lot=lot.strip(), wafer=int(wafer))
    validate_values(pd.DataFrame([row]))
    expected = predict(db, pid, saved['model_id'], [inputs]).iloc[0]
    return db.add(pid, 'measurement', dict(prediction_id=prediction_id, index=index,
        inputs={f: float(inputs[f]) for f in INPUTS}, actual={f: float(actual[f]) for f in OUTPUTS},
        lot=lot.strip(), wafer=int(wafer), measured_at=date.isoformat(), model_id=saved['model_id'],
        expected={t: float(expected[t]) for t in OUTPUTS}))


def reminders(db, pid):
    project = db.project(pid)
    if not project['active_model']:
        return []
    active = db.record(pid, project['active_model'], 'model')
    fresh = [r for r in db.records(pid, 'measurement') if r['id'] not in active['payload']['measurements_used']]
    messages = []
    if len(fresh) >= project['settings']['retrain_count']:
        messages.append(f'새 실제 측정 {len(fresh)}건: 재학습을 검토하세요.')
    if (datetime.now(timezone.utc)-datetime.fromisoformat(active['created'])).days >= project['settings']['retrain_days']:
        messages.append('설정한 재학습 검토 주기가 지났습니다.')
    return messages


def save_demo_measurement(db, pid, prediction_id, index):
    """A labeled UI simulation, never a training target or real measurement."""
    prediction = db.record(pid, prediction_id, 'prediction')['payload']
    if not 0 <= index < len(prediction['rows']):
        raise ValueError('유효하지 않은 후보입니다.')
    for record in db.records(pid, 'demo_measurement'):
        p = record['payload']
        if p['prediction_id'] == prediction_id and p['index'] == index:
            return record['id']
    row = prediction['rows'][index]
    # Explicitly simulated offsets from the saved prediction: for UI practice only.
    simulated = dict(thickness=max(.000001, float(row['thickness'])*1.012),
                     nonuniformity=max(0., float(row['nonuniformity'])+.15))
    return db.add(pid, 'demo_measurement', dict(prediction_id=prediction_id, index=index,
        model_id=prediction['model_id'], inputs={f: float(row[f]) for f in INPUTS},
        simulated=simulated, lot='DEMO-'+prediction_id[:6].upper(), wafer=index+1,
        measured_at=datetime.now(timezone.utc).isoformat(), synthetic=True,
        generation='saved_prediction_offset', training_eligible=False))
