"""Integrated five-input local process lab."""
import hashlib
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from lab.store import Store
from lab import extraction, workflow as flow, llm, condition_files

ROOT = Path(__file__).resolve().parent
DEMOS = ROOT/'demo_uploads'
st.set_page_config(page_title='TikiTaka · 공정 실험실', page_icon='◈', layout='wide')
st.markdown('<style>'+ (ROOT/'assets/theme.css').read_text(encoding='utf-8') +'</style>', unsafe_allow_html=True)

BODY = None
STEPS = ['파일 준비', '추출 확인', '모델 준비', '가상 실험', '결과 등록']

def go(pid, step):
    st.session_state['step_'+pid] = step
    if BODY is not None:
        BODY.empty()
    st.rerun()

def navigate(pid, step):
    st.session_state['step_'+pid] = step


def activate_model(db, pid, mid):
    db.activate(pid, mid)
    navigate(pid, 3)


def save_candidates(db, pid, mid, saved):
    token = repr((mid, saved['inputs'], flow.experiment_target(db.project(pid)['settings'])))
    if st.session_state.get('saved_token_'+pid) != token:
        st.session_state['saved_prediction_'+pid] = flow.save_prediction(db, pid, mid, saved['inputs'])
        st.session_state['saved_token_'+pid] = token
    navigate(pid, 4)


def start_file(pid):
    st.session_state.pop('pending_'+pid, None)
    navigate(pid, 0)


def back(pid, step, label='이전 단계'):
    st.button('← '+label, key='back_'+str(step), on_click=navigate, args=(pid, step))




@st.cache_resource
def database(path):
    return Store(path)


def display(frame, settings):
    units = flow.units(settings)
    return frame.rename(columns={f: flow.LABELS[f]+(f' ({units[f]})' if f in units else '') for f in flow.CANON if f in frame})


def show_bkm(reference):
    st.caption('비교 데이터: '+reference['name'])
    if not reference['available']:
        st.info(reference['reason']+' BKM 대비 변화는 표시하지 않습니다.')
        return
    a, b = st.columns(2)
    a.metric('BKM 평균 두께 ('+reference['units']['thickness']+')', f"{reference['thickness']:,.2f}")
    b.metric('BKM 평균 불균일도 (%)', f"{reference['nonuniformity']:.2f}")
    st.caption(f"BKM 기록 {reference['count']}개 · {reference['lots']}개 Lot. 두께 변화는 증가·감소를 뜻하며 품질 개선 판정이 아닙니다.")


def scope_notice():
    st.caption('지원 공정: '+flow.PROCESS_SCOPE)
    st.caption('Si₂H₆ 유량·전력·압력·온도·시간을 입력으로 사용합니다. 다른 가스나 보조가스 유량 변화는 현재 모델의 지원 범위가 아닙니다.')


def source_notice(meta):
    kind = flow.source_kind(meta)
    st.caption('학습 데이터 구분: '+flow.SOURCE_KINDS.get(kind, flow.SOURCE_KINDS['unknown']))
    if kind == 'modified_field':
        st.info('현장 자료를 수정한 데이터의 평가 결과입니다. 실제 생산 현장의 예측 성능을 의미하지 않습니다.')
    elif kind == 'unknown':
        st.info('이 데이터의 출처·수정 여부는 확인되지 않았습니다. 실측 원본으로 간주하지 않습니다.')
    elif kind == 'synthetic':
        st.info('합성·시연 데이터의 예측이며 현장 성능의 근거가 아닙니다.')


def upload(db, project):
    pid, settings = project['id'], project['settings']
    st.title('공정 데이터 불러오기')
    scope_notice()
    st.write('로컬 AI가 필요한 표와 항목을 찾습니다. 추출 결과를 확인한 뒤 학습에 사용하세요.')
    st.caption('입력: 전력 · 압력 · 온도 · Si₂H₆ 유량 · 증착 시간 / 출력: 평균 두께 · 반범위 불균일도')
    data, filename = None, None
    file = st.file_uploader('학습용 공정 엑셀 또는 CSV', type=['xlsx', 'csv'], key='file_'+pid)
    if file:
        data, filename = file.getvalue(), file.name
    with st.expander('지원 범위와 계산 정의'):
        st.write('이 프로젝트의 제공 자료는 현장 데이터를 일부 수정한 자료이며, 가스 조합은 고정이고 설정값은 임의로 정한 값입니다. 업계 최다 사용 레시피나 생산 합격 기준으로 취급하지 않습니다.')
        st.write('한 시트에 하나의 표를 지원합니다. 한 표로 정리된 기록 또는 조건표·웨이퍼 이력·측정표의 연결을 처리합니다. XLSX/CSV 20MB 이하, 시트 20개 이하입니다.')
        st.write('측정점 방식은 1~43번 지점과 THK1_1_TOP을 사용하며 GOF_1은 제외합니다. 수식 셀은 실행하지 않으므로 값으로 저장해주세요.')
        st.code('반범위 불균일도(%) = 100 × (최대 두께 − 최소 두께) / (2 × 평균 두께)')
        st.write('불균일도는 낮을수록 좋습니다. 회사의 지표가 이 정의와 같은지 확인해야 합니다. 온도·두께는 원본 단위를 유지합니다.')
    if data is None:
        return
    digest, key = hashlib.sha256(data).hexdigest(), 'pending_'+pid
    pending = st.session_state.get(key)
    if pending and pending['hash'] != digest:
        st.session_state.pop(key, None)
        pending = None
    analyzed = False
    if st.button('파일 분석하고 다음 →', type='primary'):
        st.session_state.pop(key, None)
        pending = None
        log = 'extractions/'+uuid.uuid4().hex
        started = time.monotonic()
        with st.status('로컬 Qwen3 14B로 분석 중 · 수 분 소요', expanded=True) as status:
            try:
                frame, plan, diagnostics, provenance = extraction.extract(data, filename, db.directory(pid)/log, progress=st.write)
                pending = dict(hash=digest, data=data, filename=filename, frame=frame, plan=plan,
                               diagnostics=diagnostics, provenance=provenance, log=log)
                st.session_state[key] = pending
                analyzed = True
                status.update(label=f'추출 완료 · {len(frame)}행 · {time.monotonic()-started:.0f}초', state='complete', expanded=False)
            except Exception as exc:
                status.update(label='분석을 완료하지 못했습니다', state='error')
                st.error(str(exc))
                st.write('필수 항목·단위·표 구성을 확인해주세요. 누락된 기록을 임의로 제거하거나 학습하지 않았습니다.')
                st.caption('연결 오류: ollama serve / 모델 없음: ollama pull qwen3:14b')
    if analyzed:
        go(pid, 1)
    if pending is None:
        return
    st.button('추출 결과 확인 →', type='primary', on_click=navigate, args=(pid, 1))


def approve_review(db, pid, pending, unit_key, known_demo):
    error_key = 'review_error_'+pid
    st.session_state.pop(error_key, None)
    if not st.session_state.get('confirmed_'+unit_key):
        st.session_state[error_key] = '원본과 결과를 확인한 뒤 확인 항목을 선택해주세요.'
        return
    try:
        did = flow.approve(db, pid, pending, st.session_state['temp_'+unit_key],
                           st.session_state['thick_'+unit_key],
                           demo=known_demo,
                           source_kind=st.session_state['source_'+unit_key],
                           scope_confirmed=st.session_state.get('scope_'+unit_key, False))
        st.session_state['dataset_'+pid] = did
        st.session_state.pop('pending_'+pid, None)
        st.session_state['step_'+pid] = 2
    except ValueError as exc:
        st.session_state[error_key] = str(exc)


def review(db, project):
    pid, settings = project['id'], project['settings']
    st.title('추출 데이터 확인')
    st.write('단위와 원본 근거를 확인하면 학습에 사용할 데이터로 저장합니다.')
    pending = st.session_state.get('pending_'+pid)
    if pending is None:
        records = db.records(pid, 'dataset')
        if not records:
            st.info('먼저 파일을 분석해주세요.')
            back(pid, 0, '파일 준비')
            return
        meta = records[-1]['payload']
        st.success('승인한 데이터: '+meta['name'])
        source_notice(meta)
        st.dataframe(display(pd.read_csv(db.path(pid, meta['path'])), settings), hide_index=True, use_container_width=True)
        st.button('모델 준비로 계속 →', type='primary', on_click=navigate, args=(pid, 2))
        back(pid, 0, '다른 파일 선택')
        return
    data, filename, digest = pending['data'], pending['filename'], pending['hash']
    key, examples = 'pending_'+pid, sorted(DEMOS.glob('*.xlsx'))
    frame = pending['frame']
    a, b, c = st.columns(3)
    a.metric('추출 기록', f'{len(frame)}행')
    b.metric('Lot', frame.lot.nunique())
    c.metric('입력 / 출력', '5개 / 2개')
    st.subheader('원본과 비교하고 승인하세요')
    # No confirmed units are implied before approval.
    st.dataframe(display(frame.head(12), dict(settings, temperature_unit='원본 단위', thickness_unit='원본 단위')), hide_index=True, use_container_width=True)
    st.caption(f'전체 {len(frame)}행 중 12행 미리보기 · 전체 데이터는 CSV로 내려받을 수 있습니다.')
    with st.expander('선택된 표·열과 원본 행 보기'):
        rows = []
        for table in pending['plan']['tables'].values():
            for field, column in table['columns'].items():
                rows.append(dict(표=table['sheet'], 항목=flow.LABELS.get(field, field), 원본열=column,
                                 제목행=table['header_row'], 시작행=table['data_start_row']))
            if table['point_columns']:
                rows.append(dict(표=table['sheet'], 항목='두께 측정점 43개', 원본열=', '.join(table['point_columns']),
                                 제목행=table['header_row'], 시작행=table['data_start_row']))
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.dataframe(pending['provenance'], hide_index=True, use_container_width=True)
        st.json(pending['diagnostics'])
    st.download_button('추출 CSV 다운로드', frame.to_csv(index=False).encode('utf-8-sig'), '학습용_입출력.csv', 'text/csv')
    if frame.lot.nunique() < 7:
        st.warning('추출은 완료됐지만 학습에는 독립적인 Lot/실험 묶음 최소 7개가 필요합니다. Lot를 임의로 쪼개지 말고 추가 실험 기록을 확보해주세요.')
    known_demo = any(hashlib.sha256(f.read_bytes()).hexdigest() == digest for f in examples)
    suggestions = pending['plan'].get('unit_suggestions', {})
    if not suggestions:
        st.info('이미 추출한 파일도 원본에서 단위를 자동 확인할 수 있습니다.')
        if st.button('원본에서 단위 자동 확인'):
            try:
                with st.spinner('로컬 AI가 원본 제목·주석의 단위 근거를 확인하고 있습니다…'):
                    suggestions = extraction.suggest_units(data, filename, pending['plan'],
                                                           db.directory(pid)/(pending['log']+'/unit_review'))
                    pending['plan']['unit_suggestions'] = suggestions
                    st.session_state[key] = pending
            except Exception as exc:
                st.error('단위 확인 실패: '+str(exc))
    for field, label in [('temperature', '온도'), ('thickness', '두께')]:
        suggestion = suggestions.get(field, {})
        if suggestion.get('unit'):
            st.success(f"{label}: {suggestion['unit']} · LLM이 찾은 원본 근거를 확인해 자동 선택했습니다.")
        elif suggestions:
            st.warning(label+': '+suggestion.get('reason', '원본 단위를 확인해주세요.'))
        for evidence in suggestion.get('evidence', []):
            st.caption(f"근거: {evidence['sheet']} · {evidence['cell']} — {evidence['text']}")
    temp_options, thick_options = ['원본 단위', '°C', 'K'], ['원본 단위', 'nm', 'µm']
    # Stable for a particular source/review; a later verified review updates defaults.
    unit_key = pid+digest+str(suggestions.get('temperature', {}).get('unit'))+str(suggestions.get('thickness', {}).get('unit'))
    with st.form('approve_'+pid):
        a, b = st.columns(2)
        temp = a.selectbox('원본 온도 단위', temp_options,
                           index=temp_options.index(suggestions.get('temperature', {}).get('unit') or '원본 단위'), key='temp_'+unit_key)
        thick = b.selectbox('원본 두께 단위', thick_options,
                            index=thick_options.index(suggestions.get('thickness', {}).get('unit') or '원본 단위'), key='thick_'+unit_key)
        st.caption('원본 근거가 명확하면 자동 선택합니다. 단위를 모르면 원본 단위로 유지하세요. 이 선택은 숫자 변환이 아닙니다.')
        st.selectbox('데이터 출처와 수정 여부', list(flow.SOURCE_KINDS),
                     index=3 if known_demo else 0, format_func=flow.SOURCE_KINDS.get,
                     disabled=known_demo, key='source_'+unit_key)
        st.checkbox('Si₂H₆·N₂O·N₂ 조합이며 N₂O 4,000 / N₂ 3,000 sccm 고정 조건임을 확인했습니다', key='scope_'+unit_key)
        st.caption('가스 종류와 보조가스 유량은 사용자가 원본에서 확인합니다. 자동 식별·검증한 결과가 아닙니다. 현장 제공 수정본은 해당 구분을 직접 선택하세요.')
        confirmed = st.checkbox('항목의 의미·단위·불균일도 계산 정의와 원본 행을 확인했습니다', key='confirmed_'+unit_key)
        st.form_submit_button('확인하고 모델 준비로 →', type='primary',
                              on_click=approve_review, args=(db, pid, pending, unit_key, known_demo))
    if st.session_state.get('review_error_'+pid):
        st.error(st.session_state['review_error_'+pid])

    back(pid, 0, '파일 준비')

def training(db, project):
    pid = project['id']
    st.title('모델 학습 및 평가')
    st.write('기존 회귀 모델 3종을 비교합니다. 같은 Lot이 학습·검증·테스트에 섞이지 않도록 나눕니다.')
    choices = {r['id']: r for r in db.records(pid, 'dataset')}
    if not choices:
        st.info('먼저 파일을 분석하고 추출 결과를 승인해주세요.')
        return
    did = st.selectbox('학습할 데이터', list(choices), index=len(choices)-1,
                       format_func=lambda k, choices=choices: f"{choices[k]['payload']['name']} · {choices[k]['payload']['rows']}행")
    st.session_state['dataset_'+pid] = did
    source_notice(choices[did]['payload'])
    include = st.checkbox('이 프로젝트의 새 실제 측정도 포함', value=False)
    if st.button('모델 학습 시작', type='primary'):
        with st.status('학습과 평가 중', expanded=True) as status:
            try:
                flow.train(db, pid, did, include, progress=st.write)
                status.update(label='학습 완료', state='complete', expanded=False)
            except Exception as exc:
                status.update(label='학습 보류', state='error')
                st.error(str(exc))
    choices = {r['id']: r for r in db.records(pid, 'model') if r['payload']['dataset_id'] == did}
    if not choices:
        back(pid, 1, '추출 확인')
        return
    mid = st.selectbox('모델 버전', list(choices), index=len(choices)-1,
                       format_func=lambda k, choices=choices: f"{choices[k]['created'][:19]} · {choices[k]['payload']['name']} · {k[:6]}")
    meta = choices[mid]['payload']
    st.subheader('검증 데이터로 모델 선택')
    scores = [dict(모델=n, 두께_MAE=m['thickness']['MAE'], 불균일도_MAE=m['nonuniformity']['MAE'], 비교점수=m['normalized_MAE']) for n, m in meta['validation'].items()]
    st.dataframe(pd.DataFrame(scores), hide_index=True, use_container_width=True)
    st.caption('MAE는 평균 절대 오차입니다. 두께는 확인한 원본 단위, 불균일도 오차는 퍼센트포인트입니다. Baseline은 학습 평균값만 예측합니다.')
    a, b = st.columns(2)
    a.metric('최종 테스트 · 두께 MAE', f"{meta['test']['thickness']['MAE']:.3f}")
    b.metric('최종 테스트 · 불균일도 MAE', f"{meta['test']['nonuniformity']['MAE']:.3f}")
    st.caption(f"학습 당시 허용 MAE: 두께 {meta['tolerances'][0]}, 불균일도 {meta['tolerances'][1]}. 테스트는 모델 선택에 사용하지 않습니다.")
    with st.expander('분할 내역·실제값과 예측값'):
        st.json(meta['split'])
        st.dataframe(pd.read_csv(db.path(pid, meta['test_path'])), hide_index=True, use_container_width=True)
        st.download_button('테스트 결과 다운로드', db.path(pid, meta['test_path']).read_bytes(), '모델_테스트결과.csv')
    if meta['eligible']:
        st.success('학습 당시 검증 오차 기준을 충족하고 평균값 기준 모델보다 개선됐습니다.')
        st.button('이 모델로 가상 실험 →', type='primary', on_click=activate_model, args=(db, pid, mid))
    else:
        st.warning('검증 기준 미달로 활성화할 수 없습니다. 데이터를 보완하고 공정에 맞는 허용 오차를 설정해주세요.')

    back(pid, 1, '추출 확인')

def experiments(db, project):
    pid, mid = project['id'], project['active_model']
    st.title('공정 조건 비교')
    scope_notice()
    st.caption('실제 장비 실험 전 후보 비교를 돕습니다. 실험 횟수 감소 효과와 현장 정확도는 별도 검증이 필요합니다.')
    if not mid:
        st.info('먼저 모델을 학습하고 검증 결과를 확인한 뒤 활성화해주세요.')
        return
    meta = db.record(pid, mid, 'model')['payload']
    source_notice(meta)
    if meta['demo']:
        st.warning('합성 데이터로 학습한 시연용 예측입니다.')
    reference = flow.dataset_bkm(db, pid, meta['dataset_id'])
    show_bkm(reference)
    target = flow.experiment_target(project['settings'])
    st.info(f'불균일도 {target:g}% 이하를 실험 목표로 비교합니다. 산업 공통 합격 기준이 아닙니다.')
    st.write('전력·압력·온도·가스 유량·증착 시간을 입력하면 평균 두께와 불균일도를 예측합니다.')
    st.caption('초기값은 학습 구간 중앙값입니다. 시험할 조건에 맞게 5개 값을 모두 수정할 수 있습니다.')
    candidates = {}
    columns = st.columns(2)
    for i, field in enumerate(['power', 'pressure', 'temperature', 'gas', 'time']):
        low, high = meta['ranges'][field]
        candidates[field] = columns[i % 2].number_input(
            f'{flow.LABELS[field]} ({meta["units"][field]})',
            value=float(meta['defaults'][field]), format='%.2f',
            key='manual_'+pid+'_'+mid+'_'+field,
            help=f'학습 데이터 범위: {low:g} ~ {high:g}')
    prediction_key = 'prediction_'+pid
    digest = repr((mid, candidates))
    saved = st.session_state.get(prediction_key)
    if saved and (saved.get('source_hash') != digest or saved['model_id'] != mid):
        st.session_state.pop(prediction_key, None)
    if st.button('결과 예측하기', type='primary'):
        try:
            result = flow.predict(db, pid, mid, [candidates])
            st.session_state[prediction_key] = dict(model_id=mid, source_hash=digest,
                inputs=[candidates], result=result)
        except Exception as exc:
            st.session_state.pop(prediction_key, None)
            st.error(str(exc))
    saved = st.session_state.get(prediction_key)
    if saved and saved['model_id'] == mid and saved.get('source_hash') == digest:
        # Refresh comparison labels when settings change without rerunning inference.
        result = flow.compare_bkm(saved['result'], reference, target)
        st.subheader('예측 결과')
        left, right = st.columns(2)
        left.metric('예상 평균 두께 ('+meta['units']['thickness']+')', f"{result.iloc[0].thickness:,.2f}")
        right.metric('예상 불균일도 (%)', f"{result.iloc[0].nonuniformity:.2f}")
        st.dataframe(display(result, project['settings']), hide_index=True, use_container_width=True)
        st.caption('BKM 대비 불균일도 변화가 음수이면 BKM보다 낮을 것으로 예측합니다. 실험 목표 도달은 모든 위치의 두께 규격 충족을 뜻하지 않습니다.')
        if result['학습 범위 밖'].astype(bool).any():
            st.warning('학습 범위를 벗어난 조건이 있습니다. 범위 안이라도 예측 정확도가 보장되지는 않습니다.')
        if (result[flow.OUTPUTS] < 0).any().any():
            st.warning('물리적으로 해석하기 어려운 음수 예측이 있습니다. 실행 전에 검토하세요.')
        st.download_button('예측 결과 CSV 다운로드', display(result, project['settings']).to_csv(index=False).encode('utf-8-sig'), '가상실험_예측.csv')
        st.button('후보 저장하고 결과 등록으로 →', type='primary',
                  on_click=save_candidates, args=(db, pid, mid, saved))

    back(pid, 2, '모델 준비')

def measurements(db, project):
    pid = project['id']
    st.title('측정 결과 기록')
    st.write('데모에서는 버튼 한 번으로 등록 흐름을 체험할 수 있습니다. 실제 실험은 측정한 값을 입력하세요.')
    predictions = {r['id']: r for r in db.records(pid, 'prediction')}
    if not predictions:
        st.info('「가상 실험」에서 조건 후보를 먼저 저장해주세요.')
        return
    ids = list(predictions)
    preferred = st.session_state.get('saved_prediction_'+pid, ids[-1])
    rid = st.selectbox('저장한 실험', ids, index=ids.index(preferred) if preferred in ids else len(ids)-1,
                       format_func=lambda k: predictions[k]['created'][:19]+' · '+k[:6])
    rows = predictions[rid]['payload']['rows']
    index = st.selectbox('후보', range(len(rows)), format_func=lambda i: f'후보 {i+1}')
    meta = db.record(pid, predictions[rid]['payload']['model_id'], 'model')['payload']
    mode = st.radio('어떤 결과를 등록하나요?', ['데모 체험', '실제 측정'], horizontal=True,
                    index=0 if meta['demo'] else 1, key='result_mode_'+pid)
    if mode == '데모 체험':
        with st.container(border=True):
            st.subheader('측정 없이 전체 흐름 체험하기')
            st.write('저장한 예측값에 작은 차이를 더해 시연 결과를 만듭니다. Lot·웨이퍼·시각도 자동으로 채웁니다.')
            st.caption('합성 결과로 별도 보관하며 실제 측정과 모델 학습에는 포함하지 않습니다.')
            if st.button('데모 결과 만들고 저장', type='primary'):
                flow.save_demo_measurement(db, pid, rid, index)
            demos = [r for r in db.records(pid, 'demo_measurement')
                     if r['payload']['prediction_id'] == rid and r['payload']['index'] == index]
            if demos:
                demo = demos[-1]['payload']
                st.success('체험 완료 · 파일부터 결과 등록까지 한 바퀴 진행했습니다.')
                a, b = st.columns(2)
                a.metric('시연 평균 두께', f"{demo['simulated']['thickness']:.2f} {meta['units']['thickness']}")
                b.metric('시연 불균일도', f"{demo['simulated']['nonuniformity']:.2f}%")
                st.caption(f"합성 기록 · {demo['lot']} · 웨이퍼 {demo['wafer']} · 학습 제외")
                st.download_button('데모 결과 다운로드', pd.DataFrame([dict(demo['inputs'], **demo['simulated'],
                    lot=demo['lot'], wafer=demo['wafer'], synthetic=True, training_eligible=False)]).to_csv(index=False).encode('utf-8-sig'), '시연용_결과_학습제외.csv')
        a, b = st.columns(2)
        with a:
            st.button('다른 조건 비교하기 →', on_click=navigate, args=(pid, 3))
        with b:
            st.button('다른 파일로 시작하기 →', on_click=start_file, args=(pid,))
        return
    with st.form('measurement_'+pid+'_'+rid+'_'+str(index)):
        st.subheader('1. 실제 실행 조건')
        inputs = {field: col.number_input(flow.LABELS[field]+f" ({meta['units'][field]})", value=float(rows[index][field])) for col, field in zip(st.columns(5), flow.INPUTS)}
        st.caption('실제로 실행한 조건이 달랐다면 입력값을 수정하세요. 단위는 프로젝트에 저장된 원본 단위입니다.')
        st.subheader('2. 측정값과 실험 정보')
        a, b = st.columns(2)
        thick = a.number_input('실제 평균 두께', min_value=.000001, value=None, placeholder='측정값 입력')
        nonuni = b.number_input('실제 반범위 불균일도 (%)', min_value=0., value=None, placeholder='측정값 입력')
        lot = a.text_input('실제 Lot ID')
        wafer = b.number_input('웨이퍼 번호', min_value=1, value=1)
        day = st.date_input('실험 날짜', value=datetime.now().date(), max_value=datetime.now().date())
        moment = st.time_input('실험 시각', value=datetime.now().time())
        submitted = st.form_submit_button('실제 측정 저장', type='primary')
    if submitted:
        try:
            if thick is None or nonuni is None or not lot.strip():
                raise ValueError('실제 평균 두께, 불균일도와 Lot ID를 입력해주세요. 데모라면 위에서 「데모 체험」을 선택하세요.')
            flow.add_measurement(db, pid, rid, index, inputs, dict(thickness=thick, nonuniformity=nonuni), lot, wafer,
                                 datetime.combine(day, moment).astimezone().isoformat())
            st.success('저장했습니다. 다음 학습에서 실제 측정 포함을 선택할 수 있습니다.')
        except Exception as exc:
            st.error(str(exc))
    records = db.records(pid, 'measurement')
    if records:
        st.dataframe(pd.DataFrame([dict(Lot=r['payload']['lot'], 웨이퍼=r['payload']['wafer'], **r['payload']['actual']) for r in records]), hide_index=True)
    back(pid, 3, '가상 실험')
    st.button('모델 준비로 돌아가기 →', on_click=navigate, args=(pid, 2))


def settings_page(db, project):
    st.title('공정 기준과 로컬 AI')
    scope_notice()
    st.caption('BKM은 제공 데이터 안의 비교 기준입니다. 임의 설정값을 생산 최적 조건이나 산업 공통 규격으로 해석하지 않습니다.')
    s = project['settings']
    st.caption('온도·두께 단위는 데이터 승인 때 확인합니다. 모델의 학습 당시 평가 기준은 보존됩니다.')
    st.subheader('BKM 비교 기준')
    datasets = db.records(project['id'], 'dataset')
    if project['active_model']:
        model = db.record(project['id'], project['active_model'], 'model')['payload']
        show_bkm(flow.dataset_bkm(db, project['id'], model['dataset_id']))
        st.caption('현재 활성 모델이 학습한 데이터의 BKM을 사용합니다.')
    elif datasets:
        show_bkm(flow.dataset_bkm(db, project['id'], datasets[-1]['id']))
        st.caption('최근 승인 데이터의 참고 기준입니다. 가상 실험은 활성 모델의 데이터 기준을 사용합니다.')
    else:
        st.info('BKM 측정 기록이 포함된 데이터를 업로드하고 승인하면 비교 기준을 계산합니다.')
    st.caption('BKM은 레시피 이름으로 식별하고 Lot·웨이퍼 번호로 실측값을 연결합니다. 고정된 두께 합격 범위는 사용하지 않습니다.')
    with st.form('settings'):
        a, b = st.columns(2)
        nonuni = a.number_input('불균일도 실험 목표 (%) · 낮을수록 좋음', value=flow.experiment_target(s), min_value=0., max_value=100., step=.1)
        b.caption('기본 실험 목표는 5% 이하입니다. 산업 공통 합격 기준이 아니며, 지점별 두께의 합격 여부를 판정하지 않습니다.')
        tmae = a.number_input('허용 두께 MAE', value=float(s['thickness_mae']), min_value=.000001)
        nmae = b.number_input('허용 불균일도 MAE (퍼센트포인트)', value=float(s['nonuniformity_mae']), min_value=.000001)
        days = a.number_input('재학습 검토 주기 (일)', value=int(s['retrain_days']), min_value=1)
        count = b.number_input('재학습 알림 신규 측정 수', value=int(s['retrain_count']), min_value=1)
        if st.form_submit_button('설정 저장'):
            updated = dict(s, experiment_nonuniformity_target=nonuni,
                           thickness_mae=tmae, nonuniformity_mae=nmae, retrain_days=days, retrain_count=count)
            try:
                flow.validate_settings(updated)
                db.settings(project['id'], updated)
                st.success('저장했습니다.')
            except ValueError as exc:
                st.error(str(exc))
    st.caption('알림은 앱을 열었을 때 확인합니다. 앱 종료 중 자동 학습하거나 모델을 자동 교체하지 않습니다.')
    if st.button('Ollama 연결 확인'):
        try:
            if 'qwen3:14b' in llm.models():
                st.success('로컬 Qwen3 14B가 준비되어 있습니다.')
            else:
                st.warning('qwen3:14b가 없습니다. ollama pull qwen3:14b를 실행해주세요.')
        except Exception as exc:
            st.error('로컬 Ollama 연결 실패: '+str(exc))


def main():
    db = database(os.environ.get('TIKITAKA_V2_DATA_DIR', str(ROOT/'local_data/integrated')))
    if not db.projects():
        db.create_project('첫 공정', flow.DEFAULTS.copy())
    choices = {p['id']: p['name'] for p in db.projects()}
    if 'requested_project' in st.session_state:
        st.session_state['project_select'] = st.session_state.pop('requested_project')
    if st.session_state.get('project_select') not in choices:
        st.session_state['project_select'] = next(iter(choices))
    top, project_col = st.columns([2, 1])
    top.markdown('<div class="brand">TikiTaka<span>박막 공정 분석</span></div>', unsafe_allow_html=True)
    with project_col:
        pid = st.selectbox('프로젝트', list(choices), format_func=choices.get, key='project_select')
    project = db.project(pid)
    with st.expander('프로젝트 관리 · 공정 기준 · 로컬 AI'):
        tab1, tab2 = st.tabs(['프로젝트 만들기', '공정 기준과 AI'])
        with tab1:
            st.caption('같은 지원 공정의 데이터를 프로젝트별로 분리합니다. 새 프로젝트를 만들어도 다른 가스 조합을 지원하지는 않습니다.')
            with st.form('create'):
                name = st.text_input('프로젝트 이름')
                if st.form_submit_button('프로젝트 만들기'):
                    try:
                        st.session_state['requested_project'] = db.create_project(name, flow.DEFAULTS.copy())
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        with tab2:
            settings_page(db, project)
    datasets = db.records(pid, 'dataset')
    did = st.session_state.get('dataset_'+pid, datasets[-1]['id'] if datasets else None)
    active = db.record(pid, project['active_model'], 'model')['payload'] if project['active_model'] else None
    ready = bool(active and active['dataset_id'] == did)
    predictions = db.records(pid, 'prediction')
    allowed = [True, bool(st.session_state.get('pending_'+pid) or datasets), bool(datasets), ready, bool(predictions)]
    default = 4 if predictions else 3 if ready else 2 if datasets else 1 if st.session_state.get('pending_'+pid) else 0
    step = st.session_state.get('step_'+pid, default)
    if not allowed[step]:
        step = default
        st.session_state['step_'+pid] = step
    with st.container(key='workflow-nav'):
        for i, col in enumerate(st.columns(5)):
            with col:
                st.button(f'{i+1:02}  {STEPS[i]}', disabled=not allowed[i],
                          type='primary' if i == step else 'secondary', use_container_width=True,
                          key=f'step_{pid}_{i}', on_click=navigate, args=(pid, i))
    st.markdown(f'<div class="step-label">STEP {step+1:02} / 05 · {STEPS[step]}</div>', unsafe_allow_html=True)
    for reminder in flow.reminders(db, pid):
        st.info(reminder)
    global BODY
    BODY = st.empty()
    with BODY.container():
        [upload, review, training, experiments, measurements][step](db, project)
    st.divider()
    st.caption('로컬 LLM은 파일 해석, 회귀 모델은 품질 예측을 담당합니다. 원본과 모델은 이 컴퓨터에 저장됩니다. 로컬 실행만으로 사내 보안 검증이 완료되는 것은 아닙니다.')


main()
