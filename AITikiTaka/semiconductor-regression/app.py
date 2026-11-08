"""Run with: streamlit run app.py. All project data stays on this machine."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from lab.constants import ROOT, FEATURES, TARGETS, FIELDS, LABELS, UNITS, UNIT_OPTIONS
from lab.store import Store
from lab import ingest, engine, llm

st.set_page_config(page_title='TikiTaka · 공정 실험실', page_icon='◈', layout='wide')
st.markdown('''<style>
 .block-container {max-width:1220px;padding-top:2.2rem;padding-bottom:4rem}
 h1 {letter-spacing:-.055em;font-size:2.3rem!important}
 h2,h3 {letter-spacing:-.035em}
 [data-testid="stSidebar"] {border-right:1px solid #e1e7ef}
 [data-testid="stMetric"] {background:white;border:1px solid #e1e7ef;border-radius:14px;padding:18px}
 .eyebrow {color:#147d78;font-size:12px;font-weight:700;letter-spacing:.14em}
 .hero {padding:26px 30px;border:1px solid #d9e7e8;border-radius:18px;background:linear-gradient(115deg,#edf8f5,#fff);margin-bottom:24px}
 .hero p {color:#53677a;margin-bottom:0}
 .step {color:#738293;font-size:13px;margin-bottom:10px}
 div.stButton > button {border-radius:9px}
 </style>''', unsafe_allow_html=True)


def store():
    return Store()


def defaults(demo=False):
    return dict(equipment='장비 A', process='박막 증착',
                definition='동일 장비·공정의 RF 전력, 챔버 압력, 설정온도, 지정 가스의 유량. 균일도는 동일 측정 정의를 사용합니다.',
                thickness_min=98.0, thickness_max=102.0, uniformity_min=98.0,
                thickness_mae=5.0 if demo else 2.0, uniformity_mae=1.0 if demo else .5,
                retrain_days=30, retrain_count=20, demo=demo)


def flash(text):
    st.session_state['flash'] = text


def demo_project(db):
    existing = [p for p in db.projects() if db.project(p['id'])['settings'].get('demo')]
    if existing:
        return existing[0]['id']
    pid = db.create_project('데모 · 증착 장비 A', defaults(True))
    data = (ROOT / 'data/process_data.csv').read_bytes()
    raw = ingest.read_workbook(data, 'process_data.csv')['CSV']
    mapping = ingest.suggest(raw, 1)
    frame = ingest.extract(raw, 'CSV', mapping, UNITS, 2, len(raw), group='A')
    ingest.save_import(db, pid, data, '합성_공정데이터.csv', frame,
                       dict(sheet='CSV', header=1, start=2, end=len(raw), mapping=mapping, units=UNITS, group='A', timestamp=None), True)
    return pid


def welcome(db):
    st.markdown('<div class="hero"><div class="eyebrow">LOCAL PROCESS LAB</div><h1>다음 실험을, 데이터로 준비하세요.</h1><p>과거 공정 기록을 정리하고, 새로운 조건의 두께와 균일도를 비교합니다.</p></div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for column, title, text in [(a, '01  기록 정리', '엑셀에서 공정 조건과 측정 결과를 찾습니다.'),
                                (b, '02  모델 준비', '우리 공정의 데이터로 학습하고 오차를 확인합니다.'),
                                (c, '03  후보 비교', '조건을 바꿔 다음 실제 실험을 준비합니다.')]:
        with column:
            with st.container(border=True):
                st.subheader(title)
                st.write(text)
    st.write('')
    if st.button('샘플 데이터로 시작하기', type='primary', use_container_width=True):
        st.session_state['project'] = demo_project(db)
        flash('합성 데모 데이터를 준비했습니다. 「모델 학습」에서 첫 모델을 만들어보세요.')
        st.rerun()
    st.caption('데모는 기존 저장소의 합성 데이터입니다. 실제 공장 성능을 입증하는 자료가 아닙니다.')
    create_project(db)


def create_project(db):
    with st.expander('새 공정 프로젝트 만들기', expanded=not db.projects()):
        with st.form('create_project'):
            name = st.text_input('프로젝트 이름', placeholder='예: 장비 A · 산화막 증착')
            a, b = st.columns(2)
            equipment = a.text_input('장비', placeholder='장비 식별자')
            process = b.text_input('공정', placeholder='동일한 재료·공정 단위')
            definition = st.text_area('입력 항목의 의미', value=defaults()['definition'],
                                      help='설정값/실측값, 가스 종류, 균일도 측정 정의를 구체적으로 적어주세요.')
            if st.form_submit_button('프로젝트 생성', type='primary'):
                if not name.strip() or not equipment.strip() or not process.strip():
                    st.error('프로젝트 이름, 장비, 공정을 모두 입력해주세요.')
                else:
                    config = defaults()
                    config.update(equipment=equipment, process=process, definition=definition)
                    st.session_state['project'] = db.create_project(name, config)
                    st.rerun()


def overview(db, project):
    pid = project['id']
    st.title('공정 실험실')
    st.write(f"{project['settings']['equipment']} · {project['settings']['process']}")
    datasets, models, measurements = [db.records(pid, kind) for kind in ['dataset', 'model', 'measurement']]
    a, b, c = st.columns(3)
    a.metric('등록된 데이터셋', len(datasets))
    b.metric('저장된 모델', len(models))
    c.metric('추가 실제 측정', len(measurements))
    for reason in engine.retraining_status(db, pid):
        st.warning('재학습 검토 · ' + reason)
    st.write('')
    if not datasets:
        st.info('먼저 「데이터 등록」에서 과거 공정 엑셀을 올려주세요.')
    elif not project['active_model']:
        st.info('「모델 학습」에서 데이터를 선택해 학습하고, 평가를 확인한 뒤 활성화하세요.')
    else:
        model = db.record(pid, project['active_model'], 'model')
        st.success(f"{model['payload']['name']} 모델이 준비되었습니다. 「가상 실험」에서 조건을 비교해보세요.")
    with st.container(border=True):
        st.subheader('사용 순서')
        st.write('**처음 한 번** · 데이터 등록 → 항목·단위 확인 → 모델 학습 → 평가 후 활성화')
        st.write('**실험할 때마다** · 조건 후보 비교 → 실제 실험 → 결과 등록')
        st.write('**데이터가 쌓이면** · 재학습 → 기존 모델 비교 → 승인 후 교체')
    st.caption('모든 파일·모델은 이 컴퓨터에 저장됩니다. 예측은 실제 실험의 후보를 고르는 보조 자료입니다.')
    with st.expander('예제 파일 받기'):
        st.download_button('기본 CSV 다운로드', (ROOT / 'data/process_data.csv').read_bytes(), '공정데이터_합성.csv', 'text/csv')
        for path in sorted((ROOT / 'demo_data').glob('*.xlsx')):
            st.download_button(path.stem, path.read_bytes(), path.name,
                               'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.caption('예제들은 같은 합성 기록의 다른 양식입니다. 서로 합쳐 학습하지 마세요.')


def upload(db, project):
    pid = project['id']
    st.title('과거 기록을 가져오세요')
    st.write('공정 조건 4개와 실제 측정 결과 2개가 있는 Excel 또는 CSV를 올려주세요.')
    file = st.file_uploader('공정 데이터 파일', type=['xlsx', 'csv'], key=f'upload_{pid}')
    if file is None:
        st.info('다른 회사의 양식이라도 항목을 찾아 연결할 수 있습니다. 원본 파일은 변경하지 않습니다.')
        return
    data = file.getvalue()
    token = pid + hashlib.sha256(data).hexdigest()[:12]
    try:
        if st.session_state.get('workbook_token') != token:
            st.session_state['workbook'] = ingest.read_workbook(data, file.name)
            st.session_state['workbook_token'] = token
        workbook = st.session_state['workbook']
    except Exception as exc:
        st.error(f'파일을 읽을 수 없습니다: {exc}')
        return
    sheet = st.selectbox('데이터가 있는 시트', list(workbook), key=f'sheet_{token}')
    raw = workbook[sheet]
    token += sheet
    st.caption('왼쪽은 Excel 행 번호, 위쪽은 열 위치입니다. 셀의 내부 저장값을 보여줍니다. 수식은 계산하지 않으므로 값으로 저장한 파일을 사용하세요.')
    if raw.attrs.get('percentage_cells'):
        st.info('Excel 백분율 서식 셀이 있습니다. 98%는 내부 저장값 0.98로 표시됩니다. 균일도 원본 단위는 fraction을 선택하세요.')
    preview_start = st.number_input('미리보기 시작 행', 1, len(raw), 1, key=f'preview_{token}')
    st.dataframe(raw.loc[preview_start:preview_start + 19], use_container_width=True)
    with st.expander('로컬 AI로 항목 찾기', expanded=True):
        st.write('선택한 시트의 미리보기 영역만 로컬 LLM에 전달합니다. 결과는 아래에서 직접 확인하세요.')
        model_name = st.text_input('Ollama 모델 이름', 'qwen3:4b', key=f'llm_name_{pid}')
        if st.button('AI로 항목 찾기', key=f'ai_{token}'):
            try:
                with st.spinner('로컬 AI가 제목과 단위를 살펴보고 있습니다… 최대 2분이 걸릴 수 있습니다.'):
                    answer = llm.analyze(raw, sheet, project['settings']['definition'], model_name, int(preview_start))
                st.session_state[f'ai_answer_{token}'] = answer
                st.session_state[f'header_{token}'] = answer['header_row']
                st.session_state[f'start_{token}'] = answer['data_start_row']
                for field, item in answer['fields'].items():
                    st.session_state[f'col_{token}_{field}'] = item['column'] or '선택 안 함'
                    st.session_state[f'unit_{token}_{field}'] = item['source_unit'] if item['source_unit'] in UNIT_OPTIONS[field] else '확인 필요'
                st.rerun()
            except Exception as exc:
                st.warning(f'AI 분석을 완료하지 못했습니다. 아래에서 직접 지정할 수 있습니다. ({exc})')
        answer = st.session_state.get(f'ai_answer_{token}')
        if answer:
            for issue in answer['issues']:
                st.warning(issue)
            st.dataframe(pd.DataFrame([{'항목': LABELS[f], '근거': v['reason'], '확인 필요': v['needs_confirmation']}
                                        for f, v in answer['fields'].items()]), hide_index=True)
        else:
            st.caption('AI를 실행하지 않은 경우 아래 기본 후보는 열 이름 규칙으로 찾은 결과입니다. LLM 분석 결과가 아닙니다.')
    header_key = f'header_{token}'
    if header_key not in st.session_state:
        st.session_state[header_key] = ingest.best_header(raw)
    header = st.number_input('제목 행 번호', 1, len(raw), key=header_key)
    signature = ingest.fingerprint(raw, sheet, header)
    suggestions = ingest.suggest(raw, header)
    with st.expander('이전에 확인한 양식 불러오기'):
        saved = [r for r in db.records(pid, 'dataset') if r['payload']['mapping'].get('fingerprint') == signature]
        if saved and st.button('같은 제목 구조의 매핑 적용'):
            info = saved[-1]['payload']['mapping']
            for field in FIELDS:
                st.session_state[f'col_{token}_{field}'] = info['mapping'][field]
                st.session_state[f'unit_{token}_{field}'] = info['units'][field]
            st.success('저장된 열·단위를 적용했습니다. 데이터 범위와 보조 열도 확인해주세요.')
        elif not saved:
            st.caption('일치하는 저장 양식이 없습니다.')
    st.subheader('항목과 단위를 확인해주세요')
    mapping, units = {}, {}
    for field in FIELDS:
        a, b, c = st.columns([1.1, 2.2, 1])
        a.write(f"**{LABELS[field]}**")
        a.caption('예측 입력' if field in FEATURES else '실제 측정 결과')
        ck, uk = f'col_{token}_{field}', f'unit_{token}_{field}'
        if ck not in st.session_state:
            st.session_state[ck] = suggestions[field] or '선택 안 함'
        if uk not in st.session_state:
            st.session_state[uk] = '확인 필요'
        mapping[field] = b.selectbox(f'{LABELS[field]} 열', ['선택 안 함'] + list(raw.columns), key=ck,
                                    format_func=lambda c: c if c == '선택 안 함' else f'{c} · {raw.loc[header, c]}', label_visibility='collapsed')
        units[field] = c.selectbox(f'{LABELS[field]} 원본 단위', ['확인 필요'] + UNIT_OPTIONS[field], key=uk, label_visibility='collapsed')
    a, b = st.columns(2)
    sk = f'start_{token}'
    if sk not in st.session_state:
        st.session_state[sk] = min(header + 1, len(raw))
    start = a.number_input('데이터 시작 행', 1, len(raw), key=sk)
    end = b.number_input('데이터 마지막 행', int(start), len(raw), len(raw), key=f'end_{token}_{start}')
    a, b = st.columns(2)
    choices = ['없음'] + list(raw.columns)
    group = a.selectbox('Lot / 배치 ID 열 (선택)', choices, key=f'group_{token}',
                        format_func=lambda c: c if c == '없음' else f'{c} · {raw.loc[header, c]}')
    timestamp = b.selectbox('실험 일시 열 (선택)', choices, key=f'time_{token}',
                            format_func=lambda c: c if c == '없음' else f'{c} · {raw.loc[header, c]}')
    st.caption('Lot ID는 같은 배치가 학습과 평가에 섞이지 않도록 사용합니다. 일시는 시간순 평가에 사용합니다.')
    try:
        frame = ingest.extract(raw, sheet, mapping, units, int(start), int(end),
                               None if group == '없음' else group, None if timestamp == '없음' else timestamp)
    except ValueError as exc:
        st.info(str(exc))
        return
    a, b, c = st.columns(3)
    a.metric('선택한 행', len(frame))
    b.metric('학습 가능한 행', int(frame.valid.sum()))
    c.metric('확인할 행', int((~frame.valid).sum()))
    if frame.duplicate_suspect.any():
        st.warning('같은 입력·출력 기록이 있습니다. 반복 실험일 수 있어 삭제하지 않고, 학습 시 같은 분할에 묶습니다.')
    st.dataframe(frame, hide_index=True, use_container_width=True)
    invalid = frame[~frame.valid]
    if len(invalid):
        st.warning(f'{len(invalid)}행은 원본에 보존하지만 학습에서는 제외됩니다. issue 열에서 이유를 확인하세요.')
    confirmed = st.checkbox('항목의 의미·단위·범위를 확인했으며, 표시된 유효 행만 학습에 사용하는 것에 동의합니다.', key=f'confirmed_{token}')
    is_demo = st.checkbox('합성·데모 데이터입니다', value=project['settings'].get('demo', False), key=f'demo_{token}')
    if st.button('확인한 데이터 저장', type='primary', disabled=not confirmed or not frame.valid.any()):
        info = dict(sheet=sheet, header=int(header), start=int(start), end=int(end), mapping=mapping,
                    units=units, group=None if group == '없음' else group,
                    timestamp=None if timestamp == '없음' else timestamp, fingerprint=signature)
        ingest.save_import(db, pid, data, file.name, frame, info, is_demo)
        flash('데이터를 저장했습니다. 「모델 학습」으로 이동하세요.')
        st.rerun()


def model_report(db, pid, record):
    p = record['payload']
    st.subheader(f"{p['name']} · {record['id'][:8]}")
    st.caption(f"{record['created'][:19]} UTC · {p['split_mode']}")
    if p['demo']:
        st.warning('합성 데이터 기반 결과입니다. 실제 공장 정확도를 의미하지 않습니다.')
    cols = st.columns(3)
    for col, (label, ids) in zip(cols, p['split_ids'].items()):
        col.metric({'train': '학습', 'validation': '검증', 'test': '테스트'}[label] + ' 기록', len(ids))
    validation = pd.DataFrame(p['validation']).T
    st.write('**검증 데이터에서 모델 선택**')
    st.dataframe(validation[['Thickness_MAE', 'Uniformity_MAE', 'normalized_MAE']], use_container_width=True)
    st.caption('정규화 점수 = (두께 MAE / 허용 MAE + 균일도 MAE / 허용 MAE) ÷ 2. 낮을수록 좋습니다. Baseline은 학습 평균값 예측입니다.')
    st.write('**선택된 모델의 별도 테스트 결과**')
    a, b = st.columns(2)
    a.metric('두께 MAE', f"{p['test']['Thickness_MAE']:.3f} nm")
    b.metric('균일도 MAE', f"{p['test']['Uniformity_MAE']:.3f} %p")
    st.caption('MAE는 여러 테스트 기록의 평균 절대오차입니다. 개별 예측의 보장 범위가 아닙니다.')
    table = pd.read_csv(db.path(pid, p['test_predictions']))
    columns = st.columns(2)
    for col, target in zip(columns, TARGETS):
        chart = alt.Chart(table).mark_circle(size=55, opacity=.65, color='#147d78').encode(
            x=alt.X(target + ':Q', title=f'실제 {LABELS[target]} ({UNITS[target]})', scale=alt.Scale(zero=False)),
            y=alt.Y('predicted_' + target + ':Q', title=f'예측 {LABELS[target]}', scale=alt.Scale(zero=False)),
            tooltip=[target, 'predicted_' + target]).properties(height=230)
        col.altair_chart(chart, use_container_width=True)
    st.caption(p['comparison_note'])
    if p['incumbent_test']:
        st.dataframe(pd.DataFrame({'새 모델': p['test'], '기존 모델': p['incumbent_test']}).loc[['Thickness_MAE', 'Uniformity_MAE']], use_container_width=True)
    if p['eligible']:
        st.success('학습 당시 설정한 검증 MAE 기준을 충족하고 기준 모델보다 개선되었습니다.')
    else:
        st.warning('검증 MAE 기준 또는 기준 모델 대비 개선 조건을 충족하지 못했습니다. 데이터를 보완하거나 공정에 맞는 허용 오차를 설정한 뒤 재학습하세요.')
    st.download_button('평가 결과 JSON 다운로드', db.path(pid, p['artifact']).parent.joinpath('report.json').read_bytes(),
                       f"평가_{record['id'][:8]}.json", 'application/json', key=f'download_{record["id"]}')
    if st.button('이 모델 활성화 / 이전 버전으로 복원', type='primary', disabled=not p['eligible'], key=f'activate_{record["id"]}'):
        db.activate(pid, record['id'])
        flash('활성 모델을 변경했습니다. 「가상 실험」에서 사용할 수 있습니다.')
        st.rerun()


def training(db, project):
    pid = project['id']
    st.title('우리 공정의 모델을 준비하세요')
    st.write('기존 선형회귀·Random Forest·XGBoost를 비교하고, 검증 데이터로 선택합니다.')
    datasets = db.records(pid, 'dataset')
    if not datasets:
        st.info('먼저 「데이터 등록」에서 과거 실험 기록을 저장해주세요.')
        return
    ids = [r['id'] for r in datasets]
    lookup = {r['id']: r for r in datasets}
    selected = st.selectbox('학습할 데이터셋', ids, index=len(ids) - 1,
                            format_func=lambda i: f"{lookup[i]['payload']['filename']} · 유효 {lookup[i]['payload']['valid_rows']}행 · {i[:6]}")
    st.caption('데이터셋 하나를 기준으로 학습합니다. 다른 양식으로 만든 같은 기록을 중복 합치지 않습니다.')
    include = st.checkbox('이 프로젝트에서 추가 등록한 실제 실험 결과 포함', value=True)
    config = project['settings']
    st.info(f"모델 선택 기준 · 두께 MAE ≤ {config['thickness_mae']} nm / 균일도 MAE ≤ {config['uniformity_mae']} %p. 「설정」에서 변경할 수 있습니다.")
    if st.button('학습 시작', type='primary'):
        progress = st.progress(0.0, text='데이터 검사 중')
        try:
            with st.spinner('학습 중입니다. 완료될 때까지 이 화면을 유지해주세요.'):
                mid = engine.train(db, pid, selected, include, lambda n, text: progress.progress(n, text=text))
            st.session_state[f'last_model_{pid}'] = mid
            st.success('학습을 완료했습니다. 아래 결과를 확인하고 모델을 활성화하세요.')
        except Exception as exc:
            st.error(f'학습을 완료하지 못했습니다: {exc}')
    models = db.records(pid, 'model')
    if models:
        choices = [r['id'] for r in models]
        chosen = st.selectbox('모델 버전', choices, index=len(choices) - 1,
                              format_func=lambda i: f"{i[:8]}" + (' · 활성' if i == project['active_model'] else ''))
        model_report(db, pid, db.record(pid, chosen, 'model'))


def experiments(db, project):
    pid, mid = project['id'], project['active_model']
    st.title('다음 실험 조건을 비교하세요')
    if not mid:
        st.info('먼저 모델을 학습하고 평가 화면에서 활성화해주세요.')
        return
    meta = db.record(pid, mid, 'model')['payload']
    st.caption(f"활성 모델 · {meta['name']} / {mid[:8]} · 목표 두께 {project['settings']['thickness_min']}~{project['settings']['thickness_max']} nm / 균일도 {project['settings']['uniformity_min']}% 이상")
    if meta['demo']:
        st.warning('합성 데이터 기반 시연입니다. 예측 조건을 실제 장비에 바로 적용하지 마세요.')
    candidate_key = f'candidates_{pid}_{mid}'
    if candidate_key not in st.session_state:
        st.session_state[candidate_key] = pd.DataFrame([meta['defaults']])
    with st.expander('과거 실험 조건에서 시작하기'):
        historical = pd.read_csv(db.path(pid, meta['snapshot']))
        index = st.number_input('과거 기록 번호', 1, len(historical), 1)
        st.dataframe(historical.iloc[[index - 1]][FIELDS], hide_index=True)
        if st.button('이 조건을 후보 목록에 추가'):
            st.session_state[candidate_key] = pd.concat([st.session_state[candidate_key], historical.iloc[[index - 1]][FEATURES]], ignore_index=True)
            st.session_state[f'editor_revision_{pid}'] = st.session_state.get(f'editor_revision_{pid}', 0) + 1
    st.write('값을 수정하고 표 아래의 **+**로 후보를 추가하세요. 단위는 열 제목에 표시되어 있습니다.')
    edited = st.data_editor(st.session_state[candidate_key], num_rows='dynamic', hide_index=True, use_container_width=True,
                            column_config={f: st.column_config.NumberColumn(f'{LABELS[f]} ({UNITS[f]})', required=True, format='%.3f') for f in FEATURES},
                            key=f'editor_{candidate_key}_{st.session_state.get(f"editor_revision_{pid}", 0)}')
    if st.button('예상 품질 비교', type='primary'):
        result = engine.predict(db, pid, mid, edited.to_dict('records'))
        st.session_state[f'result_{pid}'] = {'model_id': mid, 'rows': result.to_dict('records'), 'saved': False}
        st.session_state[candidate_key] = edited.copy()
    result_state = st.session_state.get(f'result_{pid}')
    if result_state and result_state['model_id'] == mid:
        result = pd.DataFrame(result_state['rows'])
        changed = not edited[FEATURES].reset_index(drop=True).equals(result[FEATURES].reset_index(drop=True))
        if changed:
            st.warning('입력 조건이 변경되었습니다. 「예상 품질 비교」를 다시 눌러주세요.')
        st.subheader('예측 결과')
        st.dataframe(result.rename(columns={f: f'{LABELS[f]} ({UNITS[f]})' for f in FIELDS}), hide_index=True, use_container_width=True)
        if result['범위 확인'].str.len().gt(0).any():
            st.warning('학습 최소·최대 범위를 벗어난 항목이 있습니다. 범위 안에 있어도 정확도가 보장되지는 않습니다.')
        cols = st.columns(2)
        for col, target in zip(cols, TARGETS):
            plot = result[[target]].copy()
            plot['후보'] = [f'후보 {i + 1}' for i in range(len(result))]
            col.altair_chart(alt.Chart(plot).mark_bar(color='#147d78', cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X('후보:N', sort=None), y=alt.Y(target + ':Q', title=f'{LABELS[target]} ({UNITS[target]})'), tooltip=['후보', target]).properties(height=220), use_container_width=True)
        st.caption('예측값은 실제 측정 결과가 아닙니다. 실제 실험 후 결과를 등록해 오차를 확인하세요.')
        if st.button('이 후보와 예측 결과 저장', disabled=changed or result_state['saved']):
            engine.save_prediction(db, pid, mid, result[FEATURES].to_dict('records'))
            result_state['saved'] = True
            st.success('저장했습니다. 「실제 결과」에서 실험 결과를 연결할 수 있습니다.')
        st.download_button('예측 결과 CSV 다운로드', result.to_csv(index=False).encode('utf-8-sig'), '예측결과.csv', 'text/csv')


def measurements(db, project):
    pid = project['id']
    st.title('실제 실험 결과를 연결하세요')
    st.write('실제 사용한 조건과 측정값을 기록합니다. 다음 학습에는 실제 측정값만 사용합니다.')
    predictions = db.records(pid, 'prediction')
    if not predictions:
        st.info('「가상 실험」에서 예측 후보를 먼저 저장해주세요.')
        return
    lookup = {r['id']: r for r in predictions}
    selected = st.selectbox('저장한 예측', list(lookup), format_func=lambda i: f"{lookup[i]['created'][:19]} UTC · {i[:8]}")
    payload = lookup[selected]['payload']
    index = st.selectbox('실험한 후보', list(range(len(payload['rows']))), format_func=lambda i: f'후보 {i + 1}')
    row = payload['rows'][index]
    st.info(f"당시 예측 · 두께 {row['Thickness']:.3f} nm / 균일도 {row['Uniformity']:.3f}%")
    with st.form(f'measure_{selected}_{index}'):
        st.write('**실제 사용한 공정 조건**')
        columns = st.columns(4)
        inputs = {f: col.number_input(f'{LABELS[f]} ({UNITS[f]})', value=float(row[f]), format='%.3f') for col, f in zip(columns, FEATURES)}
        a, b = st.columns(2)
        thickness = a.number_input('실제 두께 (nm)', min_value=.001, value=100.0, format='%.3f')
        uniformity = b.number_input('실제 균일도 (%)', min_value=0.0, max_value=100.0, value=98.0, format='%.3f')
        lot = st.text_input('Lot / 배치 ID', help='학습 데이터에 Lot ID가 있으면 동일한 규칙으로 입력해야 재학습할 수 있습니다.')
        when = st.text_input('실제 실험 일시 (시간대 포함)', datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                             help='예: 2026-11-07T14:30:00+09:00')
        note = st.text_input('실험 메모')
        checked = st.checkbox('예측값이 아닌 실제로 측정한 결과를 입력했습니다.')
        if st.form_submit_button('실제 결과 저장', type='primary'):
            model = db.record(pid, payload['model_id'], 'model')['payload']
            if not checked:
                st.error('실제 측정 결과 확인란을 선택해주세요.')
            elif model['seen_groups'] and not lot.strip():
                st.error('이 모델은 Lot 단위로 학습했습니다. Lot ID를 입력해주세요.')
            else:
                engine.add_measurement(db, pid, selected, index, inputs, {'Thickness': thickness, 'Uniformity': uniformity}, lot, when, note)
                st.success('실제 결과를 저장했습니다.')
    records = db.records(pid, 'measurement')
    if records:
        rows = []
        for record in records:
            p = record['payload']
            rows.append({'실험 일시': p['measured_at'], 'Lot': p['lot'], '조건 변경': p['conditions_changed'],
                         **p['actual'], **{t + ' 절대오차': abs(p['actual'][t] - p['expected_at_actual_inputs'][t]) for t in TARGETS}})
        st.subheader('누적 측정 기록')
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption('조건이 변경된 경우 당시 모델에 실제 조건을 다시 넣어 계산한 예측과 비교합니다.')


def settings(db, project):
    st.title('공정 설정')
    p = project['settings']
    with st.form('settings'):
        st.text_area('입력 정의 (프로젝트 생성 시 확정)', p['definition'], disabled=True)
        st.caption('입력 의미나 장비·공정이 달라지면 새 프로젝트를 만들어주세요.')
        a, b, c = st.columns(3)
        low = a.number_input('목표 두께 최솟값 (nm)', min_value=.001, value=p['thickness_min'])
        high = b.number_input('목표 두께 최댓값 (nm)', min_value=.001, value=p['thickness_max'])
        uniformity = c.number_input('목표 균일도 최솟값 (%)', 0.0, 100.0, p['uniformity_min'])
        a, b = st.columns(2)
        tm = a.number_input('허용 두께 MAE (nm)', min_value=.001, value=p['thickness_mae'])
        um = b.number_input('허용 균일도 MAE (%p)', min_value=.001, value=p['uniformity_mae'])
        st.caption('허용 오차는 공정 요구사항에 따라 정하세요. 변경된 기준은 다음 학습부터 적용됩니다.')
        a, b = st.columns(2)
        days = a.number_input('재학습 검토 주기 (일)', 1, 3650, p['retrain_days'])
        count = b.number_input('재학습 알림 신규 측정 수', 1, 100000, p['retrain_count'])
        st.caption('앱 실행 중 조건을 확인해 알림을 표시합니다. 자동으로 모델을 교체하지 않습니다.')
        if st.form_submit_button('설정 저장', type='primary'):
            if low >= high:
                st.error('목표 두께 최솟값은 최댓값보다 작아야 합니다.')
            else:
                p.update(thickness_min=low, thickness_max=high, uniformity_min=uniformity,
                         thickness_mae=tm, uniformity_mae=um, retrain_days=days, retrain_count=count)
                db.settings(project['id'], p)
                flash('설정을 저장했습니다.')
                st.rerun()
    st.subheader('로컬 AI 연결')
    if st.button('Ollama 연결 확인'):
        try:
            names = llm.models()
            st.success('연결되었습니다: ' + ', '.join(names)) if names else st.info('Ollama는 실행 중이나 설치된 모델이 없습니다.')
        except Exception:
            st.warning('Ollama에 연결할 수 없습니다. 설치 안내에 따라 실행해주세요. 수동 매핑은 계속 사용할 수 있습니다.')
    st.code('ollama pull qwen3:4b\nollama serve', language='bash')
    st.caption('모델 설치는 인터넷이 필요합니다. 설치 후 파일 분석은 127.0.0.1의 로컬 Ollama만 사용합니다.')
    create_project(db)


def main():
    db = store()
    projects = db.projects()
    with st.sidebar:
        st.markdown('### ◈ TikiTaka')
        st.caption('LOCAL PROCESS LAB')
        if not projects:
            st.write('파일부터 다음 실험까지, 한 곳에서.')
        else:
            ids = [p['id'] for p in projects]
            lookup = {p['id']: p['name'] for p in projects}
            if st.session_state.get('project') not in ids:
                st.session_state['project'] = ids[0]
            pid = st.selectbox('공정 프로젝트', ids, format_func=lambda i: lookup[i], key='project')
            page = st.radio('작업', ['개요', '데이터 등록', '모델 학습', '가상 실험', '실제 결과', '설정'])
            st.divider()
            st.caption('로컬 실행 · 외부 AI API 미사용')
            st.caption('입력 4개 / 예측 품질 2개')
    if st.session_state.get('flash'):
        st.success(st.session_state.pop('flash'))
    if not projects:
        welcome(db)
        return
    project = db.project(pid)
    if project['settings'].get('demo'):
        st.caption('DEMO · 합성 데이터로 체험하는 공정 실험실')
    pages = {'개요': overview, '데이터 등록': upload, '모델 학습': training,
             '가상 실험': experiments, '실제 결과': measurements, '설정': settings}
    try:
        pages[page](db, project)
    except Exception as exc:
        st.error(f'작업을 완료하지 못했습니다: {exc}')
        st.caption('입력 내용을 확인하고 다시 시도해주세요. 기존에 저장한 데이터와 활성 모델은 유지됩니다.')


if __name__ == '__main__':
    main()
