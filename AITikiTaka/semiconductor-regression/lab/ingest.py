from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
import zipfile

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from lab.constants import ALIASES, FIELDS, UNIT_OPTIONS

MAX_ROWS = 30000
MAX_COLS = 120


def read_workbook(data, filename):
    if len(data) > 20 * 1024 * 1024:
        raise ValueError('파일은 20MB 이하로 올려주세요.')
    if filename.lower().endswith('.csv'):
        for encoding in ['utf-8-sig', 'cp949']:
            try:
                raw = pd.read_csv(io.BytesIO(data), header=None, dtype=object,
                                  encoding=encoding, keep_default_na=False, nrows=MAX_ROWS + 1)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError('CSV는 UTF-8 또는 CP949로 저장해주세요.')
        result = {'CSV': raw}
    elif filename.lower().endswith('.xlsx'):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(i.file_size for i in archive.infolist()) > 100 * 1024 * 1024:
                raise ValueError('압축 해제 크기가 100MB를 초과합니다.')
        # Formulas are deliberately not evaluated. They fail numeric validation.
        book = load_workbook(io.BytesIO(data), read_only=True, data_only=False, keep_links=False)
        try:
            if len(book.worksheets) > 20:
                raise ValueError('첫 버전에서는 시트 20개 이하만 지원합니다.')
            result = {}
            for sheet in book.worksheets:
                if sheet.max_row is None or sheet.max_column is None:
                    sheet.calculate_dimension(force=True)
                if sheet.max_row > MAX_ROWS or sheet.max_column > MAX_COLS:
                    raise ValueError(f'{sheet.title}: 최대 {MAX_ROWS}행, {MAX_COLS}열까지 지원합니다.')
                values, percentage_cells = [], []
                for row in sheet.iter_rows():
                    values.append([cell.value for cell in row])
                    percentage_cells.extend(cell.coordinate for cell in row
                                            if cell.value is not None and '%' in cell.number_format)
                result[sheet.title] = pd.DataFrame(values).fillna('')
                result[sheet.title].attrs['percentage_cells'] = percentage_cells
        finally:
            book.close()
    else:
        raise ValueError('XLSX 또는 CSV 파일을 선택해주세요.')
    for sheet, frame in result.items():
        if frame.empty or len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLS:
            raise ValueError(f'{sheet}: 빈 표이거나 지원 크기를 초과합니다.')
        frame.columns = [get_column_letter(i + 1) for i in range(len(frame.columns))]
        frame.index = range(1, len(frame) + 1)
    return result


def normalize(value):
    return re.sub(r'[^a-z0-9가-힣]', '', str(value).lower())


def suggest(raw, row):
    """Conservative, deterministic fallback; explicitly not an LLM result."""
    mapping = {}
    headers = raw.loc[row].to_dict()
    for field, aliases in ALIASES.items():
        candidates = [col for col, value in headers.items()
                      if any(normalize(value).startswith(alias) for alias in aliases)]
        mapping[field] = candidates[0] if len(candidates) == 1 else None
    return mapping


def best_header(raw):
    candidates = list(raw.index[:40])
    return max(candidates, key=lambda row: sum(v is not None for v in suggest(raw, row).values()))


def fingerprint(raw, sheet, row):
    body = [sheet, row, [str(x) for x in raw.loc[row].tolist()]]
    return hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()


def convert(values, field, unit):
    if unit not in UNIT_OPTIONS[field]:
        raise ValueError(f'{field}: 단위를 확인해주세요.')
    values = pd.to_numeric(values, errors='coerce')
    factor = {'kW': 1000, 'Torr': 1000, 'Pa': 7.500616827, 'slm': 1000,
              'µm': 1000, 'fraction': 100}.get(unit, 1)
    return values - 273.15 if unit == 'K' else values * factor


def extract(raw, sheet, mapping, units, start, end, group=None, timestamp=None):
    if not 1 <= start <= end <= len(raw):
        raise ValueError('데이터 시작·종료 행을 확인해주세요.')
    if any(mapping.get(f) not in raw.columns for f in FIELDS):
        raise ValueError('학습에는 입력 4개와 실제 측정 결과 2개가 모두 필요합니다.')
    if len(set(mapping[f] for f in FIELDS)) != len(FIELDS):
        raise ValueError('하나의 원본 열을 여러 항목에 중복 지정할 수 없습니다.')
    pct_cells = set(raw.attrs.get('percentage_cells', []))
    uniformity_pct = [f"{mapping['Uniformity']}{i}" in pct_cells for i in range(start, end + 1)
                      if str(raw.loc[i, mapping['Uniformity']]).strip()]
    if any(uniformity_pct) and not all(uniformity_pct):
        raise ValueError('균일도 열에 Excel 백분율 서식과 일반 숫자가 혼재합니다. 저장값을 같은 단위로 통일해주세요.')
    if any(uniformity_pct) and units.get('Uniformity') != 'fraction':
        raise ValueError('균일도는 Excel 백분율 서식입니다. 저장값 0.98이 98%를 뜻하므로 원본 단위를 fraction으로 선택해주세요.')
    source = raw.loc[start:end]
    out = pd.DataFrame(index=source.index)
    issues = pd.Series('', index=source.index)
    for field in FIELDS:
        out[field] = convert(source[mapping[field]], field, units[field])
        bad = ~np.isfinite(out[field])
        issues.loc[bad] += field + ': 누락/숫자 아님; '
    for field in ['RF_Power', 'Pressure', 'Gas_Flow_Rate', 'Thickness']:
        bad = out[field] <= 0 if field in ['Pressure', 'Thickness'] else out[field] < 0
        issues.loc[bad] += field + ': 물리 범위 확인; '
    issues.loc[(out.Temperature < -273.15)] += '절대영도 미만; '
    issues.loc[~out.Uniformity.between(0, 100) & out.Uniformity.notna()] += '균일도 0~100% 범위; '
    out['source_sheet'] = sheet
    out['source_row'] = source.index
    if group:
        out['Lot_ID'] = source[group].astype(str).str.strip()
        issues.loc[out.Lot_ID.eq('')] += '그룹 식별자 누락; '
    if timestamp:
        parsed = pd.to_datetime(source[timestamp], errors='coerce', utc=True)
        issues.loc[parsed.isna()] += '시간 형식 오류; '
        out['measured_at'] = parsed.astype(str)
    out['issue'] = issues
    out['valid'] = issues.eq('')
    out['duplicate_suspect'] = out.duplicated(FIELDS, keep=False)
    return out.reset_index(drop=True)


def save_import(store, pid, data, filename, frame, mapping_info, demo=False):
    digest = hashlib.sha256(data).hexdigest()
    signature = hashlib.sha256(json.dumps(mapping_info, sort_keys=True).encode()).hexdigest()
    for record in store.records(pid, 'dataset'):
        p = record['payload']
        if p.get('file_hash') == digest and p.get('signature') == signature:
            raise ValueError('동일한 파일과 매핑이 이미 등록되어 있습니다.')
    rid = uuid.uuid4().hex
    frame = frame.copy()
    # Stable identity for same imported file / sheet / row, independent of display name.
    frame['record_id'] = [hashlib.sha256(f'{digest}:{r.source_sheet}:{r.source_row}'.encode()).hexdigest()
                          for r in frame.itertuples()]
    suffix = '.xlsx' if filename.lower().endswith('.xlsx') else '.csv'
    raw_path = store.write(pid, f'raw/{rid}{suffix}', data)
    path = store.write(pid, f'datasets/{rid}.csv', frame.to_csv(index=False).encode('utf-8'))
    return store.add(pid, 'dataset', {
        'filename': filename, 'path': path, 'raw_path': raw_path,
        'file_hash': digest, 'signature': signature, 'mapping': mapping_info,
        'rows': len(frame), 'valid_rows': int(frame.valid.sum()), 'demo': demo,
    }, rid)
