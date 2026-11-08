"""Explicit, unit-checked input files for batch predictions (no LLM needed)."""
import csv
import io
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from openpyxl import load_workbook

FIELDS = ['power', 'pressure', 'temperature', 'gas', 'time']
LABELS = ['RF 전력', '압력', '온도', 'Si₂H₆ 유량', '증착 시간']
TEMPLATE = Path(__file__).resolve().parents[1]/'assets/가상실험_조건예시.xlsx'


def headers(meta):
    return [f'{label} ({meta["units"][field]})' for field, label in zip(FIELDS, LABELS)]


def example_frame(meta):
    rows = [dict(meta['defaults']) for _ in range(3)]
    low, high = meta['ranges']['time']
    for row, fraction in zip(rows, [.25, .5, .75]):
        row['time'] = low + (high-low)*fraction
    return pd.DataFrame(rows)[FIELDS].round(6).set_axis(headers(meta), axis=1)


def example_xlsx(meta):
    """Fill the bundled styled template with this model's units and conditions."""
    values = [headers(meta), *example_frame(meta).values.tolist()]
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    out = io.BytesIO()
    with ZipFile(TEMPLATE) as source, ZipFile(out, 'w') as dest:
        for entry in source.infolist():
            data = source.read(entry.filename)
            if entry.filename == 'xl/worksheets/sheet1.xml':
                root = ET.fromstring(data)
                for cell in root.findall('.//s:sheetData/s:row/s:c', ns):
                    ref = cell.attrib['r']
                    col, row = ord(ref[0])-65, int(ref[1:])-1
                    if not (0 <= col < 5 and 0 <= row < 4):
                        continue
                    for child in list(cell):
                        cell.remove(child)
                    value = values[row][col]
                    if isinstance(value, str):
                        cell.set('t', 'inlineStr')
                        node = ET.SubElement(cell, '{'+ns['s']+'}is')
                        ET.SubElement(node, '{'+ns['s']+'}t').text = value
                    else:
                        cell.set('t', 'n')
                        ET.SubElement(cell, '{'+ns['s']+'}v').text = str(value)
                data = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            dest.writestr(entry, data)
    return out.getvalue()


def read_conditions(data, filename, meta):
    if len(data) > 20*1024*1024:
        raise ValueError('파일 크기는 20MB 이하여야 합니다.')
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == '.csv':
            try:
                text = data.decode('utf-8-sig')
            except UnicodeDecodeError:
                text = data.decode('cp949')
            rows = []
            for row in csv.reader(io.StringIO(text)):
                rows.append(row)
                if len(rows) > 102:
                    raise ValueError('실험 조건은 한 번에 1~100행까지 업로드할 수 있습니다.')
        elif suffix == '.xlsx':
            with ZipFile(io.BytesIO(data)) as archive:
                if sum(x.file_size for x in archive.infolist()) > 100*1024*1024:
                    raise ValueError('압축 해제한 엑셀 파일이 너무 큽니다.')
            book = load_workbook(io.BytesIO(data), read_only=True, data_only=False)
            try:
                if len(book.worksheets) != 1:
                    raise ValueError('엑셀에는 실험 조건 시트 하나만 남겨주세요.')
                sheet = book.worksheets[0]
                if (sheet.max_column or 0) > 5 or (sheet.max_row or 0) > 101:
                    raise ValueError('예시와 같은 5개 열, 최대 100개 조건으로 작성해주세요.')
                rows = []
                for row in sheet.iter_rows(values_only=True):
                    rows.append(row)
                    if len(rows) > 101:
                        raise ValueError('실험 조건은 한 번에 1~100행까지 업로드할 수 있습니다.')
            finally:
                book.close()
        else:
            raise ValueError('CSV 또는 XLSX 파일을 업로드해주세요.')
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('파일을 읽을 수 없습니다. 예시 양식으로 다시 저장해주세요.') from exc
    if not rows:
        raise ValueError('파일이 비어 있습니다.')
    names = [str(v).strip() if v is not None else '' for v in rows[0]]
    expected = headers(meta)
    if len(names) != 5 or len(set(names)) != 5 or set(names) != set(expected):
        raise ValueError('열 이름과 단위를 예시 파일과 동일하게 작성해주세요: '+', '.join(expected))
    body = rows[1:]
    while body and all(v is None or str(v).strip() == '' for v in body[-1]):
        body.pop()
    if not 1 <= len(body) <= 100:
        raise ValueError('실험 조건은 한 번에 1~100행까지 업로드할 수 있습니다.')
    if any(len(r) != 5 for r in body):
        raise ValueError('모든 행에 5개 입력값이 필요합니다.')
    frame = pd.DataFrame(body, columns=names)[expected]
    numeric = frame.apply(pd.to_numeric, errors='coerce')
    invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
    if invalid.any():
        cells = [f'{r+2}행 {expected[c]}' for r, c in zip(*np.where(invalid))]
        raise ValueError('빈 값·수식·문자 대신 숫자를 입력해주세요: '+', '.join(cells[:8]))
    numeric.columns = FIELDS
    bad = (numeric.power < 0) | (numeric.gas < 0) | (numeric.pressure <= 0) | (numeric.time <= 0)
    floor = {'°C': -273.15, 'K': 0}.get(meta['units']['temperature'])
    if floor is not None:
        bad |= numeric.temperature < floor
    if bad.any():
        raise ValueError('입력 범위를 확인해주세요 (전력·유량 ≥ 0, 압력·시간 > 0, 절대영도 이상): '+', '.join(str(i+2)+'행' for i in numeric.index[bad]))
    return numeric
