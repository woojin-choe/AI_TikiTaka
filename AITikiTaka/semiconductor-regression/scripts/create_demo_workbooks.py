"""Generate local app test fixtures from the existing synthetic CSV.

Run from the project root: python scripts/create_demo_workbooks.py
These are alternate layouts of the SAME records, not independent training datasets.
"""
from pathlib import Path
import json

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'demo_data'


def build():
    OUT.mkdir(exist_ok=True)
    source = pd.read_csv(ROOT / 'data/process_data.csv')
    expected = {}
    variants = [
        ('01_영문_양식.xlsx', 'Process', 1, list(source.columns), list(source.columns)),
        ('02_한글_보고서.xlsx', '공정기록', 4,
         ['Lot_ID', 'RF_Power', 'Pressure', 'Temperature', 'Gas_Flow_Rate', 'Thickness', 'Uniformity'],
         ['Lot_ID', '인가전력 (W)', '챔버압력 (mTorr)', '설정온도 (°C)', '가스유량 (sccm)', '측정두께 (nm)', '균일도 (%)']),
        ('03_다중시트_다른단위.xlsx', '실험 데이터', 3,
         ['Temperature', 'Lot_ID', 'Uniformity', 'RF_Power', 'Thickness', 'Gas_Flow_Rate', 'Pressure'],
         ['공정온도 (K)', 'Lot_ID', '균일도 (%)', 'RF 출력 (kW)', '박막두께 (µm)', 'Gas Flow (slm)', 'Pressure (Torr)']),
    ]
    for filename, title, header, order, names in variants:
        book = Workbook()
        sheet = book.active
        sheet.title = title
        if header > 1:
            sheet.append(['합성 공정 데이터 · 실제 공장 기록이 아닙니다'])
            sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=7)
            sheet['A1'].font = Font(size=15, bold=True, color='16334A')
            sheet.row_dimensions[1].height = 30
            if header == 4:
                sheet.append(['같은 기록의 다른 양식입니다. 다른 예제와 합쳐 학습하지 마세요.'])
            while sheet.max_row < header - 1:
                sheet.append([''])
        sheet.append(names)
        frame = source[order].copy()
        if filename.startswith('03'):
            frame['Temperature'] += 273.15
            for field in ['RF_Power', 'Thickness', 'Gas_Flow_Rate', 'Pressure']:
                frame[field] /= 1000
            instructions = book.create_sheet('안내', 0)
            instructions.append(['데모 파일 안내'])
            instructions.append(['공정 기록은 실험 데이터 시트에 있습니다.'])
            instructions.append(['이 데이터는 합성 예제이며 현장 검증 자료가 아닙니다.'])
            instructions.column_dimensions['A'].width = 75
            instructions['A1'].font = Font(bold=True, size=15, color='16334A')
        for row in frame.itertuples(index=False, name=None):
            sheet.append([None if pd.isna(v) else float(v) for v in row])
        for cell in sheet[header]:
            cell.fill = PatternFill('solid', fgColor='147D78')
            cell.font = Font(bold=True, color='FFFFFF')
            cell.alignment = Alignment(vertical='center')
        sheet.row_dimensions[header].height = 28
        for i, field in enumerate(order, 1):
            sheet.column_dimensions[get_column_letter(i)].width = 24
            for row in sheet.iter_rows(min_row=header + 1, min_col=i, max_col=i):
                row[0].number_format = '0' if field == 'Lot_ID' else '0.000'
        sheet.freeze_panes = f'A{header + 1}'
        sheet.auto_filter.ref = f'A{header}:G{sheet.max_row}'
        book.save(OUT / filename)
        expected[filename] = {'sheet': title, 'header_row': header, 'data_start_row': header + 1,
                              'mapping': {field: get_column_letter(i + 1) for i, field in enumerate(order)}}
        check = load_workbook(OUT / filename, read_only=True)
        assert check[title].max_row == len(source) + header
        check.close()
    (OUT / 'expected_mapping.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2))
    print(f'Created {len(variants)} synthetic workbook fixtures in {OUT}')


if __name__ == '__main__':
    build()
