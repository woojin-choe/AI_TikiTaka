import io
import pandas as pd
import pytest
from lab import condition_files as files


@pytest.fixture
def meta():
    return dict(units=dict(power='W', pressure='mTorr', temperature='°C', gas='sccm', time='s'),
                defaults=dict(power=300., pressure=2500., temperature=350., gas=180., time=30.),
                ranges=dict(time=[20., 40.]))


def test_excel_csv_roundtrip_and_column_order(meta):
    frame = files.example_frame(meta)
    excel = files.read_conditions(files.example_xlsx(meta), 'test.xlsx', meta)
    csv = files.read_conditions(frame.iloc[:, ::-1].to_csv(index=False).encode('utf-8-sig'), 'test.csv', meta)
    pd.testing.assert_frame_equal(excel, csv, check_dtype=False)
    assert excel.time.tolist() == [25, 30, 35]
    meta['units']['temperature'] = 'K'
    meta['defaults']['temperature'] = 623.15
    result = files.read_conditions(files.example_xlsx(meta), 'test.xlsx', meta)
    assert result.temperature.tolist() == [623.15]*3


@pytest.mark.parametrize('value', ['', 'NaN', 'inf', '=1+2', 'abc'])
def test_invalid_numeric_cells(meta, value):
    frame = files.example_frame(meta).astype(object)
    frame.iloc[0, 0] = value
    with pytest.raises(ValueError, match='2행'):
        files.read_conditions(frame.to_csv(index=False).encode(), 'x.csv', meta)


def test_wrong_units_duplicates_limits_and_ranges(meta):
    frame = files.example_frame(meta)
    for names in [[*frame.columns[:2], '온도 (K)', *frame.columns[3:]], [frame.columns[0]]*5]:
        with pytest.raises(ValueError, match='열 이름과 단위'):
            files.read_conditions(frame.set_axis(names, axis=1).to_csv(index=False).encode(), 'x.csv', meta)
    for count in [0, 101]:
        data = pd.concat([frame]*34).iloc[:count].to_csv(index=False).encode()
        with pytest.raises(ValueError, match='1~100'):
            files.read_conditions(data, 'x.csv', meta)
    frame.iloc[0, 4] = 0
    with pytest.raises(ValueError, match='입력 범위'):
        files.read_conditions(frame.to_csv(index=False).encode(), 'x.csv', meta)


def test_excel_formula_rejected(meta):
    # Mutate a generated workbook to emulate an uploaded formula, not cached values.
    from zipfile import ZipFile
    from xml.etree import ElementTree as ET
    raw = files.example_xlsx(meta)
    out = io.BytesIO()
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with ZipFile(io.BytesIO(raw)) as src, ZipFile(out, 'w') as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == 'xl/worksheets/sheet1.xml':
                root = ET.fromstring(data)
                cell = next(c for c in root.iter(ns+'c') if c.get('r') == 'A2')
                ET.SubElement(cell, ns+'f').text = '100+200'
                data = ET.tostring(root)
            dst.writestr(item, data)
    with pytest.raises(ValueError, match='수식'):
        files.read_conditions(out.getvalue(), 'x.xlsx', meta)
