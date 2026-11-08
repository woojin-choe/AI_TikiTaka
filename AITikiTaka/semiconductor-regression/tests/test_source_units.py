import pandas as pd
from lab.extraction import infer_source_units


class Reply:
    def __init__(self, wrong=False):
        self.wrong = wrong

    def call(self, folder, stage, prompt, payload, schema):
        return {f: {'unit': u, 'evidence_id': 'invented' if self.wrong else f+'_0'}
                for f, u in [('temperature', '°C'), ('thickness', 'nm')]}


def source(temp='온도 (°C)', thickness='평균 두께 (nm)', note='보고서'):
    raw = pd.DataFrame([[note, ''], [temp, thickness], [350, 3000]],
                       index=[1, 2, 3], columns=['A', 'B'])
    plan = {'tables': {'flat': {'sheet': '공정', 'header_row': 2, 'data_start_row': 3,
                              'columns': {'temperature': 'A', 'thickness': 'B'}, 'point_columns': []}}}
    return {'공정': raw}, plan


def test_grounded_units(tmp_path):
    book, plan = source()
    result = infer_source_units(book, plan, tmp_path, Reply())
    assert result['temperature']['unit'] == '°C'
    assert result['temperature']['evidence'][0]['cell'] == 'A2'
    assert result['thickness']['unit'] == 'nm'


def test_no_units_no_numeric_guess(tmp_path):
    book, plan = source('온도', '두께')
    result = infer_source_units(book, plan, tmp_path, Reply())
    assert all(r['unit'] is None for r in result.values())


def test_conflicting_units_block_auto_selection(tmp_path):
    book, plan = source(note='온도 단위 K / 두께 µm')
    result = infer_source_units(book, plan, tmp_path, Reply())
    assert all(r['unit'] is None for r in result.values())


def test_fabricated_evidence_rejected(tmp_path):
    book, plan = source()
    result = infer_source_units(book, plan, tmp_path, Reply(wrong=True))
    assert all(r['unit'] is None for r in result.values())


def test_unsupported_unit_is_not_relabelled(tmp_path):
    book, plan = source('온도 (°F)', '두께 (Å)')
    result = infer_source_units(book, plan, tmp_path, Reply())
    assert all(r['unit'] is None for r in result.values())
