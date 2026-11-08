"""Verify local source-unit suggestions, using no expected values in inference."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.extraction import suggest_units
for company in ['K', 'L']:
    file = next((ROOT/'demo_uploads').glob(company+'*.xlsx'))
    plan = json.loads((ROOT/'demo_validation'/f'{company}_plan.json').read_text())
    result = suggest_units(file.read_bytes(), file.name, plan, ROOT/'demo_validation'/f'{company}_unit_check')
    assert result['temperature']['unit'] == '°C', result
    assert result['thickness']['unit'] == 'nm', result
    (ROOT/'demo_validation'/f'{company}_units.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(company, json.dumps(result, ensure_ascii=False), flush=True)
