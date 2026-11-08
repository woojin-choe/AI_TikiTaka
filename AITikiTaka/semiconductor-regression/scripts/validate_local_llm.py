"""Opt-in integration test against locally installed Ollama; synthetic inputs only."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.constants import ROOT, FIELDS
from lab import ingest, llm

expected = json.loads((ROOT / 'demo_data/expected_mapping.json').read_text())
results = []
for filename, truth in expected.items():
    raw = ingest.read_workbook((ROOT / 'demo_data' / filename).read_bytes(), filename)[truth['sheet']]
    start = time.monotonic()
    result = llm.analyze(raw, truth['sheet'], '동일 증착 공정. 전력, 챔버 압력, 설정 온도, 가스 유량. 두께와 균일도는 실측 결과.')
    matched = sum(result['fields'][f]['column'] == truth['mapping'][f] for f in FIELDS)
    report = {'file': filename, 'seconds': round(time.monotonic() - start, 2), 'matched': matched,
              'expected': len(FIELDS), 'header_correct': result['header_row'] == truth['header_row'], 'result': result}
    results.append(report)
    print(json.dumps({k: v for k, v in report.items() if k != 'result'}, ensure_ascii=False), flush=True)
path = ROOT / 'local_data/verification/local_llm.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
assert all(r['matched'] == 6 and r['header_correct'] for r in results), 'Review mapping mismatches'
