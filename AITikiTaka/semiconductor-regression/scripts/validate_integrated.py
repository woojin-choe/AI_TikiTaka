"""Real local-LLM integration test, with separate post-extraction answer comparison."""
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab import extraction, workflow as flow
from lab.store import Store

OUT = ROOT/'demo_validation'
run = OUT/'runtime'/uuid.uuid4().hex
db = Store(run/'store')
report = {'run': str(run), 'cases': []}
for file in sorted((ROOT/'demo_uploads').glob('*.xlsx')):
    company = file.name[0]
    print('Extracting', company, flush=True)
    pid = db.create_project(company+' 통합 시험', flow.DEFAULTS.copy())
    log = db.directory(pid)/'extraction'
    frame, plan, diagnostics, provenance = extraction.extract(file.read_bytes(), file.name, log, progress=lambda message: print(company, message, flush=True))
    # Answers are first accessed AFTER model extraction has completed.
    expected = pd.read_csv(OUT/f'{company}_expected.csv').set_index(['lot', 'wafer']).sort_index()
    actual = frame.set_index(['lot', 'wafer']).sort_index()
    assert actual.index.equals(expected.index)
    np.testing.assert_allclose(actual[extraction.NUMBERS], expected[extraction.NUMBERS], atol=1e-6, rtol=1e-8)
    frame.to_csv(OUT/f'{company}_extracted.csv', index=False)
    provenance.to_csv(OUT/f'{company}_provenance.csv', index=False)
    (OUT/f'{company}_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    (OUT/f'{company}_diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    pending = dict(frame=frame, provenance=provenance, filename=file.name, data=file.read_bytes(),
                   plan=plan, diagnostics=diagnostics, log='extraction')
    did = flow.approve(db, pid, pending, '°C', 'nm', demo=True, scope_confirmed=True)
    mid = flow.train(db, pid, did, progress=lambda text: print(company, text, flush=True))
    meta = db.record(pid, mid, 'model')['payload']
    assert meta['eligible']
    db.activate(pid, mid)
    defaults = meta['defaults']
    prediction = flow.predict(db, pid, mid, [defaults, dict(defaults, time=defaults['time']+5)])
    assert prediction.iloc[1].thickness > prediction.iloc[0].thickness
    saved = flow.save_prediction(db, pid, mid, [defaults])
    flow.add_measurement(db, pid, saved, 0, defaults,
                         {t: float(prediction.iloc[0][t]) for t in flow.OUTPUTS},
                         'SYNTHETIC-NEW-LOT', 1, datetime.now(timezone.utc).isoformat())
    second = flow.train(db, pid, did, include_measurements=True)
    assert db.project(pid)['active_model'] == mid  # No automatic model replacement.
    assert len(db.record(pid, second, 'model')['payload']['measurements_used']) == 1
    report['cases'].append(dict(company=company, rows=len(frame), cells=len(frame)*7, matched=True,
                                model=meta['name'], test=meta['test'], eligible=meta['eligible']))
    print(company, 'PASS: extraction -> approval -> training -> activation -> prediction -> measurement -> retraining', flush=True)
(OUT/'integration_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
