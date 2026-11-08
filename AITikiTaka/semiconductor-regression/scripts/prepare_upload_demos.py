"""Independent synthetic records for the integrated upload flow, not production evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'demo_validation'
OUT.mkdir(exist_ok=True)
specs = []


def sheet(name, header, columns, rows, start=2):
    return dict(name=name, header=header, start=start, columns=columns, rows=rows)


for company, seed in [('K', 781), ('L', 904)]:
    rng = np.random.default_rng(seed)
    n = 120
    power = rng.uniform(200, 500, n).round(2)
    pressure = rng.uniform(1500, 3500, n).round(2)
    temperature = rng.uniform(250, 450, n).round(2)
    gas = rng.uniform(120, 250, n).round(2)
    duration = rng.uniform(20, 50, n).round(2)
    thickness = 1.2*power+.1*pressure+2*temperature+1.3*gas+50*duration+rng.normal(0, 3, n)
    nonuni = 3+.002*power+.0003*pressure+.004*temperature+.005*gas-.025*duration+rng.normal(0, .04, n)
    frame = pd.DataFrame(dict(lot=[f'{company}-LOT-{i//10+1:02}' for i in range(n)], wafer=[i%10+1 for i in range(n)],
                             power=power, pressure=pressure, temperature=temperature, gas=gas, time=duration,
                             thickness=thickness, nonuniformity=nonuni))
    frame.to_csv(OUT/f'{company}_expected.csv', index=False)
    if company == 'K':
        order = ['nonuniformity', 'lot', 'time', 'power', 'wafer', 'thickness', 'gas', 'temperature', 'pressure']
        columns = ['반범위 불균일도 (%)', '로트번호', '증착 시간 (s)', 'RF 전력 설정값 (W)', '웨이퍼 번호',
                   '평균 두께 (nm)', 'Si2H6 설정 유량 (sccm)', '온도 설정값 (°C)', '압력 설정값 (mTorr)']
        rows = frame[order].sample(frac=1, random_state=seed).values.tolist()
        sheets = [sheet('공정 실험 기록', 6, columns, rows, 3),
                  sheet('장비 점검', 3, ['설비', '점검 결과', '담당자'], [['K-01', '완료', '데모']])]
    else:
        recipes, tracking, measurements = [], [], []
        sites = np.linspace(-1, 1, 43)
        for i, row in frame.iterrows():
            recipe = f'RUN-{i+1:03}'
            recipes.append([recipe, row.gas/1000, row.temperature, row.time/60, row.pressure/1000, row.power/1000])
            tracking.append([row.wafer, recipe, row.lot, 'PM1' if i%2 else 'PM2'])
            points = row.thickness*(1+sites*row.nonuniformity/100)
            measurements.append([row.lot, row.wafer, 'THK1_1_TOP', *points.tolist()])
            measurements.append([row.lot, row.wafer, 'GOF_1', *rng.uniform(.92, .99, 43).tolist()])
        permutation = rng.permutation(46)
        columns = np.array(['Lot ID', 'Wafer No', 'ITEM']+[f'P{i:02}' for i in range(1,44)])[permutation].tolist()
        measurement_rows = pd.DataFrame(measurements).iloc[:, permutation].sample(frac=1, random_state=seed).values.tolist()
        sheets = [sheet('측정 원장', 8, columns, measurement_rows, 2),
                  sheet('장비 점검', 4, ['설비', '점검 결과', '담당자'], [['L-01', '완료', '데모']]),
                  sheet('Wafer tracking', 5, ['웨이퍼 번호', '공정 레시피', '로트번호', '챔버'],
                        pd.DataFrame(tracking).sample(frac=1, random_state=seed).values.tolist(), 4),
                  sheet('Recipe settings', 11, ['Recipe ID', 'Si2H6 setpoint (slm)', 'Temperature setpoint (°C)',
                        'Deposition time (min)', 'Pressure setpoint (Torr)', 'RF setpoint (kW)'],
                        pd.DataFrame(recipes).sample(frac=1, random_state=seed).values.tolist(), 2)]
    specs.append(dict(company=company, filename=f'{company}회사_새공정_업로드데모.xlsx', sheets=sheets))
(OUT/'build_spec.json').write_text(json.dumps(specs, ensure_ascii=False))
print('Prepared independent synthetic K/L inputs and separate expected values')
