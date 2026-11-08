"""Read-only source audit and exploratory wafer-level evaluation.

Never activates a model or edits the uploaded workbooks. Thickness units are
unconfirmed. NonUniformity_range_pct is an explicitly chosen diagnostic target,
not an asserted company definition of Uniformity.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from src.models import get_models

SOURCE = APP.parents[1] / 'DataSet'
OUT = APP / 'local_data' / 'dataset_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {unicodedata.normalize('NFC', p.name): p for p in SOURCE.glob('*.xlsx')}
    trace_path = next(p for name, p in paths.items() if 'Trace' in name)
    source3 = paths['3회차 과제 데이터.xlsx']
    source2 = paths['2회차 과제 데이터.xlsx']
    combined = pd.read_excel(trace_path, sheet_name=None)
    original = pd.read_excel(source3, sheet_name=None)
    old_trace = pd.read_excel(source2)
    trace = combined['FDC_Trace_Data']
    summary = combined['Wafer_Depo_Summary']
    tracking = original['Tracking'].rename(columns={'WAFER_ID': 'WF'})
    measurements = original['두께']
    thickness = measurements[measurements.ITEM_ID.eq('THK1_1_TOP')].copy()
    keys = ['LOT_ID', 'WF']
    points = list(range(1, 44))
    assert not thickness.duplicated(keys).any(), 'Duplicate thickness wafer keys'
    assert not tracking.duplicated(keys).any(), 'Duplicate tracking wafer keys'
    assert thickness[points].notna().all().all(), 'Missing thickness point'
    a = thickness[points].to_numpy(dtype=float)
    computed = {'AVG': a.mean(axis=1), 'MAX': a.max(axis=1), 'MIN': a.min(axis=1),
                'RANGE': np.ptp(a, axis=1), 'STD': a.std(axis=1, ddof=1)}
    differences = {k: float(np.max(np.abs(thickness[k].to_numpy() - v))) for k, v in computed.items()}
    thickness['Thickness_original_unit'] = computed['AVG']
    thickness['NonUniformity_range_pct'] = 100 * computed['RANGE'] / (2 * computed['AVG'])
    thickness['CV_pct'] = 100 * computed['STD'] / computed['AVG']

    summary = summary.rename(columns={'CTC_chA_LotID': 'LOT_ID', 'CTC_chA_WaferNo': 'WF'})
    assert not summary.duplicated(keys).any(), 'Duplicate summary wafer keys'
    coverage = tracking.merge(summary, on=keys, how='outer', indicator=True, validate='one_to_one')
    assert coverage['_merge'].eq('both').all(), 'Trace and tracking keys do not match'
    recipe_mismatch = int((coverage.RECIPE_ID != coverage.chA_VIR_Recipe_Name).sum())
    assert recipe_mismatch == 0

    # Recompute summary from Depo steps, not from all trace rows.
    depo = trace[trace.chA_VIR_Step_Name.eq('Depo')].rename(
        columns={'CTC_chA_LotID': 'LOT_ID', 'CTC_chA_WaferNo': 'WF'})
    grouped = depo.groupby(keys)
    recalculated = grouped.size().rename('Depo_Duration_s').to_frame()
    aggregation = {
        'Si2H6_Setpoint': ('chA_AO_mfc2_setpoint_Si2H6', 'first'),
        'Si2H6_Actual_Mean': ('chA_AI_mfc2_flow_Si2H6', 'mean'),
        'Si2H6_Actual_Std': ('chA_AI_mfc2_flow_Si2H6', 'std'),
        'N2O_Actual_Mean': ('chA_AI_mfc3_flow_N2O', 'mean'),
        'Temp_Setpoint': ('chA_VIR_Heater_Temp_Set', 'first'),
        'Temp_Actual_Mean': ('chA_VIR_Heater_Temp', 'mean'),
        'Temp_Actual_Std': ('chA_VIR_Heater_Temp', 'std'),
        'Pressure_Actual_Mean': ('chA_AI_Manometer_Pressure', 'mean'),
        'Pressure_Actual_Std': ('chA_AI_Manometer_Pressure', 'std'),
        'RF_Forward_Power_Mean': ('FORWARD POWER', 'mean'),
        'Heater_Power_Center_Mean': ('chA_VIR_Heater_Power_Center', 'mean'),
        'Heater_Power_Edge_Mean': ('chA_VIR_Heater_Power_Edge', 'mean'),
    }
    for out, (col, operation) in aggregation.items():
        recalculated[out] = grouped[col].agg(operation)
    check = summary.set_index(keys).join(recalculated, rsuffix='_recalculated')
    summary_diff = {col: float((check[col] - check[col + '_recalculated']).abs().max())
                    for col in recalculated.columns}
    intervals = grouped.datetime.apply(lambda s: (s.sort_values().diff().dropna().dt.total_seconds() == 1).all())

    # Recipe table has three header rows; use explicit positions verified from source.
    raw_recipe = pd.read_excel(source3, sheet_name='조건표', header=None)
    recipe = raw_recipe.iloc[3:, [1, 2, 6, 7, 8, 9]].copy()
    recipe.columns = ['RECIPE_ID', 'Gas_Set_sccm', 'Pressure_Set_mTorr', 'RF_Set_W', 'Temp_Set_raw', 'Duration_s']
    recipe = recipe.dropna(subset=['RECIPE_ID'])
    for col in recipe.columns[1:]:
        recipe[col] = pd.to_numeric(recipe[col], errors='raise')
    joined = thickness.merge(tracking, on=keys, how='left', validate='one_to_one')
    joined = joined.merge(recipe, on='RECIPE_ID', how='left', validate='many_to_one')
    joined = joined.merge(summary, on=keys, how='left', validate='one_to_one')
    joined['measured_at'] = pd.to_datetime(joined.TKOUT_TIME)
    assert joined[['Gas_Set_sccm', 'RF_Set_W', 'Duration_s', 'RF_Forward_Power_Mean']].notna().all().all()
    assert len(joined) == 50
    joined.to_csv(OUT / 'wafer_level_audit.csv', index=False, encoding='utf-8-sig')

    feature_sets = {
        'recipe_4': ['RF_Set_W', 'Pressure_Set_mTorr', 'Temp_Set_raw', 'Gas_Set_sccm'],
        'recipe_5_with_time': ['RF_Set_W', 'Pressure_Set_mTorr', 'Temp_Set_raw', 'Gas_Set_sccm', 'Duration_s'],
        'trace_4_after_run': ['RF_Forward_Power_Mean', 'Pressure_Actual_Mean', 'Temp_Actual_Mean', 'Si2H6_Actual_Mean'],
    }
    target_cols = ['Thickness_original_unit', 'NonUniformity_range_pct']
    y = joined[target_cols].to_numpy()
    model_defs = get_models()
    model_defs['Baseline'] = DummyRegressor(strategy='mean')
    plans = {
        'leave_one_lot_out': list(LeaveOneGroupOut().split(joined, groups=joined.LOT_ID)),
        'random_wafer_5fold_diagnostic': list(KFold(5, shuffle=True, random_state=42).split(joined)),
        'leave_one_recipe_out': list(LeaveOneGroupOut().split(joined, groups=joined.RECIPE_ID)),
    }
    rows, fold_rows, predictions = [], [], []
    with threadpool_limits(limits=2):
        for features_name, columns in feature_sets.items():
            X = joined[columns].to_numpy(dtype=float)
            for plan, splits in plans.items():
                for name, estimator in model_defs.items():
                    pred = np.full_like(y, np.nan)
                    for fold, (tr, te) in enumerate(splits):
                        pipeline = make_pipeline(StandardScaler(), clone(estimator))
                        pipeline.fit(X[tr], y[tr])
                        pred[te] = pipeline.predict(X[te])
                        for i, target in enumerate(target_cols):
                            fold_rows.append(dict(features=features_name, split=plan, model=name, fold=fold,
                                                  heldout_groups=','.join(sorted(joined.iloc[te].LOT_ID.unique())),
                                                  target=target, n=len(te), MAE=float(mean_absolute_error(y[te, i], pred[te, i]))))
                    for i, target in enumerate(target_cols):
                        rows.append(dict(features=features_name, split=plan, model=name, target=target,
                                         MAE=float(mean_absolute_error(y[:, i], pred[:, i])),
                                         RMSE=float(np.sqrt(np.mean((y[:, i] - pred[:, i]) ** 2))),
                                         R2=float(r2_score(y[:, i], pred[:, i]))))
                    predictions.extend(dict(features=features_name, split=plan, model=name,
                                            LOT_ID=joined.iloc[i].LOT_ID, WF=int(joined.iloc[i].WF),
                                            predicted_thickness=float(pred[i, 0]), predicted_nonuniformity=float(pred[i, 1]))
                                       for i in range(len(joined)))
            print('Evaluated', features_name, flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / 'evaluation.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(fold_rows).to_csv(OUT / 'fold_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(predictions).to_csv(OUT / 'out_of_fold_predictions.csv', index=False, encoding='utf-8-sig')
    collisions = joined.groupby(feature_sets['recipe_4']).agg(
        wafers=('WF', 'size'), distinct_times=('Duration_s', 'nunique'),
        time_min=('Duration_s', 'min'), time_max=('Duration_s', 'max'),
        thickness_min=('Thickness_original_unit', 'min'), thickness_max=('Thickness_original_unit', 'max'))
    collisions[collisions.distinct_times > 1].to_csv(OUT / 'four_input_collisions.csv', encoding='utf-8-sig')

    audit = {
        'source_files': {name: str(path) for name, path in paths.items()},
        'trace_rows': len(trace), 'summary_wafers': len(summary), 'tracking_wafers': len(tracking),
        'thickness_rows': len(thickness), 'gof_rows_excluded': int(measurements.ITEM_ID.eq('GOF_1').sum()),
        'missing_thickness_wafers': len(summary) - len(thickness),
        'lots': joined.LOT_ID.value_counts().to_dict(), 'recipes': int(joined.RECIPE_ID.nunique()),
        'chambers': joined.CH_ID.value_counts().to_dict(),
        'original_trace_date_range': [str(old_trace.datetime.min()), str(old_trace.datetime.max())],
        'aligned_trace_date_range': [str(trace.datetime.min()), str(trace.datetime.max())],
        'summary_recalculation_max_abs_diff': summary_diff,
        'thickness_statistics_max_abs_diff': differences,
        'depo_sampling_one_second': bool(intervals.all()), 'recipe_mismatches': recipe_mismatch,
        'unique_setting_counts': joined[feature_sets['recipe_5_with_time']].nunique().to_dict(),
        'target_ranges': {t: [float(joined[t].min()), float(joined[t].max())] for t in target_cols + ['CV_pct']},
        'four_input_colliding_groups': int((collisions.distinct_times > 1).sum()),
        'thickness_unit': 'unconfirmed: original stored unit, not assumed nm',
        'nonuniformity_definition': '100 * (max(point1..43)-min(point1..43)) / (2*mean(point1..43)); lower is more uniform; provisional diagnostic',
        'limitations': ['50 labeled wafers in 4 lots; exploratory cross-validation only',
                        'Trace summary contains no wafer outcome labels',
                        'Trace actual means are post-process measurements, not known future settings',
                        'No model activated; original workbooks unchanged'],
    }
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(metrics[metrics.split.eq('leave_one_lot_out')].to_string(index=False))


if __name__ == '__main__':
    main()
