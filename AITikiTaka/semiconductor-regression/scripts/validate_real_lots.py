"""Development-only leave-one-Lot-out evaluation. Never activates a model.

Run: python scripts/validate_real_lots.py --data /path/to/source.xlsx --output /path/to/report
Thickness units are retained as unconfirmed original values. All three candidate
models use the app's fixed hyperparameters. These folds are for development
comparison, not an untouched final test after model selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

FEATURES = ['power', 'pressure', 'temperature', 'gas', 'time']
TARGETS = ['thickness', 'nonuniformity']


def read_source(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    def df(name):
        rows = list(book[name].values)
        return pd.DataFrame(rows[1:], columns=rows[0])
    measured, tracking = df('두께'), df('Tracking')
    points = list(range(1, 44))
    selected = measured[measured.ITEM_ID.eq('THK1_1_TOP')].copy()
    a = selected[points].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    if not len(a) or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError('Thickness points must be positive and finite; no silent drops.')
    selected['thickness'] = a.mean(axis=1)
    selected['nonuniformity'] = 100 * np.ptp(a, axis=1) / (2 * a.mean(axis=1))
    if not np.allclose(selected['AVG'].to_numpy(float), selected.thickness, atol=1e-6, rtol=1e-8):
        raise ValueError('Recomputed means disagree with source AVG.')
    rows = list(book['조건표'].values)
    if list(rows[0][12:14]) != ['Recipe name', 'Dep_recipe']:
        raise ValueError('Unexpected recipe table layout.')
    recipe = pd.DataFrame([r[12:18] for r in rows[3:25]],
                          columns=['RECIPE_ID', 'gas', 'pressure', 'power', 'temperature', 'time'])
    data = selected.merge(tracking, left_on=['LOT_ID', 'WF'], right_on=['LOT_ID', 'WAFER_ID'],
                          how='left', validate='one_to_one', indicator=True)
    if not data['_merge'].eq('both').all():
        raise ValueError('Missing wafer tracking.')
    data = data.drop(columns='_merge').merge(recipe, on='RECIPE_ID', how='left', validate='many_to_one')
    if data[FEATURES+TARGETS].isna().any().any():
        raise ValueError('Missing recipe/input/output values.')
    data = data.rename(columns={'LOT_ID':'lot', 'WF':'wafer', 'RECIPE_ID':'recipe',
                                'EQP_ID':'equipment', 'CH_ID':'chamber'})
    data = data[['lot','wafer','recipe','equipment','chamber']+FEATURES+TARGETS].reset_index(drop=True)
    gof = measured[measured.ITEM_ID.eq('GOF_1')][points].apply(pd.to_numeric,errors='coerce')
    audit = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 measured_wafers=len(data), measured_points=int(a.size), tracking_wafers=len(tracking),
                 gof_values_above_one=int((gof>1).sum().sum()),
                 excluded_item='GOF_1 (not a thickness target)',
                 units={'thickness':'원본 수치 단위 (미확인)', 'nonuniformity':'%', 'error_nonuniformity':'%p'})
    book.close()
    return data, audit


def folds(data):
    if data.duplicated(['lot','wafer']).any():
        raise ValueError('Duplicate wafer identities.')
    if data.lot.nunique()<2:
        raise ValueError('At least two lots are required.')
    signatures = pd.util.hash_pandas_object(data[FEATURES+TARGETS], index=False)
    for lot in sorted(data.lot.unique()):
        test = data[data.lot.eq(lot)]
        # A byte-identical numeric record must not be memorized across folds.
        train = data[~data.lot.eq(lot) & ~signatures.isin(signatures.loc[test.index])]
        if len(train)<2:
            raise ValueError('Insufficient independent training records.')
        yield lot, train, test


def metrics(actual, predicted):
    err=np.abs(np.asarray(actual)-np.asarray(predicted))
    return dict(n=len(err), MAE=float(err.mean()), RMSE=float(np.sqrt(mean_squared_error(actual,predicted))),
                MaxAE=float(err.max()), R2=float(r2_score(actual,predicted)) if len(err)>1 and np.var(actual)>0 else None)


def evaluate(data, get_models):
    output, split_rows = [], []
    for lot, train, test in folds(data):
        split_rows.append(dict(held_out_lot=lot, train_lots=sorted(train.lot.unique()),
                               train_n=len(train), test_n=len(test)))
        candidates=get_models()
        candidates['Baseline']=DummyRegressor(strategy='mean')
        for name, model in candidates.items():
            pipe=Pipeline([('scale',StandardScaler()),('model',model)])
            with threadpool_limits(limits=2):
                pipe.fit(train[FEATURES].to_numpy(),train[TARGETS].to_numpy())
                pred=pipe.predict(test[FEATURES].to_numpy())
            if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions.')
            for j, (_,r) in enumerate(test.iterrows()):
                outside=[f for f in FEATURES if not train[f].min()<=r[f]<=train[f].max()]
                seen=bool((train[FEATURES].eq(r[FEATURES]).all(axis=1)).any())
                for k,target in enumerate(TARGETS):
                    output.append(dict(model=name,lot=lot,wafer=int(r.wafer),recipe=r.recipe,chamber=r.chamber,
                        target=target,actual=float(r[target]),predicted=float(pred[j,k]),
                        error=float(pred[j,k]-r[target]),absolute_error=float(abs(pred[j,k]-r[target])),
                        outside_features=','.join(outside),seen_condition=seen))
    out=pd.DataFrame(output)
    def summarize(keys):
        return pd.DataFrame([dict(zip(keys,key if isinstance(key,tuple) else (key,)),**metrics(g.actual,g.predicted))
                             for key,g in out.groupby(keys,sort=False)])
    return out, summarize(['model','target']),summarize(['model','lot','target']),split_rows


def plots(out,summary,bylot,dest,font):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    names=['LinearRegression','RandomForest','XGBoost','Baseline']
    colors=['#7f9fb6','#8eb6a6','#b1a2c3','#bcc1c9']
    labels={'thickness':'평균 두께','nonuniformity':'반범위 불균일도'}
    units={'thickness':'원본 두께 단위','nonuniformity':'퍼센트포인트 (%p)'}
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
    fig.suptitle('실제 원본 4개 Lot 교차검증 · 모델별 평균 오차',fontsize=20)
    for ax,target in zip(axes,TARGETS):
        s=summary[summary.target.eq(target)].set_index('model').loc[names]
        ax.bar(range(4),s.MAE,color=colors)
        for i,v in enumerate(s.MAE):ax.text(i,v,f'{v:,.3f}',ha='center',va='bottom',fontsize=12)
        ax.set(xticks=range(4),xticklabels=['선형회귀','랜덤포레스트','XGBoost','평균값만 예측'],ylabel='MAE · '+units[target],title=labels[target]+' · 낮을수록 좋음')
        ax.set_ylim(0,s.MAE.max()*1.2)
    fig.supxlabel('각 웨이퍼는 자신이 속한 Lot을 학습에서 제외한 모델로 예측했습니다.\n개발용 비교 결과이며, 모델 선택 후 새 Lot으로 최종 검증해야 합니다.',fontsize=11)
    fig.savefig(dest/'01_모델별_오차.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    lots=sorted(out.lot.unique()); palette=['#6f98af','#9bb99a','#bc9bba','#d5aa76']
    fig.suptitle('실제값과 예측값 · 점이 대각선에 가까울수록 정확',fontsize=20)
    for row,target in enumerate(TARGETS):
        subset=out[out.target.eq(target)&~out.model.eq('Baseline')]
        lo=min(subset.actual.min(),subset.predicted.min());hi=max(subset.actual.max(),subset.predicted.max());pad=(hi-lo)*.06
        for col,name in enumerate(names[:3]):
            ax=axes[row,col];s=subset[subset.model.eq(name)]
            ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],ls='--',color='#9fa8b5',label='완전 일치')
            for lot,color in zip(lots,palette):
                z=s[s.lot.eq(lot)];ax.scatter(z.actual,z.predicted,c=color,label=lot,s=28,alpha=.85)
            score=summary[summary.model.eq(name)&summary.target.eq(target)].iloc[0]
            ax.set(title=f'{name} · {labels[target]}\nMAE {score.MAE:.3f} / R² {score.R2:.3f}',xlabel='실제값',ylabel='예측값',xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad))
            if row==0 and col==0:ax.legend(fontsize=8)
    fig.supxlabel('두께는 단위 미확인 원본 수치, 불균일도는 %. R²는 정확도 백분율이 아닙니다.',fontsize=11)
    fig.savefig(dest/'02_실제값_예측값.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained');fig.suptitle('어느 Lot에서 크게 틀렸나? · Lot별 MAE',fontsize=20)
    for ax,target in zip(axes,TARGETS):
        s=bylot[bylot.target.eq(target)].pivot(index='model',columns='lot',values='MAE').reindex(names)
        im=ax.imshow(s.values,cmap='YlOrRd',aspect='auto')
        ax.set(xticks=range(len(lots)),xticklabels=lots,yticks=range(4),yticklabels=names,title=labels[target]+' ('+units[target]+')')
        for i in range(4):
            for j in range(len(lots)):ax.text(j,i,f'{s.iloc[i,j]:.3f}',ha='center',va='center',color='white' if s.iloc[i,j]>s.values.max()*.6 else '#263849')
    fig.supxlabel('Lot마다 실험한 조건이 달라, 새 Lot 효과와 처음 보는 조건의 효과가 함께 나타납니다.',fontsize=11)
    fig.savefig(dest/'03_Lot별_오차.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--font',default='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    args=p.parse_args();sys.path.insert(0,str(args.project))
    from src.models import get_models
    data,audit=read_source(args.data)
    out,summary,bylot,splits=evaluate(data,get_models)
    args.output.mkdir(parents=True,exist_ok=True)
    for filename,df in [('matched_source.csv',data),('predictions.csv',out),('model_metrics.csv',summary),('lot_metrics.csv',bylot)]:
        df.to_csv(args.output/filename,index=False,encoding='utf-8-sig')
    subgroups=[]
    for group in ['recipe','chamber']:
        for key,g in out.groupby(['model','target',group]):
            subgroups.append(dict(group=group,model=key[0],target=key[1],value=key[2],**metrics(g.actual,g.predicted)))
    pd.DataFrame(subgroups).to_csv(args.output/'subgroup_metrics.csv',index=False,encoding='utf-8-sig')
    audit.update(features=FEATURES,targets=TARGETS,splits=splits,model_selection=False,activation=False,
        status='development_leave_one_lot_out',experiment_target=5,
        actual_at_or_below_five=int((data.nonuniformity<=5).sum()),
        conditions_outside_training=out[['lot','wafer','outside_features']].drop_duplicates().to_dict('records'),
        metrics=summary.to_dict('records'))
    (args.output/'validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False))
    plots(out,summary,bylot,args.output,args.font)
    print(summary.to_string(index=False));print(json.dumps(splits,ensure_ascii=False));print('Saved:',args.output)


if __name__=='__main__':main()
