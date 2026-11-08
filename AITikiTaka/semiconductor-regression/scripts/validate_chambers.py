"""Paired development comparison: same leave-Lot-out splits, add chamber only.
Does not activate models or modify source workbooks. Run next to validate_real_lots.py.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from validate_real_lots import read_source, folds, FEATURES, TARGETS, metrics


def condition_group(train, row):
    if train[FEATURES].eq(row[FEATURES]).all(axis=1).any():
        return 'seen'
    if any(not train[f].min() <= row[f] <= train[f].max() for f in FEATURES):
        return 'outside'
    return 'new_combination'


def compare(data, get_models):
    rows, audits = [], []
    for lot, train, test in folds(data):
        assert set(train.lot).isdisjoint(test.lot)
        assert set(test.chamber).issubset(set(train.chamber))
        audits.append(dict(lot=lot, train_n=len(train), test_n=len(test),
                           train_lots=sorted(train.lot.unique()),
                           train_chambers=sorted(train.chamber.unique())))
        for variant in ['before', 'chamber']:
            transformers=[('numeric', StandardScaler(), FEATURES)]
            if variant=='chamber':
                transformers.append(('chamber', OneHotEncoder(drop='first',handle_unknown='error',sparse_output=False), ['chamber']))
            for name, model in get_models().items():
                pipe=Pipeline([('prepare',ColumnTransformer(transformers)),('model',model)])
                with threadpool_limits(limits=2):
                    pipe.fit(train,train[TARGETS].to_numpy())
                    predictions=pipe.predict(test)
                assert np.allclose(pipe['prepare'].named_transformers_['numeric'].mean_,train[FEATURES].mean())
                for i,(_,r) in enumerate(test.iterrows()):
                    for j,target in enumerate(TARGETS):
                        rows.append(dict(variant=variant,model=name,lot=lot,wafer=int(r.wafer),recipe=r.recipe,
                                         chamber=r.chamber,condition=condition_group(train,r),target=target,
                                         actual=float(r[target]),predicted=float(predictions[i,j])))
    out=pd.DataFrame(rows)
    out['absolute_error']=(out.actual-out.predicted).abs()
    assert len(out)==len(data)*3*2*2
    assert not out.duplicated(['variant','model','lot','wafer','target']).any()
    assert np.isfinite(out.predicted).all()
    return out,audits


def summarize(out,keys):
    return pd.DataFrame([dict(zip(keys,key),**metrics(g.actual,g.predicted)) for key,g in out.groupby(keys,sort=False)])


def audit_trace(data, path, dest):
    book=openpyxl.load_workbook(path,read_only=True,data_only=True)
    rows=list(book['Wafer_Depo_Summary'].values);book.close()
    trace=pd.DataFrame(rows[1:],columns=rows[0]).rename(columns={
        'CTC_chA_LotID':'lot','CTC_chA_WaferNo':'wafer','chA_VIR_Recipe_Name':'trace_recipe'})
    joined=data.merge(trace,on=['lot','wafer'],how='left',validate='one_to_one',indicator=True)
    assert joined._merge.eq('both').all()
    assert joined.recipe.eq(joined.trace_recipe).all()
    for a,b in [('gas','Si2H6_Setpoint'),('temperature','Temp_Setpoint'),('time','Depo_Duration_s')]:
        assert np.allclose(joined[a],joined[b])
    joined.to_csv(dest/'trace_join_audit.csv',index=False,encoding='utf-8-sig')
    pairs=[]
    for recipe,g in data.groupby('recipe'):
        if len(g)<2:continue
        pairs.append(dict(recipe=recipe,n=len(g),lots=g.lot.nunique(),chambers=g.chamber.nunique(),
                          thickness_min=g.thickness.min(),thickness_max=g.thickness.max(),
                          thickness_range=np.ptp(g.thickness),nu_range=np.ptp(g.nonuniformity)))
    pd.DataFrame(pairs).sort_values('thickness_range',ascending=False).to_csv(dest/'same_recipe_spread.csv',index=False,encoding='utf-8-sig')
    joined[joined.recipe.eq('COMPLEX_1_4')].to_csv(dest/'COMPLEX_1_4_audit.csv',index=False,encoding='utf-8-sig')
    return dict(trace_matches=len(joined),recipe_matches=True,gas_temp_time_setpoints_match=True,
                equipment=sorted(data.equipment.unique()),chamber_counts=data.chamber.value_counts().to_dict())


def plots(summary,cond,lotmetrics,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
                         'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    models=['LinearRegression','RandomForest','XGBoost']
    names=['선형회귀','랜덤포레스트','XGBoost']
    titles=['평균 두께 · 원본 단위','반범위 불균일도 · %p']
    colors=['#aebcd0','#86b6a5']
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    fig.suptitle('챔버 정보 추가 전후 · 평균 오차 비교',fontsize=21)
    for ax,target,title in zip(axes,TARGETS,titles):
        s=summary[summary.target.eq(target)].pivot(index='model',columns='variant',values='MAE').loc[models]
        for k,v in enumerate(['before','chamber']):
            bars=ax.bar(np.arange(3)+(k-.5)*.34,s[v],width=.34,color=colors[k],label=['기존 5개 입력','기존 입력 + 챔버'][k])
            ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=10)
        ax.set(xticks=range(3),xticklabels=names,ylabel='MAE · 낮을수록 좋음',title=title,ylim=(0,s.values.max()*1.24))
        ax.legend(fontsize=10)
    fig.supxlabel('동일한 50개 웨이퍼·4개 Lot 분리·모델 설정. PM1/PM2만 입력에 추가했습니다.\n개발용 비교 결과이며, 새 데이터에서의 성능 향상을 보장하지 않습니다.',fontsize=11)
    fig.savefig(dest/'01_챔버_추가_전후.png',dpi=160);plt.close(fig)
    groups=['seen','new_combination','outside']; labels=['동일 조건 경험 있음','새 조합·개별 범위 안','학습 범위 밖']
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained');fig.suptitle('어떤 조건에서 틀렸나? · 조건별 평균 오차',fontsize=20)
    for i,target in enumerate(TARGETS):
        for j,model in enumerate(models):
            ax=axes[i,j];s=cond[cond.target.eq(target)&cond.model.eq(model)]
            p=s.pivot(index='condition',columns='variant',values='MAE').reindex(groups)
            n=s[s.variant.eq('before')].set_index('condition').n.reindex(groups).fillna(0)
            for k,v in enumerate(['before','chamber']):
                bars=ax.bar(np.arange(3)+(k-.5)*.34,p[v],width=.34,color=colors[k],label=['기존','챔버 추가'][k]);ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
            ax.set(xticks=range(3),xticklabels=[f'{l}\n{int(c)}개' for l,c in zip(labels,n)],title=f'{names[j]} · {titles[i]}',ylabel='MAE')
            ax.tick_params(axis='x',labelsize=9);ax.margins(y=.2)
            if i==0 and j==0:ax.legend(fontsize=9)
    fig.supxlabel('조건 구분은 각 시험 Lot을 제외한 학습 데이터의 5개 공정 입력으로 판단합니다.\n범위 안의 새 조합도 학습된 조건과 동일하다는 뜻은 아닙니다.',fontsize=11)
    fig.savefig(dest/'02_조건별_성능.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');fig.suptitle('Lot마다 챔버 추가 효과가 같은가?',fontsize=20)
    for i,target in enumerate(TARGETS):
        for j,model in enumerate(models):
            ax=axes[i,j];s=lotmetrics[lotmetrics.target.eq(target)&lotmetrics.model.eq(model)].pivot(index='lot',columns='variant',values='MAE')
            for k,v in enumerate(['before','chamber']):
                bars=ax.bar(np.arange(4)+(k-.5)*.34,s[v],width=.34,color=colors[k],label=['기존','챔버 추가'][k]);ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
            ax.set(xticks=range(4),xticklabels=s.index,title=f'{names[j]} · {titles[i]}',ylabel='MAE');ax.margins(y=.2)
            if i==0 and j==0:ax.legend(fontsize=9)
    fig.supxlabel('각 Lot 전체를 시험용으로 남겼습니다. Lot 효과와 조건 차이가 함께 반영됩니다.',fontsize=11)
    fig.savefig(dest/'03_Lot별_변화.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.project))
    from src.models import get_models
    data,audit=read_source(a.data)
    audit.update(audit_trace(data,a.trace,a.output))
    out,splits=compare(data,get_models)
    if a.previous:
        old=pd.read_csv(a.previous).query("model != 'Baseline'")
        joined=out[out.variant.eq('before')].merge(old,on=['model','lot','wafer','target'],suffixes=('_new','_old'),validate='one_to_one')
        assert len(joined)==300 and np.allclose(joined.predicted_new,joined.predicted_old,rtol=1e-6,atol=1e-6)
    summary=summarize(out,['variant','model','target']);cond=summarize(out,['variant','model','target','condition']);lots=summarize(out,['variant','model','target','lot'])
    change=summary.pivot(index=['model','target'],columns='variant',values='MAE').reset_index()
    change['MAE_reduction_percent']=100*(change.before-change.chamber)/change.before
    for f,d in [('predictions',out),('metrics',summary),('condition_metrics',cond),('lot_metrics',lots),('change',change)]:
        d.to_csv(a.output/(f+'.csv'),index=False,encoding='utf-8-sig')
    audit.update(splits=splits,activation=False,condition_counts=out[['lot','wafer','condition']].drop_duplicates().condition.value_counts().to_dict(),
                 comparison='same fixed hyperparameters and folds; chamber one-hot fitted on training only',test_checks_passed=True)
    (a.output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    plots(summary,cond,lots,a.output)
    print(change.to_string(index=False));print(json.dumps(audit['condition_counts']));print(cond.to_string(index=False))


if __name__=='__main__':main()
