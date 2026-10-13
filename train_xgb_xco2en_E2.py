"""Edit CONFIG, run in IDE. E2 cohort; training-only selection; no label trimming."""
from pathlib import Path
from datetime import datetime
import json, logging, gc, platform, hashlib
import numpy as np
import pandas as pd
import joblib
import optuna
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import e2_data as base

# ======================= CONFIG =======================
HERE = Path(__file__).resolve().parent
DATA_FILE = Path('/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl')
OUTPUT_ROOT = Path('/home/whdong/dl/E2_feature_parameter_experiments')
PARAMS_FILE = Path('/home/whdong/dl/best_params/train_xgb_xco2en_SHP_best-A01.json')
REFERENCE_FILE = HERE / 'E2_reference_predictions.csv'
DEVICE = 'cuda'  # xgboost >=2.0; single A40, serial trials.
THREADS = 8
EXPECTED_SHA256 = '72ae983629258899658de5141e60def6d9fb8e3d7774e91a22730e6253a3ff82'
EXPECTED_COUNTS = (17522, 4382)
CV_FOLDS = 3
CV_SEED = 20261010
INNER_VALID_FRACTION = 0.15
N_TRIALS_PER_FEATURE_SET = 80  # total 160 trials, each fits 3 folds
MAX_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 150
SEEDS = [42,100,2023,888,999,1234,5678,777,666,1024]
TOP_EVENTS = 100
# Repeat runs use this directory's SQLite studies; increasing trial count continues.
# None makes a fresh run. Resume only with identical configuration/code/input.
RESUME_RUN_DIR = Path('/home/whdong/dl/E2_feature_parameter_experiments/run_20261011_155530_014013') 
# Time columns below are ELAPSED UTC seconds since 1993, per original preparation.
# Do not reinterpret as physical TAI solely because of their names.
TIME_ORIGIN = '1993-01-01'
NO2_TIME_COL = 'no2_time'  # missing -> unresolved in audit, never guessed
# ======================================================


def save_csv(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig')


def build_folds(n):
    folds=[]
    for k,(tr,va) in enumerate(KFold(CV_FOLDS,shuffle=True,random_state=CV_SEED).split(np.arange(n))):
        fit,stop=train_test_split(tr,test_size=INNER_VALID_FRACTION,random_state=CV_SEED+k)
        assert not set(fit)&set(stop) and not set(tr)&set(va)
        folds.append((fit,stop,va,tr))
    return folds


def model_params(params,seed,rounds,early=False):
    p={k:v for k,v in params.items() if k not in ['n_estimators','callbacks','early_stopping_rounds','device','n_jobs','random_state','tree_method']}
    p.update(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',device=DEVICE,n_jobs=THREADS,
             random_state=seed,n_estimators=int(rounds))
    if early:p['early_stopping_rounds']=EARLY_STOPPING_ROUNDS
    return p


# def cv_evaluate(train,features,params,folds,early):
#     X=train[features].to_numpy(float);y=train[base.TARGET].to_numpy(float)
#     oof=np.full(len(y),np.nan);rounds=[];rows=[]
#     for k,(fit,stop,va,tr) in enumerate(folds):
#         # Early stopping sees an inner validation set; outer fold stays unseen.
#         use=fit if early else tr
#         scaler=StandardScaler().fit(X[use]);a=scaler.transform(X[use]);v=scaler.transform(X[va])
#         model=xgb.XGBRegressor(**model_params(params,42,MAX_ROUNDS if early else params['n_estimators'],early))
#         if early:
#             model.fit(a,y[use],eval_set=[(scaler.transform(X[stop]),y[stop])],verbose=False)
#             r=int(model.best_iteration)+1
#         else:
#             model.fit(a,y[use],verbose=False);r=int(params['n_estimators'])
#         oof[va]=model.predict(v);rounds.append(r)
#         rows.append(dict(fold=k,fit_n=len(use),stop_n=len(stop) if early else 0,score_n=len(va),rounds=r,
#                          **base.metrics(y[va],oof[va])))
#         del model;gc.collect()
#     assert np.isfinite(oof).all()
#     return base.metrics(y,oof),rounds,oof,rows
def cv_evaluate(train, features, params, folds, early):
    X = train[features].to_numpy(float)
    y = train[base.TARGET].to_numpy(float)

    oof = np.full(len(y), np.nan)
    rounds = []
    rows = []

    for k, (fit, stop, va, tr) in enumerate(folds):
        # fit、stop均属于外折训练部分tr。
        # va只负责最终评分，不参与早停或拟合。
        assert not np.intersect1d(tr, va).size
        assert not np.intersect1d(fit, stop).size

        if early:
            # 第一步：内部早停，仅确定本折最佳轮数。
            inner_scaler = StandardScaler().fit(X[fit])

            inner_model = xgb.XGBRegressor(
                **model_params(
                    params,
                    seed=42,
                    rounds=MAX_ROUNDS,
                    early=True,
                )
            )

            inner_model.fit(
                inner_scaler.transform(X[fit]),
                y[fit],
                eval_set=[
                    (inner_scaler.transform(X[stop]), y[stop])
                ],
                verbose=False,
            )

            r = int(inner_model.best_iteration) + 1

            del inner_model, inner_scaler
            gc.collect()

        else:
            # 固定参数对照使用预先指定的轮数。
            r = int(params["n_estimators"])

        # 第二步：两类方案均使用完整外折训练部分重新拟合。
        # 这里重新拟合标准化器，纳入此前用于早停的样本。
        outer_scaler = StandardScaler().fit(X[tr])

        outer_model = xgb.XGBRegressor(
            **model_params(
                params,
                seed=42,
                rounds=r,
                early=False,
            )
        )

        outer_model.fit(
            outer_scaler.transform(X[tr]),
            y[tr],
            verbose=False,
        )

        # 第三步：外折评分样本只在此处用于预测和计算指标。
        oof[va] = outer_model.predict(
            outer_scaler.transform(X[va])
        )

        rounds.append(r)

        rows.append(
            dict(
                fold=k,
                fit_n=len(tr),
                stop_n=len(stop) if early else 0,
                early_stop_fit_n=len(fit) if early else 0,
                score_n=len(va),
                rounds=r,
                **base.metrics(y[va], oof[va]),
            )
        )

        del outer_model, outer_scaler
        gc.collect()

    assert np.isfinite(oof).all()

    return base.metrics(y, oof), rounds, oof, rows

def sample_params(t):
    return dict(learning_rate=t.suggest_float('learning_rate',.008,.10,log=True),
        max_depth=t.suggest_int('max_depth',3,12),min_child_weight=t.suggest_float('min_child_weight',1,100,log=True),
        subsample=t.suggest_float('subsample',.60,1.0),colsample_bytree=t.suggest_float('colsample_bytree',.55,1.0),
        gamma=t.suggest_float('gamma',0,.8),reg_alpha=t.suggest_float('reg_alpha',1e-5,15,log=True),
        reg_lambda=t.suggest_float('reg_lambda',.1,80,log=True))


def audit_events(raw,predictions,folder,label):
    folder.mkdir(parents=True,exist_ok=True)
    p=predictions.copy();p['error']=p.prediction-p[base.TARGET];p['abs_error']=p.error.abs()
    p['squared_error']=p.error**2;p=p.sort_values('abs_error',ascending=False).reset_index(drop=True)
    p['error_rank']=np.arange(1,len(p)+1)
    ids=p._row_id.to_numpy(dtype=int)
    original=raw.loc[ids].reset_index(drop=True).copy()
    # Prefix every original column: original timestamps/NO2/background remain intact.
    event=pd.concat([p.reset_index(drop=True),original.add_prefix('raw__')],axis=1)
    def numeric(c):
        return pd.to_numeric(original[c],errors='coerce') if c in original else pd.Series(np.nan,index=original.index)
    count=numeric('total_obs_count');event['audit_obs_count_le_1']=count.le(1)
    event['audit_obs_count_missing']=count.isna()
    event['audit_no2_missing_or_nonpositive']=numeric('no2_trop').le(0)|numeric('no2_trop').isna()
    delta=numeric(NO2_TIME_COL)-numeric('xco2_time_tai93')
    event['audit_no2_minus_xco2_seconds']=delta
    event['audit_no2_time_unresolved']=delta.isna()
    event['audit_abs_no2_time_over_3600s']=delta.abs().gt(3600)
    xt=numeric('xco2_time_tai93')
    event['audit_xco2_utc_assuming_elapsed1993']=pd.to_datetime(xt,unit='s',origin=TIME_ORIGIN,errors='coerce')
    bg=[c for c in raw.columns if any(s in str(c).lower() for s in ['background','baseline','bg_xco2','xco2_bg'])]
    save_csv(event,folder/f'{label}_all_events_raw.csv')
    save_csv(event.head(TOP_EVENTS),folder/f'{label}_top{TOP_EVENTS}_raw.csv')
    if 'total_obs_count' in original:
        g=pd.cut(count,[-np.inf,1,3,10,30,np.inf],labels=['<=1','2-3','4-10','11-30','>30'])
        stats=[]
        for name,sub in p.groupby(g,observed=True):
            stats.append(dict(obs_count_bin=str(name),**base.metrics(sub[base.TARGET],sub.prediction)))
        save_csv(pd.DataFrame(stats),folder/f'{label}_metrics_by_obs_count.csv')
    base.write_json(folder/f'{label}_audit_notes.json',dict(
        background_columns_found=bg,background_status='Original background construction cannot be validated from a table alone. Compare with original soundings and background-selection code.',
        no2_time_column=NO2_TIME_COL,no2_time_column_present=NO2_TIME_COL in raw,
        time_assumption=f'Elapsed UTC seconds since {TIME_ORIGIN}; verify preparation definitions before interpreting differences.',
        automatic_row_deletions=0,warning='Flags request source review, not exclusion. High labels/large errors are never deletion rules. Counts/timestamps/background are audit-only, never model inputs.'))


def train_ensemble(train,test,features,params,rounds,out):
    out.mkdir(parents=True,exist_ok=True)
    scaler=StandardScaler().fit(train[features].to_numpy(float))
    X=scaler.transform(train[features].to_numpy(float));T=scaler.transform(test[features].to_numpy(float))
    joblib.dump(scaler,out/'scaler.pkl');preds=[]
    for seed in SEEDS:
        m=xgb.XGBRegressor(**model_params(params,seed,rounds))
        m.fit(X,train[base.TARGET].to_numpy(float),verbose=False)
        preds.append(m.predict(T));m.get_booster().save_model(str(out/f'model_seed{seed}.json'))
        del m;gc.collect()
    base.write_json(out/'manifest.json',dict(features=features,params=params,n_estimators=rounds,seeds=SEEDS,
        scaler='scaler.pkl',prediction='Apply feature engineering from e2_data.load_and_preprocess, order features, transform scaler, average all saved seed models.',
        note='Full E2 train refit uses training-CV median early-stop rounds; test is never eval_set. seed_std is not an uncertainty interval.'))
    p=test[['_row_id','date','grid_lat','grid_lon',base.TARGET]].copy()
    p['prediction']=np.mean(preds,axis=0);p['seed_std']=np.std(preds,axis=0)
    for seed,v in zip(SEEDS,preds):p[f'pred_seed{seed}']=v
    save_csv(p,out/'common_test_predictions.csv')
    return p


def main():
    digest=base.sha256(DATA_FILE)
    if EXPECTED_SHA256 and digest!=EXPECTED_SHA256:
        raise ValueError('PKL hash differs from verified E2 input. Do not bypass unless intentionally starting a new, non-comparable cohort.')
    df,old,satellite,model=base.load_and_preprocess(DATA_FILE)
    plans,tests,membership=base.make_plan(df,old,satellite,model)
    train=df.loc[plans['E2_required_satellite']].copy();test=df.loc[tests['combined_satellite_holdout']].copy()
    if EXPECTED_COUNTS and (len(train),len(test))!=EXPECTED_COUNTS:raise ValueError('E2 membership count mismatch')
    assert not train._model_filled.any() and not test._model_filled.any()
    assert not set(train._row_id)&set(test._row_id)
    if REFERENCE_FILE:
        ref=pd.read_csv(REFERENCE_FILE)
        if set(ref._row_id)!=set(test._row_id):raise ValueError('Reference test membership mismatch')
        aligned=ref.set_index('_row_id').loc[test._row_id]
        for c in [base.TARGET,'grid_lat','grid_lon']:
            if not np.allclose(aligned[c].to_numpy(),test[c].to_numpy(),rtol=1e-6,atol=1e-6):raise ValueError(f'Reference {c} mismatch')
    out=Path(RESUME_RUN_DIR) if RESUME_RUN_DIR else OUTPUT_ROOT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=RESUME_RUN_DIR is not None)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'training.log',encoding='utf-8')])
    signature=dict(data_sha256=digest,script_sha256=hashlib.sha256('\n'.join(line for line in Path(__file__).read_text().splitlines() if not line.startswith(('RESUME_RUN_DIR =', 'N_TRIALS_PER_FEATURE_SET ='))).encode()).hexdigest(),helper_sha256=base.sha256(base.__file__),
        params_sha256=base.sha256(PARAMS_FILE),cv_folds=CV_FOLDS,cv_seed=CV_SEED,inner_fraction=INNER_VALID_FRACTION,
        max_rounds=MAX_ROUNDS,patience=EARLY_STOPPING_ROUNDS,seeds=SEEDS,device=DEVICE,
        xgboost=xgb.__version__,optuna=optuna.__version__,python=platform.python_version())
    guard=out/'run_signature.json'
    if guard.exists() and json.loads(guard.read_text())!=signature:raise ValueError('Resume signature changed; start a fresh run')
    base.write_json(guard,signature);save_csv(membership,out/'split_membership.csv')
    folds=build_folds(len(train));fold_table=[]
    for k,(fit,stop,va,tr) in enumerate(folds):
        for role,ix in [('inner_fit',fit),('inner_early_stop',stop),('outer_score',va)]:
            fold_table.extend(dict(fold=k,role=role,row_id=int(r)) for r in train._row_id.to_numpy()[ix])
    save_csv(pd.DataFrame(fold_table),out/'training_cv_membership.csv')
    oldp=json.loads(PARAMS_FILE.read_text());results=[];details={}
    for tag,features in [('20',base.FIXED_FEATURES),('45',base.CANDIDATES)]:
        logging.info('Fixed %s features: training CV only',tag)
        score,rounds,oof,fr=cv_evaluate(train,features,oldp,folds,False)
        name=f'fixed{tag}_oldparams';results.append(dict(candidate=name,**score));details[name]=(features,oldp,rounds,oof)
        save_csv(pd.DataFrame(fr),out/f'{name}_fold_metrics.csv')
        study=optuna.create_study(study_name=f'tuned{tag}',direction='minimize',storage='sqlite:///'+str((out/'studies.sqlite3').resolve()),
            load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=CV_SEED))
        if not study.trials:
            keys=['learning_rate','max_depth','min_child_weight','subsample','colsample_bytree','gamma','reg_alpha','reg_lambda']
            study.enqueue_trial({k:oldp[k] for k in keys})
        def objective(t):
            p=sample_params(t)
            score,rs,_,_=cv_evaluate(train,features,p,folds,True)
            t.set_user_attr('rounds',rs);t.set_user_attr('oof_r2',score['r2'])
            logging.info('features=%s trial=%d CV RMSE=%.5f R2=%.5f rounds=%s',tag,t.number,score['rmse'],score['r2'],rs)
            return score['rmse']
        completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        study.optimize(objective,n_trials=max(0,N_TRIALS_PER_FEATURE_SET-completed),n_jobs=1,gc_after_trial=True)
        p=study.best_params;score,rs,oof,fr=cv_evaluate(train,features,p,folds,True)
        name=f'tuned{tag}_earlystop';results.append(dict(candidate=name,**score));details[name]=(features,p,rs,oof)
        save_csv(study.trials_dataframe(),out/f'{name}_trials.csv');save_csv(pd.DataFrame(fr),out/f'{name}_fold_metrics.csv')
    ranking=pd.DataFrame(results).sort_values('rmse');save_csv(ranking,out/'training_cv_comparison.csv')
    selected=str(ranking.iloc[0].candidate)
    features,params,rs,oof=details[selected];rounds=int(np.median(rs))
    base.write_json(out/'selection_before_test.json',dict(selected=selected,features=features,params=params,final_rounds=rounds,
        fold_rounds=rs,rule='Lowest pooled outer training-CV RMSE. No test metric used.',
        caveat='CV scores were used in hyperparameter search; selection scores are optimistic, not independent generalization estimates. Existing common test has already been examined and is a development reference.'))
    logging.info('Selected BEFORE test evaluation: %s, rounds=%d',selected,rounds)
    raw=pd.read_pickle(DATA_FILE).reset_index(drop=True)
    poof=train[['_row_id','date','grid_lat','grid_lon',base.TARGET]].copy();poof['prediction']=oof
    audit_events(raw,poof,out/'audit_training_oof','training_oof')
    # Baseline reproduction and chosen candidate evaluated only after locked selection.
    baseline=train_ensemble(train,test,base.FIXED_FEATURES,oldp,int(oldp['n_estimators']),out/'E2_baseline')
    if selected=='fixed20_oldparams':chosen=baseline.copy()
    else:chosen=train_ensemble(train,test,features,params,rounds,out/'selected_model')
    comparison=[]
    for name,pred in [('E2_baseline',baseline),(selected,chosen)]:
        for subset,ids in tests.items():
            sub=pred.loc[pred._row_id.isin(ids)]
            comparison.append(dict(model=name,subset=subset,**base.metrics(sub[base.TARGET],sub.prediction)))
    save_csv(pd.DataFrame(comparison),out/'test_comparison.csv')
    if REFERENCE_FILE:
        b=baseline.set_index('_row_id');r=ref.set_index('_row_id').loc[b.index]
        base.write_json(out/'baseline_reproduction_check.json',dict(max_abs_prediction_diff=float(np.max(np.abs(b.prediction-r.prediction))),
            original_metrics=base.metrics(r[base.TARGET],r.prediction),reproduced_metrics=base.metrics(b[base.TARGET],b.prediction),
            note='Small floating differences can reflect GPU/software versions; cohort checked separately.'))
    audit_events(raw,baseline,out/'audit_test','baseline_test')
    audit_events(raw,chosen,out/'audit_test','selected_test')
    by_year=[]
    for name,pred in [('baseline',baseline),('selected',chosen)]:
        for yr,sub in pred.groupby(pred.date.dt.year):by_year.append(dict(model=name,year=yr,**base.metrics(sub[base.TARGET],sub.prediction)))
    save_csv(pd.DataFrame(by_year),out/'test_by_year.csv')
    logging.info('\n%s',pd.DataFrame(comparison).to_string(index=False))
    logging.info('Finished %s. No rows deleted for labels/errors; raw sources require manual verification.',out)

if __name__=='__main__':main()
