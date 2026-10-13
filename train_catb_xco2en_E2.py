"""E2 CatBoost comparison. Edit CONFIG and run in IDE; no CLI arguments."""
from pathlib import Path
from datetime import datetime
import os, json, hashlib, logging, gc, platform
import numpy as np
import pandas as pd
import joblib
import optuna
import catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import e2c_data as base
from event_audit import audit_events

# ======================= CONFIG =======================
HERE = Path(__file__).resolve().parent
DATA_FILE = Path('/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl')
OUTPUT_ROOT = Path('/home/whdong/dl/E2_CatBoost_experiments')
# Point to your corrected XGBoost run to verify exact split/CV row membership.
XGB_RUN_DIR = Path('/home/whdong/dl/E2_feature_parameter_experiments/run_20261011_155530_014013')
RESUME_RUN_DIR = Path('/home/whdong/dl/E2_CatBoost_experiments/run_20261012_134854_655383')
TASK_TYPE = 'GPU'  # Use 'CPU' for a machine without a CUDA GPU.
GPU_DEVICES = '0'
THREADS = 8
EXPECTED_SHA256 = '72ae983629258899658de5141e60def6d9fb8e3d7774e91a22730e6253a3ff82'
EXPECTED_COUNTS = (17522, 4382)
CV_FOLDS = 3
CV_SEED = 20261010
INNER_VALID_FRACTION = 0.15
N_TRIALS = 80  # Same budget as the XGBoost 45-feature search.
MAX_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 150
SEEDS = [42,100,2023,888,999,1234,5678,777,666,1024]
# Reference copies bundled from the latest uploaded XGBoost experiment.
REFERENCE_PREDICTIONS = HERE / 'XGBoost_reference_predictions.csv'
REFERENCE_CV = HERE / 'XGBoost_reference_cv.csv'
REFERENCE_SELECTION = HERE / 'XGBoost_reference_selection.json'
# ======================================================
FEATURES = base.CANDIDATES


def save_csv(df, path):
    tmp=Path(str(path)+'.tmp');df.to_csv(tmp,index=False,encoding='utf-8-sig');os.replace(tmp,path)


def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp');base.write_json(tmp,obj);os.replace(tmp,path)


def hash_object(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()


def build_folds(n):
    folds=[]
    for k,(tr,va) in enumerate(KFold(CV_FOLDS,shuffle=True,random_state=CV_SEED).split(np.arange(n))):
        fit,stop=train_test_split(tr,test_size=INNER_VALID_FRACTION,random_state=CV_SEED+k)
        assert not set(fit)&set(stop) and not set(tr)&set(va)
        folds.append((fit,stop,va,tr))
    return folds


def kwargs(params,seed,rounds):
    p=dict(params)
    p.update(iterations=int(rounds),loss_function='RMSE',eval_metric='RMSE',random_seed=seed,
        task_type=TASK_TYPE,thread_count=THREADS,allow_writing_files=False,
        grow_policy='SymmetricTree',boosting_type='Plain',nan_mode='Forbidden',verbose=False)
    if TASK_TYPE=='GPU':p.update(devices=GPU_DEVICES,gpu_ram_part=.85)
    return p


def sample_params(t):
    # GPU RMSE: do not pass unsupported rsm or CPU-only random_strength.
    return dict(depth=t.suggest_int('depth',4,10),
        learning_rate=t.suggest_float('learning_rate',.008,.1,log=True),
        l2_leaf_reg=t.suggest_float('l2_leaf_reg',.1,50,log=True),
        bootstrap_type='Bayesian',bagging_temperature=t.suggest_float('bagging_temperature',0,3),
        border_count=t.suggest_categorical('border_count',[128,254]),
        leaf_estimation_iterations=t.suggest_int('leaf_estimation_iterations',1,5))


def cv_evaluate(train,params,folds,out):
    X=train[FEATURES].to_numpy(float);y=train[base.TARGET].to_numpy(float)
    oof=np.full(len(y),np.nan);rounds=[];rows=[]
    # Cache completed folds by params, allowing an interrupted trial retry to reuse them.
    folder=out/'cv_checkpoints'/hash_object(params);folder.mkdir(parents=True,exist_ok=True)
    for k,(fit,stop,va,tr) in enumerate(folds):
        cache=folder/f'fold{k}.npz'
        if cache.exists():
            with np.load(cache) as z:
                if not np.array_equal(z['row_ids'],train._row_id.to_numpy()[va]):raise ValueError('CV cache membership changed')
                pred=z['prediction'];r=int(z['rounds'])
            logging.info('Reuse completed CV fold %d',k)
        else:
            inner_scaler=StandardScaler().fit(X[fit])
            inner=CatBoostRegressor(**kwargs(params,42,MAX_ROUNDS))
            inner.fit(inner_scaler.transform(X[fit]),y[fit],
                eval_set=(inner_scaler.transform(X[stop]),y[stop]),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,use_best_model=True,verbose=False)
            best=inner.get_best_iteration()
            r=int(best)+1 if best>=0 else int(inner.tree_count_)
            del inner,inner_scaler;gc.collect()
            # Refit ALL outer-training rows before scoring, exactly as corrected XGB.
            scaler=StandardScaler().fit(X[tr])
            model=CatBoostRegressor(**kwargs(params,42,r))
            model.fit(scaler.transform(X[tr]),y[tr],use_best_model=False,verbose=False)
            pred=model.predict(scaler.transform(X[va]),thread_count=THREADS)
            tmp=Path(str(cache)+'.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,prediction=pred,rounds=r,row_ids=train._row_id.to_numpy()[va])
            os.replace(tmp,cache)
            del model,scaler;gc.collect()
        oof[va]=pred;rounds.append(r)
        rows.append(dict(fold=k,fit_n=len(tr),stop_n=len(stop),early_stop_fit_n=len(fit),score_n=len(va),rounds=r,
            **base.metrics(y[va],pred)))
    assert np.isfinite(oof).all()
    return base.metrics(y,oof),rounds,oof,rows


def compare_membership(actual,path,columns):
    other=pd.read_csv(path)
    a=actual[columns].sort_values(columns).reset_index(drop=True)
    b=other[columns].sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False)


def main():
    digest=base.sha256(DATA_FILE)
    if EXPECTED_SHA256 and digest!=EXPECTED_SHA256:raise ValueError('Input PKL differs from verified E2 source')
    base.RUN_E3_MODEL_FILLED=False
    df,old,satellite,model=base.load_and_preprocess(DATA_FILE)
    plans,tests,membership=base.make_plan(df,old,satellite,model)
    train=df.loc[plans['E2_required_satellite']].copy();test=df.loc[tests['combined_satellite_holdout']].copy()
    if EXPECTED_COUNTS and (len(train),len(test))!=EXPECTED_COUNTS:raise ValueError('E2 counts changed')
    assert not train._model_filled.any() and not test._model_filled.any()
    assert not set(train._row_id)&set(test._row_id)
    folds=build_folds(len(train));fold_table=[]
    for k,(fit,stop,va,tr) in enumerate(folds):
        for role,ix in [('inner_fit',fit),('inner_early_stop',stop),('outer_score',va)]:
            fold_table.extend(dict(fold=k,role=role,row_id=int(r)) for r in train._row_id.to_numpy()[ix])
    fold_table=pd.DataFrame(fold_table)
    if XGB_RUN_DIR is not None:
        compare_membership(membership,Path(XGB_RUN_DIR)/'split_membership.csv',['row_id','role'])
        compare_membership(fold_table,Path(XGB_RUN_DIR)/'training_cv_membership.csv',['fold','role','row_id'])
    ref=pd.read_csv(REFERENCE_PREDICTIONS).set_index('_row_id')
    if set(ref.index)!=set(test._row_id) or ref.index.duplicated().any():raise ValueError('XGB test IDs differ')
    ref=ref.loc[test._row_id]
    for c in [base.TARGET,'grid_lat','grid_lon']:
        if not np.allclose(ref[c],test[c],rtol=1e-6,atol=1e-6):raise ValueError(f'XGB reference {c} differs')
    if not (pd.to_datetime(ref.date).to_numpy()==test.date.to_numpy()).all():raise ValueError('XGB reference dates differ')
    selected_xgb=json.loads(REFERENCE_SELECTION.read_text())
    if selected_xgb['features']!=FEATURES:raise ValueError('Feature ordering differs from XGB')
    out=Path(RESUME_RUN_DIR) if RESUME_RUN_DIR is not None else OUTPUT_ROOT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=RESUME_RUN_DIR is not None)
    # Exclusive lock: never run two copies against one resume directory.
    import fcntl  # Linux training server
    lock=(out/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another process is using this run directory')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,
        handlers=[logging.StreamHandler(),logging.FileHandler(out/'training.log',encoding='utf-8')])
    code='\n'.join(l for l in Path(__file__).read_text().splitlines() if not l.startswith(('RESUME_RUN_DIR =','N_TRIALS =')))
    signature=dict(data_sha256=digest,script_sha256=hashlib.sha256(code.encode()).hexdigest(),
        helper_sha256=base.sha256(base.__file__),audit_sha256=base.sha256(HERE/'event_audit.py'),
        cv_folds=CV_FOLDS,cv_seed=CV_SEED,inner_fraction=INNER_VALID_FRACTION,max_rounds=MAX_ROUNDS,
        patience=EARLY_STOPPING_ROUNDS,seeds=SEEDS,task_type=TASK_TYPE,gpu_devices=GPU_DEVICES,
        features=FEATURES,catboost=catboost.__version__,optuna=optuna.__version__,pandas=pd.__version__,
        sklearn=__import__('sklearn').__version__,python=platform.python_version(),
        reference_sha256=base.sha256(REFERENCE_PREDICTIONS),selection_sha256=base.sha256(REFERENCE_SELECTION))
    guard=out/'run_signature.json'
    if guard.exists() and json.loads(guard.read_text())!=signature:raise ValueError('Resume signature changed; do not delete guard; start fresh')
    atomic_json(guard,signature);save_csv(membership,out/'split_membership.csv');save_csv(fold_table,out/'training_cv_membership.csv')
    logging.info('E2 train=%d test=%d features=%d; identical CV seed %d',len(train),len(test),len(FEATURES),CV_SEED)
    study=optuna.create_study(study_name='CatBoost45',direction='minimize',storage='sqlite:///'+str((out/'studies.sqlite3').resolve()),
        load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=CV_SEED))
    # After a hard shutdown, retry unfinished params; matching fold caches are reused.
    for t in study.get_trials(deepcopy=False):
        if t.state==optuna.trial.TrialState.RUNNING:
            study.tell(t.number,state=optuna.trial.TrialState.FAIL)
            if t.params:study.enqueue_trial(t.params)
            logging.info('Interrupted trial %d marked FAIL and queued for retry',t.number)
    if not study.trials:
        study.enqueue_trial(dict(depth=8,learning_rate=.03,l2_leaf_reg=3.,bagging_temperature=1.,border_count=128,leaf_estimation_iterations=1))
    def objective(t):
        p=sample_params(t);score,rs,_,_=cv_evaluate(train,p,folds,out)
        t.set_user_attr('rounds',rs);t.set_user_attr('oof_r2',score['r2'])
        logging.info('CatBoost trial=%d CV RMSE=%.6f R2=%.6f rounds=%s',t.number,score['rmse'],score['r2'],rs)
        return score['rmse']
    def callback(s,t):save_csv(s.trials_dataframe(),out/'catboost_trials.csv')
    completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective,n_trials=max(0,N_TRIALS-completed),n_jobs=1,gc_after_trial=True,callbacks=[callback])
    params=dict(study.best_params,bootstrap_type='Bayesian')
    score,rs,oof,fr=cv_evaluate(train,params,folds,out);rounds=int(np.median(rs))
    save_csv(pd.DataFrame(fr),out/'catboost_fold_metrics.csv')
    cv=pd.read_csv(REFERENCE_CV);cv=pd.concat([cv,pd.DataFrame([dict(candidate='CatBoost45_tuned',**score)])],ignore_index=True)
    save_csv(cv,out/'training_cv_comparison.csv')
    selection=dict(model='CatBoost45_tuned',features=FEATURES,params=params,fold_rounds=rs,final_rounds=rounds,seeds=SEEDS,
        CV_metrics=score,best_trial_number=study.best_trial.number,
        note='CatBoost hyperparameters selected on training CV only. CV search score is optimistic. XGB and CatBoost test results are development comparisons, not a fresh blind test. No model/ensemble weight chosen using test.')
    atomic_json(out/'selection_before_test.json',selection)
    logging.info('CatBoost selection locked before test: trial=%d rounds=%d',study.best_trial.number,rounds)
    folder=out/'selected_model'
    model_stamp=dict(params=params,rounds=rounds,seeds=SEEDS,signature=signature)
    stamp_path=folder/'training_signature.json'
    if folder.exists() and (not stamp_path.exists() or json.loads(stamp_path.read_text())!=model_stamp):
        folder.rename(out/('selected_model_archive_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    folder.mkdir(exist_ok=True);atomic_json(folder/'training_signature.json',model_stamp)
    scaler=StandardScaler().fit(train[FEATURES].to_numpy(float));joblib.dump(scaler,folder/'scaler.pkl')
    X=scaler.transform(train[FEATURES].to_numpy(float));T=scaler.transform(test[FEATURES].to_numpy(float))
    preds=[];importances=[]
    for seed in SEEDS:
        path=folder/f'model_seed{seed}.cbm';m=CatBoostRegressor(thread_count=THREADS)
        if path.exists():m.load_model(str(path));logging.info('Reuse final seed %d',seed)
        else:
            m=CatBoostRegressor(**kwargs(params,seed,rounds));logging.info('Train final seed %d',seed)
            m.fit(X,train[base.TARGET].to_numpy(float),use_best_model=False,verbose=False)
            tmp=Path(str(path)+'.tmp');m.save_model(str(tmp),format='cbm');os.replace(tmp,path)
        preds.append(m.predict(T,thread_count=THREADS));importances.append(m.get_feature_importance(thread_count=THREADS))
        del m;gc.collect()
    predicted=test[['_row_id','date','grid_lat','grid_lon',base.TARGET]].copy()
    predicted['test_subset']=np.where(predicted._row_id.isin(tests['historical_reference']),'historical_reference','added_satellite_holdout')
    predicted['prediction']=np.mean(preds,axis=0);predicted['seed_std']=np.std(preds,axis=0)
    for seed,p in zip(SEEDS,preds):predicted[f'pred_seed{seed}']=p
    save_csv(predicted,folder/'common_test_predictions.csv')
    atomic_json(folder/'manifest.json',dict(**selection,preprocessing='e2_data feature engineering; ordered 45 features; scaler.pkl; average all .cbm seed models',
        caution='seed_std is seed disagreement, not calibrated uncertainty'))
    save_csv(pd.DataFrame({'feature':FEATURES,'mean_importance':np.mean(importances,axis=0)}).sort_values('mean_importance',ascending=False),out/'feature_importance.csv')
    reference=test[['_row_id','date',base.TARGET]].copy();reference['prediction']=ref.prediction.to_numpy()
    comparisons=[];year_rows=[]
    for name,pred in [('XGBoost45_tuned_reference',reference),('CatBoost45_tuned',predicted)]:
        for subset,ids in tests.items():
            a=pred.loc[pred._row_id.isin(ids)];comparisons.append(dict(model=name,subset=subset,**base.metrics(a[base.TARGET],a.prediction)))
        for year,a in pred.groupby(pred.date.dt.year):year_rows.append(dict(model=name,year=year,**base.metrics(a[base.TARGET],a.prediction)))
    save_csv(pd.DataFrame(comparisons),out/'test_comparison.csv');save_csv(pd.DataFrame(year_rows),out/'test_by_year.csv')
    raw=pd.read_pickle(DATA_FILE).reset_index(drop=True)
    poof=train[['_row_id','date','grid_lat','grid_lon',base.TARGET]].copy();poof['prediction']=oof
    audit_events(raw,poof,out/'audit_training_oof','training_oof')
    audit_events(raw,predicted,out/'audit_test','selected_test')
    logging.info('\n%s',pd.DataFrame(comparisons).to_string(index=False));logging.info('Finished %s',out)
    lock.close()

if __name__=='__main__':main()
