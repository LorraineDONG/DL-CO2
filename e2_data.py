"""A01 sample-selection experiment. Edit SETTINGS; run directly, no CLI.

E1: original whole-table dropna baseline.
E2: finite A01 candidate inputs, excluding is_model_filled=True.
E3: E2 training + finite model-filled rows (exploratory).
All models share fixed A01 features/hyperparameters/seeds/test IDs.
Historical test is a reference, already used in the original development.
This random split measures interpolation, not spatial/time extrapolation.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import logging
import platform
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ===================== SETTINGS =====================
DATA_FILE = Path('/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl')
OUTPUT_ROOT = Path('/home/whdong/dl/a01_sample_selection_experiments')
# On Windows replace the two paths above, for example:
# DATA_FILE = Path(r'D:\2026DATASET\DL-CO2\SHP-XCO2en\TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl')
# OUTPUT_ROOT = Path(r'D:\xco2_experiments')
PARAMS_FILE = Path(__file__).resolve().parent / 'train_xgb_xco2en_SHP_best-A01.json'
DEVICE = 'cuda'  # Use 'cpu' when no compatible GPU is available.
THREADS = 8
SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]
TEST_FRACTION = 0.2
OLD_SPLIT_SEED = 42
ADDED_SPLIT_SEED = 20261009
RUN_E1_BASELINE = True
RUN_E3_MODEL_FILLED = True  # Exploratory until ND49 units/time/column definition validated.
# ====================================================

TARGET = 'xco2_enhanced'
ERA_NAMES = ['blh','d2m','sp','ssrd','t2m','tcwv','u100','u10','v100','v10']
CANDIDATES = [f'era5_{v}{s}' for v in ERA_NAMES for s in ['', '_lag1', '_lead1']] + [
    'no2_trop_log','no2_trop_mean','ndvi_t2m_cross','ssrd_t2m_cross','ntl_nox_cross',
    'meic_nox','ndvi','ntl','dem_mean','doy_sin','doy_cos','month_sin','month_cos','grid_lon','grid_lat']
# Exact feature order from the uploaded original A01 training log, Top-20.
FIXED_FEATURES = ['doy_sin','no2_trop_mean','no2_trop_log','month_sin','grid_lat','doy_cos',
    'era5_t2m_lead1','ssrd_t2m_cross','era5_ssrd_lead1','era5_t2m','grid_lon','era5_ssrd_lag1',
    'ndvi_t2m_cross','era5_blh','era5_blh_lead1','era5_sp','dem_mean','era5_d2m_lag1','ndvi','era5_blh_lag1']


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,obj):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.generic):x=x.item()
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    Path(path).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')


def load_and_preprocess(path):
    raw=pd.read_pickle(path)
    if not isinstance(raw,pd.DataFrame) or not len(raw):raise ValueError('Input must be a nonempty DataFrame')
    if raw.columns.duplicated().any():raise ValueError('Duplicate column names')
    if 'is_model_filled' not in raw:raise ValueError('is_model_filled is required for E2/E3; do not guess provenance')
    raw=raw.reset_index(drop=True)
    df=raw.copy()
    df['_row_id']=np.arange(len(df),dtype=np.int64)
    # Do not silently call NaN False: source provenance must be known.
    flag=raw.is_model_filled.map(lambda x:str(x).strip().lower())
    parsed=flag.map({'true':True,'false':False,'1':True,'0':False,'1.0':True,'0.0':False})
    if parsed.isna().any():raise ValueError('Unknown/missing is_model_filled values')
    df['_model_filled']=parsed.astype(bool)
    df['date']=pd.to_datetime(df['date'],errors='coerce')
    for c in set(CANDIDATES+['no2_trop',TARGET,'era5_ssrd','era5_t2m','ntl','meic_nox','ndvi']):
        if c in df:df[c]=pd.to_numeric(df[c],errors='coerce')
    with np.errstate(all='ignore'):
        df['no2_trop_log']=np.log(df.no2_trop)
        df['ndvi_t2m_cross']=df.ndvi*df.era5_t2m
        df['ssrd_t2m_cross']=df.era5_ssrd*df.era5_t2m
        df['ntl_nox_cross']=df.ntl*df.meic_nox
    month=df.date.dt.month;doy=df.date.dt.dayofyear
    df['month_sin']=np.sin(2*np.pi*month/12);df['month_cos']=np.cos(2*np.pi*month/12)
    df['doy_sin']=np.sin(2*np.pi*doy/365.25);df['doy_cos']=np.cos(2*np.pi*doy/365.25)
    absent=set(CANDIDATES+[TARGET])-set(df)
    if absent:raise ValueError(f'Missing A01 required columns: {sorted(absent)}')
    required_ok=np.isfinite(df[CANDIDATES+[TARGET]].to_numpy(float)).all(axis=1)&df.date.notna().to_numpy()
    required_ok&=df.grid_lat.ge(-90).to_numpy()&df.grid_lat.lt(90).to_numpy()&df.grid_lon.ge(-180).to_numpy()&df.grid_lon.lt(180).to_numpy()
    # Whole-table NA status must be measured BEFORE adding diagnostic columns.
    original_ok=~raw.isna().any(axis=1).to_numpy()
    if np.any(original_ok&~required_ok):
        raise ValueError('Original dropna includes invalid engineered inputs. Investigate before claiming an exact E1 reproduction.')
    eligible=df.loc[required_ok].copy()
    if eligible.duplicated(['date','grid_lat','grid_lon']).any():raise ValueError('Duplicate eligible date/grid labels; cannot use current experiment design')
    old=df.loc[original_ok&required_ok].copy()
    if old._model_filled.any():raise ValueError('E1 contains model-filled rows; proposed pure-satellite baseline assumptions no longer hold')
    satellite=eligible.loc[~eligible._model_filled].copy()
    model=eligible.loc[eligible._model_filled].copy()
    return df,old,satellite,model


def make_plan(df,old,satellite,model):
    if len(old)<10:raise ValueError('Too few original complete rows for a comparison')
    # Original train_test_split applied to the same ordered old pool reproduces
    # historical row membership, provided the source PKL/order is unchanged.
    old_train,old_test=train_test_split(old._row_id.to_numpy(),test_size=TEST_FRACTION,random_state=OLD_SPLIT_SEED)
    added=satellite.loc[~satellite._row_id.isin(old._row_id),'_row_id'].to_numpy()
    if len(added)>=5:
        added_train,added_test=train_test_split(added,test_size=TEST_FRACTION,random_state=ADDED_SPLIT_SEED)
    else:raise ValueError('Too few extra satellite rows to reserve a new common test subset')
    satellite_train=np.concatenate([old_train,added_train])
    combined_test=np.concatenate([old_test,added_test])
    plans={'E1_original_dropna':old_train,'E2_required_satellite':satellite_train}
    if RUN_E3_MODEL_FILLED:
        if not len(model):raise ValueError('No eligible model-filled rows; E3 would be identical to E2')
        plans['E3_required_with_model']=np.concatenate([satellite_train,model._row_id.to_numpy()])
    tests={'historical_reference':old_test,'added_satellite_holdout':added_test,'combined_satellite_holdout':combined_test}
    for name,ids in plans.items():
        if len(np.unique(ids))!=len(ids):raise AssertionError('Repeated training row IDs')
        if np.intersect1d(ids,combined_test).size:raise AssertionError(f'Test/train overlap in {name}')
    if np.intersect1d(old_test,added_test).size:raise AssertionError('Overlapping test subsets')
    membership=pd.DataFrame({'row_id':df._row_id,'date':df.date,'grid_lat':df.grid_lat,'grid_lon':df.grid_lon,
        'model_filled':df._model_filled,'role':'excluded_required_invalid'})
    for role,ids in [('old_training',old_train),('historical_reference',old_test),('added_satellite_training',added_train),('added_satellite_holdout',added_test),('model_filled_training',model._row_id.to_numpy())]:
        membership.loc[ids,'role']=role
    return plans,tests,membership


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite target or prediction')
    slope=float(np.polyfit(y,p,1)[0]) if len(y)>1 and np.std(y)>0 else np.nan
    return dict(n=len(y),r2=float(r2_score(y,p)) if len(y)>1 and np.std(y)>0 else np.nan,
        rmse=float(np.sqrt(mean_squared_error(y,p))),mae=float(mean_absolute_error(y,p)),
        bias=float(np.mean(p-y)),slope_pred_on_true=slope)


def run_models(df,plans,tests,params,out,logger):
    combined_ids=tests['combined_satellite_holdout'];test=df.loc[combined_ids]
    xtest=test[FIXED_FEATURES].to_numpy(float);ytest=test[TARGET].to_numpy(float)
    test_roles=np.where(np.isin(combined_ids,tests['historical_reference']),'historical_reference','added_satellite_holdout')
    ensemble_rows=[];seed_rows=[];by_year=[]
    for name,train_ids in plans.items():
        if name=='E1_original_dropna' and not RUN_E1_BASELINE:continue
        folder=out/name;folder.mkdir()
        train=df.loc[train_ids];xtrain=train[FIXED_FEATURES].to_numpy(float);ytrain=train[TARGET].to_numpy(float)
        scaler=StandardScaler();xtrain=scaler.fit_transform(xtrain);xte=scaler.transform(xtest)
        joblib.dump(scaler,folder/'scaler.pkl')
        write_json(folder/'selected_features.json',FIXED_FEATURES)
        prediction_matrix=[];train_prediction_matrix=[]
        for seed in SEEDS:
            # Fixed full 1700 rounds; no early stopping/test evaluation during fit.
            kwargs=dict(params);kwargs.update(tree_method='hist',device=DEVICE,n_jobs=THREADS,random_state=seed)
            for unwanted in ['early_stopping_rounds','callbacks']:kwargs.pop(unwanted,None)
            model=xgb.XGBRegressor(**kwargs)
            logger.info('%s | train=%d | seed=%d',name,len(train),seed)
            model.fit(xtrain,ytrain,verbose=False)
            pred=model.predict(xte);train_pred=model.predict(xtrain)
            prediction_matrix.append(pred);train_prediction_matrix.append(train_pred)
            joblib.dump(model,folder/f'model_seed{seed}.pkl')
            model.save_model(str(folder/f'model_seed{seed}.json'))
            seed_rows.append(dict(experiment=name,seed=seed,subset='training',**metrics(ytrain,train_pred)))
            for subset,ids in tests.items():
                mask=np.isin(combined_ids,ids)
                seed_rows.append(dict(experiment=name,seed=seed,subset=subset,**metrics(ytest[mask],pred[mask])))
        pred=np.mean(prediction_matrix,axis=0);train_pred=np.mean(train_prediction_matrix,axis=0)
        ensemble_rows.append(dict(experiment=name,subset='training',**metrics(ytrain,train_pred)))
        for subset,ids in tests.items():
            mask=np.isin(combined_ids,ids);score=metrics(ytest[mask],pred[mask]);ensemble_rows.append(dict(experiment=name,subset=subset,**score))
            logger.info('%s | %s | ensemble R2=%.4f RMSE=%.4f n=%d',name,subset,score['r2'],score['rmse'],score['n'])
        predicted=test[['_row_id','date','grid_lat','grid_lon',TARGET]].copy();predicted['test_subset']=test_roles
        predicted['prediction']=pred;predicted['seed_std']=np.std(prediction_matrix,axis=0)
        for i,seed in enumerate(SEEDS):predicted[f'pred_seed{seed}']=prediction_matrix[i]
        predicted.to_csv(folder/'common_test_predictions.csv',index=False,encoding='utf-8-sig')
        for (role,year),sub in predicted.groupby(['test_subset',predicted.date.dt.year]):
            by_year.append(dict(experiment=name,subset=role,year=int(year),**metrics(sub[TARGET],sub.prediction)))
        write_json(folder/'manifest.json',dict(experiment=name,training_rows=len(train),training_model_rows=int(train._model_filled.sum()),
            selected_features=FIXED_FEATURES,parameters=params,device=DEVICE,seeds=SEEDS,
            prediction='transform ordered features with this scaler, then average all seed model predictions',
            caveat='seed_std is model seed disagreement, not calibrated error or prediction interval'))
    pd.DataFrame(ensemble_rows).to_csv(out/'ensemble_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(seed_rows).to_csv(out/'seed_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(by_year).to_csv(out/'test_metrics_by_year.csv',index=False,encoding='utf-8-sig')
    logger.info('\n%s',pd.DataFrame(ensemble_rows).to_string(index=False))
    return ensemble_rows


def main():
    if not DATA_FILE.is_file():raise FileNotFoundError(DATA_FILE)
    if not PARAMS_FILE.is_file():raise FileNotFoundError(PARAMS_FILE)
    if not SEEDS or len(set(SEEDS))!=len(SEEDS):raise ValueError('SEEDS must contain unique values')
    if not 0<TEST_FRACTION<.5:raise ValueError('Invalid test fraction')
    params=json.loads(PARAMS_FILE.read_text(encoding='utf-8'))
    out=OUTPUT_ROOT/('controlled_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True,exist_ok=False)
    logger=logging.getLogger('A01_controlled');logger.handlers.clear();logger.setLevel(logging.INFO)
    fmt=logging.Formatter('%(asctime)s | %(message)s')
    for h in [logging.StreamHandler(),logging.FileHandler(out/'training.log',encoding='utf-8')]:h.setFormatter(fmt);logger.addHandler(h)
    df,old,satellite,model=load_and_preprocess(DATA_FILE)
    logger.info('Candidate pools: E1=%d E2=%d E3=%d; eligible model=%d',len(old),len(satellite),len(satellite)+len(model),len(model))
    plans,tests,membership=make_plan(df,old,satellite,model)
    membership.to_csv(out/'split_membership.csv',index=False,encoding='utf-8-sig')
    year_table=membership.groupby([membership.date.dt.year.rename('year'),'role']).size().rename('rows').reset_index()
    year_table.to_csv(out/'split_counts_by_year.csv',index=False,encoding='utf-8-sig')
    write_json(out/'experiment_plan.json',dict(data_file=str(DATA_FILE),data_sha256=sha256(DATA_FILE),script_sha256=sha256(__file__),
        rows=len(df),pool_counts=dict(E1=len(old),E2=len(satellite),E3=len(satellite)+len(model),model_eligible=len(model)),
        training_counts={k:len(v) for k,v in plans.items()},test_counts={k:len(v) for k,v in tests.items()},
        features=FIXED_FEATURES,eligibility_features=CANDIDATES,params=params,seeds=SEEDS,
        old_split_seed=OLD_SPLIT_SEED,added_split_seed=ADDED_SPLIT_SEED,device=DEVICE,
        python=platform.python_version(),pandas=pd.__version__,xgboost=xgb.__version__,
        notes=['All tests use satellite-flagged labels. Model-filled eligible rows enter E3 training only.',
          'Historical test and old features/hyperparameters have already been used in model development: reference comparison, not a fresh blind test.',
          'Added holdout was not used in this script feature selection or tuning. This is a random interpolation split.',
          'E3 is exploratory until source ND49 definition/units/time are validated. Do not choose the best seed using test results.',
          'No independent uncertainty calibration in this controlled experiment.']))
    run_models(df,plans,tests,params,out,logger)
    logger.info('Saved results: %s',out)


if __name__=='__main__':main()
