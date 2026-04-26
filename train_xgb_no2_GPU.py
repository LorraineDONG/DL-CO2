import pandas as pd
import numpy as np
import optuna
import logging
import os
import json
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split

# ==========================================
# 0. 全局配置与路径初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/NO2_Filling_xgb_final.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/NO2_Optuna_xgb.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/xgb_NO2_best_params.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/NO2_Filling_xgb_model.pkl' 
SCALER_SAVE_PATH = '/home/whdong/dl/models/NO2_Filling_scaler.pkl' 

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE.replace('sqlite:///', '')), exist_ok=True)
os.makedirs(os.path.dirname(PARAMS_JSON), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 数据预处理（包含 Log 变换）
# ==========================================
def load_and_preprocess(file_path, target_col):
    logger.info(f"📂 正在加载数据: {file_path}...")
    df = pd.read_pickle(file_path)
    
    # 剔除特征或标签含空值的样本
    df_clean = df.dropna().copy()

    # 时间特征工程：捕捉季节性和周期性信号
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)

    # 目标变量 Log 变换：平衡数据分布并改善极端值预测精度
    # 公式: y' = log10(NTVCD - a), a = -130
    a = -130
    df_clean['target_log'] = np.log10(df_clean[target_col] - a)

    return df_clean

# ==========================================
# 2. Optuna 参数寻优（使用 3-Fold 避免泄露并兼顾速度）
# ==========================================
def optimize_xgb(X_pool, y_pool, n_trials=30):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-4, 0.5, log=True),
            'tree_method': 'hist',
            'n_jobs': -1,
            'device': 'cuda', 
            'random_state': 42
        }
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_rmses = []
        for train_index, val_index in kf.split(X_pool):
            X_tr_raw, X_va_raw = X_pool[train_index], X_pool[val_index]
            y_tr, y_va = y_pool[train_index], y_pool[val_index]

            # 在这里进行折叠内的缩放，避免泄漏
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_va = scaler.transform(X_va_raw)
            
            model = xgb.XGBRegressor(**param)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=30, verbose=False)
            preds = model.predict(X_va)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            
        return np.mean(cv_rmses)

    logger.info("🚀 启动 Optuna 参数搜索...")
    study = optuna.create_study(direction='minimize', storage=DB_FILE, load_if_exists=True, study_name='no2_optuna')
    study.optimize(objective, n_trials=n_trials) 
    return study.best_params

# ==========================================
# 3. 主程序：训练与评估
# ==========================================
if __name__ == "__main__":
    
    DATA_PATH = '/home/whdong/dl/NO2_0.25deg_Filling_TrainSet_2018-2025.pkl'
    features = [
        'era5_t2m', 'era5_blh', 'era5_ssrd', 'era5_u100', 'era5_v100', 'era5_tcwv',
        'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'grid_lon', 'grid_lat', 'dem_mean', 'meic_nox', 'ntl'
    ]
    target_raw = 'no2_trop'
    a_offset = -130  

    # 1. 准备数据
    df = load_and_preprocess(DATA_PATH, target_raw)
    
    # 提取特征矩阵和对数目标，使用原始数值避免提前泄露
    X_all_raw = df[features].values
    y_log = df['target_log'].values

    # 2. 参数寻优
    best_params = optimize_xgb(X_all_raw, y_log, n_trials=50)
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    
    with open(PARAMS_JSON, 'w') as f:
        json.dump(best_params, f, indent=4)

    # 3. 10-Fold 交叉验证终极评估
    logger.info("🏁 执行 10-Fold 交叉验证评估...")
    kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)
    oof_log_preds = np.zeros(len(y_log))
    
    for fold, (train_idx, test_idx) in enumerate(kf_10.split(X_all_raw)):
        X_fold_train_raw = X_all_raw[train_idx]
        y_fold_train = y_log[train_idx]
        X_fold_test_raw = X_all_raw[test_idx]
        
        # 内部分割验证集供 early_stopping 使用
        X_inner_tr_raw, X_inner_val_raw, y_inner_tr, y_inner_val = train_test_split(
            X_fold_train_raw, y_fold_train, test_size=0.1, random_state=42
        )
        
        # 折叠内部进行归一化，彻底杜绝数据泄露
        fold_scaler = StandardScaler()
        X_inner_tr = fold_scaler.fit_transform(X_inner_tr_raw)
        X_inner_val = fold_scaler.transform(X_inner_val_raw)
        X_te = fold_scaler.transform(X_fold_test_raw)
        
        fold_model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=42)
        fold_model.fit(
            X_inner_tr, y_inner_tr,
            eval_set=[(X_inner_val, y_inner_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # 用跑出来的最佳迭代次数进行预测
        # 说明：predict方法默认使用基于early_stopping得到的最优树进行预测
        oof_log_preds[test_idx] = fold_model.predict(X_te)
        
    # 4. 反向变换回物理单位进行指标计算
    y_phys_true = df[target_raw].values
    y_phys_pred = (10 ** oof_log_preds) + a_offset

    final_r2 = r2_score(y_phys_true, y_phys_pred)
    final_rmse = np.sqrt(mean_squared_error(y_phys_true, y_phys_pred))
    final_mae = mean_absolute_error(y_phys_true, y_phys_pred)

    logger.info("="*30 + " 评估报告 (原始物理单位) " + "="*30)
    logger.info(f"决定系数 R²   : {final_r2:.4f}")
    logger.info(f"RMSE (μmol/m2): {final_rmse:.4f}")
    logger.info(f"MAE  (μmol/m2): {final_mae:.4f}")
    logger.info("="*85)

    # ==========================================
    # 5. 训练并保存生产模型
    # ==========================================
    logger.info("💾 正在固化最终生产模型...")
    
    # 生产模型使用100%数据训练，需要一个全局最终的 Scaler
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all_raw)
    
    # 拆分一小部分用于监控以防止过拟合
    X_final_tr, X_final_val, y_final_tr, y_final_val = train_test_split(
        X_all_scaled, y_log, test_size=0.05, random_state=42
    )

    final_model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=42)
    final_model.fit(
        X_final_tr, y_final_tr,
        eval_set=[(X_final_val, y_final_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 将模型与固定的特征列表绑定存储
    joblib.dump({
        'model': final_model,
        'features': features
    }, MODEL_SAVE_PATH)
    
    # 保存这个对应全部15个特征的 StandardScaler 供推理预测使用
    joblib.dump(final_scaler, SCALER_SAVE_PATH)
    logger.info(f"✅ 模型与 Scaler 已保存至: {MODEL_SAVE_PATH} & {SCALER_SAVE_PATH}")