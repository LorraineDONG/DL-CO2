import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    df_clean['era5_wind_speed_100'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    df_clean['ventilation_coef'] = df_clean['era5_blh'] * df_clean['era5_wind_speed_100']
    
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
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
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
    
    DATA_PATH = '/home/whdong/dl/data/TABLE-NO2_0.25deg_Filling_TrainSet_2018-2023.pkl'
    features = [
        'era5_t2m', 'era5_blh', 'era5_ssrd', 'era5_u100', 'era5_v100', 'era5_tcwv',
        'month_sin', 'month_cos', 'doy_sin', 'doy_cos','ventilation_coef',
        'grid_lon', 'grid_lat', 'dem_mean', 'meic_nox', 'ntl'
    ]
    target_raw = 'no2_trop'
    a_offset = -130  

    # 1. 数据加载与预处理
    df = load_and_preprocess(DATA_PATH, target_raw)
    X_all_raw = df[features].values
    y_log = df['target_log'].values

    # 【修复 1】：立即切分 20% 绝对盲测集，确保调参阶段不可见
    X_pool_raw, X_test_raw, y_pool_log, y_test_log = train_test_split(
        X_all_raw, y_log, test_size=0.2, random_state=42
    )

    # 2. 参数寻优 (仅使用 X_pool_raw)
    # 函数内部已包含标准化逻辑
    best_params = optimize_xgb(X_pool_raw, y_pool_log, n_trials=50)
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    
    with open(PARAMS_JSON, 'w') as f:
        json.dump(best_params, f, indent=4)

    # 3. 终极盲测评估 (对比训练集与盲测集)
    logger.info("🏁 执行终极评估：对比训练池与盲测集表现...")
    
    # 严格标准化：fit 训练池，transform 盲测集
    eval_scaler = StandardScaler()
    X_pool_scaled = eval_scaler.fit_transform(X_pool_raw)
    X_test_scaled = eval_scaler.transform(X_test_raw)

    # 训练评估模型
    eval_model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=42)
    eval_model.fit(X_pool_scaled, y_pool_log, verbose=False)
    
    # 预测 (Log 域)
    train_log_preds = eval_model.predict(X_pool_scaled)
    test_log_preds = eval_model.predict(X_test_scaled)

    # 反向变换回物理单位 (恢复真实量纲)
    # 使用 clip 防止指数爆炸
    train_phys_preds = (10 ** np.clip(train_log_preds, -5, 5)) + a_offset
    train_phys_true = (10 ** y_pool_log) + a_offset
    test_phys_preds = (10 ** np.clip(test_log_preds, -5, 5)) + a_offset
    test_phys_true = (10 ** y_test_log) + a_offset

    # 计算物理量纲指标
    train_r2 = r2_score(train_phys_true, train_phys_preds)
    train_rmse = np.sqrt(mean_squared_error(train_phys_true, train_phys_preds))
    
    test_r2 = r2_score(test_phys_true, test_phys_preds)
    test_rmse = np.sqrt(mean_squared_error(test_phys_true, test_phys_preds))
    test_mae = mean_absolute_error(test_phys_true, test_phys_preds)
    test_bias = np.mean(test_phys_preds - test_phys_true)

    logger.info("="*35 + " NO2 XGBOOST 评估报告 " + "="*35)
    logger.info("【训练池表现 (Train Pool)】")
    logger.info(f"R²         : {train_r2:.4f}")
    logger.info(f"RMSE (μmol/m2): {train_rmse:.4f}")
    logger.info("-" * 40)
    logger.info("【盲测集表现 (Blind Test)】")
    logger.info(f"R²         : {test_r2:.4f}")
    logger.info(f"RMSE (μmol/m2): {test_rmse:.4f}")
    logger.info(f"MAE  (μmol/m2): {test_mae:.4f}")
    logger.info(f"BIAS (μmol/m2): {test_bias:.4f}")
    logger.info("="*93)

    # ==========================================
    # 4. 训练并保存生产模型 (使用全量数据)
    # ==========================================
    logger.info("💾 正在固化最终生产模型...")
    
    # 全局生产标准化器
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all_raw)
    
    # 【修复 2】：先切分 ES 验证集，再独立进行标准化 transform
    X_prod_tr_raw, X_prod_val_raw, y_prod_tr, y_prod_val = train_test_split(
        X_all_raw, y_log, test_size=0.05, random_state=42
    )

    es_scaler = StandardScaler()
    X_prod_tr = es_scaler.fit_transform(X_prod_tr_raw)
    X_prod_val = es_scaler.transform(X_prod_val_raw)

    final_model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=42)
    final_model.fit(
        X_prod_tr, y_prod_tr,
        eval_set=[(X_prod_val, y_prod_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    joblib.dump({'model': final_model, 'features': features}, MODEL_SAVE_PATH)
    joblib.dump(final_scaler, SCALER_SAVE_PATH)
    logger.info(f"✅ 模型与全局 Scaler 已保存至: {MODEL_SAVE_PATH}")