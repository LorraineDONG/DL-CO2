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
LOG_FILE = '/home/whdong/dl/logfile/NO2_Filling_xgb_final-1.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/NO2_Optuna_xgb-1.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/xgb_NO2_best_params-1.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/NO2_Filling_xgb_model-1.pkl' 
SCALER_SAVE_PATH = '/home/whdong/dl/models/NO2_Filling_scaler-1.pkl' 

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

    # 1. 准备数据
    df = load_and_preprocess(DATA_PATH, target_raw)
    
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
    
    # 👇 [修改点] 新增列表存储每一折的训练集评价指标
    train_r2_list = []
    train_rmse_list = []
    train_mae_list = []
    
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
        
        # 对验证集(测试集)进行预测
        oof_log_preds[test_idx] = fold_model.predict(X_te)
        
        # 👇 [修改点] 对当前折的训练集进行预测，以评估训练集效果
        X_fold_train_scaled = fold_scaler.transform(X_fold_train_raw)
        train_log_preds = fold_model.predict(X_fold_train_scaled)
        
        # [修复点] 添加截断限制，防止指数爆炸。这里根据真实数据y_log的上下界进行截断
        max_log = np.max(y_log) + 0.5   # 允许一定的合理外推
        min_log = np.min(y_log) - 0.5
        train_log_preds = np.clip(train_log_preds, min_log, max_log)

        # 还原物理单位以计算指标
        train_phys_preds = (10 ** train_log_preds) + a_offset
        train_phys_true = (10 ** y_fold_train) + a_offset
        
        train_r2_list.append(r2_score(train_phys_true, train_phys_preds))
        train_rmse_list.append(np.sqrt(mean_squared_error(train_phys_true, train_phys_preds)))
        train_mae_list.append(mean_absolute_error(train_phys_true, train_phys_preds))
        
    # 4. 反向变换回物理单位进行指标计算
    # 👇 [修改点] 分别计算并输出训练集和测试集的表现
    y_phys_true = df[target_raw].values
    # [修复点] 同样对Out-of-Fold测试集预测进行截断
    oof_log_preds = np.clip(oof_log_preds, min_log, max_log)
    y_phys_pred = (10 ** oof_log_preds) + a_offset
    
    # 计算测试集（OOF）整体指标
    test_r2 = r2_score(y_phys_true, y_phys_pred)
    test_rmse = np.sqrt(mean_squared_error(y_phys_true, y_phys_pred))
    test_mae = mean_absolute_error(y_phys_true, y_phys_pred)

    # 计算训练集的平均指标（10折平均）
    train_r2_mean = np.mean(train_r2_list)
    train_rmse_mean = np.mean(train_rmse_list)
    train_mae_mean = np.mean(train_mae_list)

    logger.info("="*35 + " 评估报告 (原始物理单位) " + "="*35)
    logger.info("【训练集表现 (Train Set - 10-Fold 均值)】")
    logger.info(f"决定系数 R²   : {train_r2_mean:.4f}")
    logger.info(f"RMSE (μmol/m2): {train_rmse_mean:.4f}")
    logger.info(f"MAE  (μmol/m2): {train_mae_mean:.4f}")
    logger.info("-" * 93)
    logger.info("【测试集表现 (Test Set - Out-of-Fold 整体)】")
    logger.info(f"决定系数 R²   : {test_r2:.4f}")
    logger.info(f"RMSE (μmol/m2): {test_rmse:.4f}")
    logger.info(f"MAE  (μmol/m2): {test_mae:.4f}")
    logger.info("="*93)

    # ==========================================
    # 5. 训练并保存生产模型
    # ==========================================
    logger.info("💾 正在固化最终生产模型...")
    
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all_raw)
    
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
    
    joblib.dump({
        'model': final_model,
        'features': features
    }, MODEL_SAVE_PATH)
    
    joblib.dump(final_scaler, SCALER_SAVE_PATH)
    logger.info(f"✅ 模型与 Scaler 已保存至: {MODEL_SAVE_PATH} & {SCALER_SAVE_PATH}")
    
    import gc
    del final_model
    del fold_model
    gc.collect()