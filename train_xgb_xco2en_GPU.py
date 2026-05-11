import pandas as pd
import numpy as np
import optuna
import logging
import os
import json
import joblib
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split

# ==========================================
# 0. 全局配置与路径初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SHP_xgb_training(A01-No lead and lag).log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SHP_optuna_xgb_study-A01.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/train_xgb_xco2en_SHP_best-A01.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01.pkl' 
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A01.pkl' 
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'   

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE.replace('sqlite:///', '')), exist_ok=True)
os.makedirs(os.path.dirname(PARAMS_JSON), exist_ok=True)

if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 数据加载与特征工程
# ==========================================
def load_and_preprocess(file_path):
    logger.info(f"📂 正在加载数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    
    df_clean['no2_trop_log'] = np.log(df_clean['no2_trop'])

    # 交叉验证时打乱顺序，时间排序不再是必须的，但保留特征工程
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)
    
    # 物理交叉特征
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)

    return df_clean

# ==========================================
# 2. 基于 SHAP 的两阶段特征筛选
# ==========================================
def perform_shap_feature_selection(X_train, y_train, feature_names, top_n=20):
    logger.info("🔍 阶段一：启动基线 XGBoost 模型进行全局 SHAP 特征重要性评估...")
    
    baseline_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, 
        n_jobs=-1, random_state=42, tree_method='hist',
        device='cuda'
    )
    baseline_model.fit(X_train, y_train)
    
    logger.info("🧠 计算 SHAP 值 (解释模型预测)...")
    explainer = shap.TreeExplainer(baseline_model)
    sample_X = X_train[:10000] if len(X_train) > 10000 else X_train
    shap_values = explainer.shap_values(sample_X)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': mean_abs_shap
    }).sort_values(by='SHAP_Importance', ascending=False)
    
    logger.info("-" * 25 + " SHAP 物理贡献度全排名 " + "-" * 25)
    for idx, row in shap_importance.iterrows():
        logger.info(f"  {row['Feature']:>22} : {row['SHAP_Importance']:.4f}")
        
    selected_features = shap_importance.head(top_n)['Feature'].tolist()
    logger.info(f"✨ 筛选出最具物理意义的 Top-{top_n} 特征: {selected_features}")
    
    with open(FEATURES_JSON, 'w', encoding='utf-8') as f:
        json.dump(selected_features, f, indent=4)
        
    return selected_features

# ==========================================
# 3. Optuna 深度优化 (使用 3-Fold 加速寻优)
# ==========================================
def optimize_xgb(X_pool, y_pool, n_trials=50):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42,
            'n_jobs': -1
        }

        # 寻优阶段用 3-Fold 即可，节省时间
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_rmses = []
        
        for train_index, val_index in kf.split(X_pool):
            X_tr_raw, X_va_raw = X_pool[train_index], X_pool[val_index]
            y_tr, y_va = y_pool.iloc[train_index].values, y_pool.iloc[val_index].values

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_va = scaler.transform(X_va_raw)
            
            model = xgb.XGBRegressor(**param)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=30, verbose=False)
            
            preds = model.predict(X_va)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            
        return np.mean(cv_rmses)

    logger.info("🚀 阶段二：开始 XGBoost Optuna 深度参数搜索...")
    study = optuna.create_study(
        direction='minimize', 
        storage=DB_FILE, 
        load_if_exists=True,
        study_name='xco2en_gpu_xgboost' 
    )
    study.optimize(objective, n_trials=n_trials) 
    return study.best_params

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    target = 'xco2_enhanced'
    
    initial_features = [
        'era5_blh', 'era5_d2m', 'era5_sp', 'era5_ssrd', 'era5_t2m', 'era5_tcwv', 
        'era5_u100', 'era5_v100', 'era5_u10', 'era5_v10','era5_wind_speed',
        'grid_lat', 'grid_lon', 'dem_mean', 
        'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'ndvi_t2m_cross', 'ssrd_t2m_cross', 'ntl_nox_cross',
        'meic_nox', 'ntl', 'ndvi','no2_trop_log',
    ]
    
    # 1. 准备全局数据并严格切分
    df = load_and_preprocess(file_path)
    X_full_raw = df[initial_features].values
    y_full = df[target]
    
    # 【修复 1】：最开头就切分出 20% 的“绝对盲测集”，完全冻结，不参与特征筛选和调参
    X_pool_raw, X_test_raw, y_pool, y_test = train_test_split(
        X_full_raw, y_full, test_size=0.2, random_state=42
    )
    
    # 2. 执行 SHAP 特征筛选 (仅使用训练池数据 X_pool)
    temp_scaler = StandardScaler()
    X_pool_scaled_for_shap = temp_scaler.fit_transform(X_pool_raw) # 【修复 2】：Scaler仅拟合训练池

    selected_feature_names = perform_shap_feature_selection(
        X_pool_scaled_for_shap, y_pool, initial_features, top_n=20
    )
    selected_indices = [initial_features.index(f) for f in selected_feature_names]
    
    # 获取精简特征后的数据
    X_pool_selected_raw = X_pool_raw[:, selected_indices]
    X_test_selected_raw = X_test_raw[:, selected_indices] # 测试集也要同步抽取这些特征

    # 3. 执行参数寻优 (仅使用训练池数据 X_pool)
    best_params = optimize_xgb(X_pool_selected_raw, y_pool, n_trials=500)
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ XGB 最优参数已保存至: {PARAMS_JSON}")

    # ==========================================
    # 4. 终极盲测评估 (使用从没见过测试集评估真实性能)
    # ==========================================
    logger.info("🏁 阶段三：执行终极盲测集评估，并对比训练集性能...")
    
    # 严格的标准化逻辑：fit训练池，transform测试集
    eval_scaler = StandardScaler()
    X_pool_selected_scaled = eval_scaler.fit_transform(X_pool_selected_raw)
    X_test_selected_scaled = eval_scaler.transform(X_test_selected_raw)

    eval_model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=42)
    eval_model.fit(
        X_pool_selected_scaled, y_pool,
        verbose=False
    )
    
    # 👇 [修改点]：同时对训练集(Pool)和测试集(Test)进行预测
    train_preds = eval_model.predict(X_pool_selected_scaled)
    test_preds = eval_model.predict(X_test_selected_scaled)

    # 👇 [修改点]：计算训练集的指标 (与 RF 对齐)
    train_r2 = r2_score(y_pool, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_pool, train_preds))

    # 计算测试集的指标
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_bias = np.mean(test_preds - y_test.values)

    # 👇 [修改点]：输出与 RF 完全一致的评估报告格式
    logger.info("="*30 + " XGBOOST FINAL REPORT " + "="*30)
    logger.info(f"Train R²  : {train_r2:.4f}")
    logger.info(f"Train RMSE: {train_rmse:.4f} ppm")
    logger.info("-" * 25)
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    logger.info("="*82)