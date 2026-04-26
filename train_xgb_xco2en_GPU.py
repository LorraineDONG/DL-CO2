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
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SHP_xgb_10fold.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SHP_optuna_xgb_10fold.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/train_xgb_xco2en_SHP_10fold_best.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_10fold_model.pkl' 
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_10fold_scaler.pkl' 
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features_10fold.json'   

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
    
    # NO2 专属特征工程
    df_clean['no2_trop_log'] = np.log1p(np.maximum(df_clean['no2_trop'], 0))

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
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
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
    file_path = '/home/whdong/dl/xco2_sif_no2_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    target = 'xco2_enhanced'
    
    initial_features = [
        'era5_blh', 'era5_d2m', 'era5_sp', 'era5_ssrd', 'era5_t2m', 'era5_tcwv', 
        'era5_u100', 'era5_v100', 'era5_u10', 'era5_v10', 'era5_wind_speed',
        'grid_lat', 'grid_lon', 'dem_mean', 'dem_std', 
        'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'ndvi_t2m_cross', 'ssrd_t2m_cross', 'ntl_nox_cross',
        'meic_nox', 'ntl', 'ndvi', 'ndvi_std', 'sif_740', 'sif_variance',
        'no2_amf_trop', 'no2_trop', 'no2_variance', 'no2_trop_log'
    ]
    
    # 1. 准备全局数据 (不再切分测试集/训练集)
    df = load_and_preprocess(file_path)

    X_all_raw = df[initial_features].values
    y_all = df[target]
    
    # 2. 执行 SHAP 特征筛选 
    temp_scaler = StandardScaler()
    X_all_scaled_for_shap = temp_scaler.fit_transform(X_all_raw)

    selected_feature_names = perform_shap_feature_selection(
        X_all_scaled_for_shap, y_all, initial_features, top_n=20
    )
    selected_indices = [initial_features.index(f) for f in selected_feature_names]
    
    # 获取精简后的未缩放原始数据，进入严格流程
    X_all_selected_raw = X_all_raw[:, selected_indices]

    # 3. 执行参数寻优 (传入未缩放的数据，函数内部有缩放)
    best_params = optimize_xgb(X_all_selected_raw, y_all, n_trials=50)
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ XGB 最优参数已保存至: {PARAMS_JSON}")

    # ==========================================
    # 4. 核心修改：10-Fold 交叉验证终极评估
    # ==========================================
    logger.info("🏁 阶段三：执行严谨的 10-Fold 交叉验证评估...")
    
    kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y_all)) 
    fold_rmses = []
    fold_r2s = []
    
    for fold, (train_idx, test_idx) in enumerate(kf_10.split(X_all_selected_raw)):
        # 获取原始折数据
        X_fold_train_raw, y_fold_train = X_all_selected_raw[train_idx], y_all.iloc[train_idx]
        X_fold_test_raw, y_fold_test = X_all_selected_raw[test_idx], y_all.iloc[test_idx]
        
        # 【修复4】：使用 train_test_split 避免时序切片偏差
        X_inner_tr_raw, X_inner_val_raw, y_inner_tr, y_inner_val = train_test_split(
            X_fold_train_raw, y_fold_train, test_size=0.1, random_state=42
        )
        
        # 【核心修复】：在当前折闭环内进行严格标准化
        fold_scaler = StandardScaler()
        X_inner_tr = fold_scaler.fit_transform(X_inner_tr_raw)
        X_inner_val = fold_scaler.transform(X_inner_val_raw)
        X_fold_test = fold_scaler.transform(X_fold_test_raw)
        
        fold_model = xgb.XGBRegressor(**best_params,n_jobs=-1, random_state=42)
        
        # 训练：监控内部验证集
        fold_model.fit(
            X_inner_tr, y_inner_tr,
            eval_set=[(X_inner_val, y_inner_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # 预测：对未见过的测试折进行预测 (Out-of-Fold)
        preds = fold_model.predict(X_fold_test)
        oof_predictions[test_idx] = preds
        
        cur_rmse = np.sqrt(mean_squared_error(y_fold_test, preds))
        cur_r2 = r2_score(y_fold_test, preds)
        fold_rmses.append(cur_rmse)
        fold_r2s.append(cur_r2)
        
        logger.info(f"   Fold {fold+1:>2}/10 | Best Iter: {fold_model.best_iteration:>4} | R²: {cur_r2:.4f} | RMSE: {cur_rmse:.4f}")

    # 计算全局 OOF 指标 (这是最真实的泛化性能)
    final_r2 = r2_score(y_all, oof_predictions)
    final_rmse = np.sqrt(mean_squared_error(y_all, oof_predictions))
    final_mae = mean_absolute_error(y_all, oof_predictions)
    final_bias = np.mean(oof_predictions - y_all.values)

    logger.info("="*30 + " 10-FOLD CV FINAL REPORT " + "="*30)
    logger.info(f"Average Fold R²   : {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")
    logger.info(f"Global OOF R²     : {final_r2:.4f}")
    logger.info(f"Global OOF RMSE   : {final_rmse:.4f} ppm")
    logger.info(f"Global OOF MAE    : {final_mae:.4f} ppm")
    logger.info(f"Global OOF BIAS   : {final_bias:.4f} ppm")
    logger.info("="*85)

    # ==========================================
    # 5. 训练最终的生产模型 (使用全部数据)
    # ==========================================
    logger.info("💾 阶段四：使用 100% 数据训练最终生产模型并固化...")
    
    # 【修复5】：为最终模型创建并保存一个拟合了 100% 数据的 Scaler
    final_scaler = StandardScaler()
    X_all_selected_scaled = final_scaler.fit_transform(X_all_selected_raw)
    
    # 使用 train_test_split
    X_final_tr, X_final_val, y_final_tr, y_final_val = train_test_split(
        X_all_selected_scaled, y_all, test_size=0.05, random_state=42
    )
    
    production_model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=42)
    production_model.fit(
        X_final_tr, y_final_tr,
        eval_set=[(X_final_val, y_final_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 保存模型与标准化器
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump({'model': production_model, 'feature_indices': selected_indices}, MODEL_SAVE_PATH)
    # 确保保存的是最终的 scaler 供后续预测部署使用
    joblib.dump(final_scaler, SCALER_SAVE_PATH)
    logger.info(f"✅ 生产级模型与特征索引已持久化至: {MODEL_SAVE_PATH}")