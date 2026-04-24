import pandas as pd
import numpy as np
import optuna
import logging
import os
import json
import shap
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.inspection import permutation_importance

# ==========================================
# 0. 全局配置与路径初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_WLG_rf_training.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_WLG_optuna_rf_study.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/train_rf_xco2en_WLG_best_params.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_WLG-rf_model.pkl' 
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_WLG-rf_scaler.pkl'    

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
    
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    
    df_clean.sort_values('date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    df_clean['no2_trop_log'] = np.log1p(np.maximum(df_clean['no2_trop'], 0))
    df_clean['no2_co_cross'] = df_clean['no2_trop'] / (df_clean['co'] + 1e-8)

    return df_clean

# ==========================================
# 2. 自定义递归特征消除 
# ==========================================
def auto_feature_selection(X_train, y_train, candidate_features, force_keep, cv_splits=10):
    logger.info("🕵️‍♂️ 开始自动化特征筛选 (Custom RFE)...")
    current_features = list(candidate_features)
    best_features = list(candidate_features)
    best_rmse = float('inf')
    
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    # 只要当前特征数大于强制保留的特征数，就继续尝试淘汰
    while len(current_features) >= len(force_keep):
        cv_rmses = []
        
        # 1. 快速交叉验证评估当前特征组合
        for train_idx, val_idx in kf.split(X_train):
            X_tr_raw = X_train.iloc[train_idx][current_features].values
            X_va_raw = X_train.iloc[val_idx][current_features].values
            y_tr, y_va = y_train.iloc[train_idx].values, y_train.iloc[val_idx].values
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_va = scaler.transform(X_va_raw)
            
            # 使用轻量级 RF 快速评估
            model = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            
        mean_rmse = np.mean(cv_rmses)
        logger.info(f"[{len(current_features):2d} 个特征] 当前 CV RMSE: {mean_rmse:.4f}")
        
        # 2. 更新历史最优特征组合
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_features = list(current_features)
            
        # 如果已经只剩下必须保留的特征，退出循环
        if len(current_features) == len(force_keep):
            break
            
        # 3. 计算重要性，决定淘汰哪个特征
        scaler_full = StandardScaler()
        X_full_scaled = scaler_full.fit_transform(X_train[current_features])
        model_full = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
        model_full.fit(X_full_scaled, y_train.values)
        
        importance_df = pd.DataFrame({
            'Feature': current_features,
            'Importance': model_full.feature_importances_
        })
        
        # 排除必须保留的特征，在剩下的特征中找到重要性最低的
        droppable = importance_df[~importance_df['Feature'].isin(force_keep)]
        least_important = droppable.sort_values('Importance').iloc[0]['Feature']
        
        current_features.remove(least_important)
        logger.info(f"   -> 🔪 剔除特征: {least_important}")
        
    logger.info(f"✅ 特征筛选完成！最终保留 {len(best_features)} 个特征，最优 CV RMSE: {best_rmse:.4f}")
    return best_features

# ==========================================
# 3. Optuna + KFold 深度超参数优化
# ==========================================
def optimize_rf(X_pool, y_pool, n_trials=100):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 80),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 40),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 500, 2000),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.1, 0.4]),
            'max_samples': trial.suggest_float('max_samples', 0.2, 0.8),
            'random_state': 42,
            'n_jobs': -1,
            'bootstrap': True
        }
        
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_rmses = []
        
        for train_index, val_index in kf.split(X_pool):
            X_tr_raw, X_va_raw = X_pool.iloc[train_index].values, X_pool.iloc[val_index].values
            y_tr, y_va = y_pool.iloc[train_index].values, y_pool.iloc[val_index].values
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_va = scaler.transform(X_va_raw)
            
            model = RandomForestRegressor(**param)
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_va)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            
        return np.mean(cv_rmses)

    logger.info("🚀 开始加强正则化的 Random Forest 深度参数模拟...")
    study = optuna.create_study(direction='minimize', storage=DB_FILE, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials) 
    return study.best_params

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/TABLE-SHPXCO2en_sif_no2_era5_ndvi_meic_ntl_dem_co.pkl'
    target = 'xco2_enhanced'
    
    # 特征池
    candidate_features = [
    'era5_blh', #'era5_blh_lag1', 'era5_blh_lag2', 'era5_blh_lag3', #'era5_blh_lead1', 'era5_blh_lead2', 'era5_blh_lead3', 
    'era5_d2m', #'era5_d2m_lag1', 'era5_d2m_lag2', 'era5_d2m_lag3',  #'era5_d2m_lead1', 'era5_d2m_lead2', 'era5_d2m_lead3', 
    'era5_sp',  #'era5_sp_lag1', 'era5_sp_lag2', 'era5_sp_lag3', #'era5_sp_lead1', 'era5_sp_lead2', 'era5_sp_lead3', 
    'era5_ssrd',#'era5_ssrd_lag1', 'era5_ssrd_lag2', 'era5_ssrd_lag3', #'era5_ssrd_lead1', 'era5_ssrd_lead2', 'era5_ssrd_lead3', 
    'era5_t2m', #'era5_t2m_lag1', 'era5_t2m_lag2', 'era5_t2m_lag3', #'era5_t2m_lead1', 'era5_t2m_lead2', 'era5_t2m_lead3', 
    'era5_tcwv',#'era5_tcwv_lag1', 'era5_tcwv_lag2', 'era5_tcwv_lag3', #'era5_tcwv_lead1', 'era5_tcwv_lead2', 'era5_tcwv_lead3', 
    'era5_u100', #'era5_u100_lag1', 'era5_u100_lag2', 'era5_u100_lag3', 'era5_u100_lead1', 'era5_u100_lead2', 'era5_u100_lead3', 
    #'era5_u10', 'era5_u10_lag1', 'era5_u10_lag2', 'era5_u10_lag3', 'era5_u10_lead1', 'era5_u10_lead2', 'era5_u10_lead3', 
    'era5_v100', #'era5_v100_lag1', 'era5_v100_lag2', 'era5_v100_lag3', 'era5_v100_lead1', 'era5_v100_lead2', 'era5_v100_lead3', 
    #'era5_v10', #'era5_v10_lag1', 'era5_v10_lag2', 'era5_v10_lag3', 'era5_v10_lead1', 'era5_v10_lead2', 'era5_v10_lead3', 
    #'era5_wind_dir_100m', 'era5_wind_dir_100m_lag1', 'era5_wind_dir_100m_lag2', 'era5_wind_dir_100m_lag3', 'era5_wind_dir_100m_lead1', #'era5_wind_dir_100m_lead2', 'era5_wind_dir_100m_lead3', 
    #'era5_wind_dir_10m', 'era5_wind_dir_10m_lag1', #'era5_wind_dir_10m_lag2', 'era5_wind_dir_10m_lag3', 'era5_wind_dir_10m_lead1', #'era5_wind_dir_10m_lead2', 'era5_wind_dir_10m_lead3', 
    #'era5_wind_speed_100m','era5_wind_speed_100m_lag1', 'era5_wind_speed_100m_lag2', 'era5_wind_speed_100m_lag3', 'era5_wind_speed_100m_lead1', #'era5_wind_speed_100m_lead2', 'era5_wind_speed_100m_lead3', 
    #'era5_wind_speed_10m', # 'era5_wind_speed_10m_lag1', 'era5_wind_speed_10m_lag2', 'era5_wind_speed_10m_lag3', 'era5_wind_speed_10m_lead1', #'era5_wind_speed_10m_lead2', 'era5_wind_speed_10m_lead3', 
    'grid_lon', 'grid_lat', 
    'ndvi_t2m_cross','ssrd_t2m_cross','ntl_nox_cross','no2_trop_log','no2_co_cross',
    'meic_nox', 'ndvi', 'ndvi_std', 'ntl', 
    'no2_amf_trop', 'no2_variance', 
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos', 
    'co_variance', 'dem_mean', 'dem_std',
    'sif_740', 'sif_variance'
]
    
    # 强制保留名单
    force_keep_features = ['no2_trop'] #'co', 
    
    # 准备数据
    df = load_and_preprocess(file_path)
    X_full = df[candidate_features]
    y_full = df[target]
    
    # 预留测试集，绝对不参与特征筛选和调参
    X_pool_raw, X_test_raw, y_pool, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    # 自动化特征筛选
    best_features = auto_feature_selection(X_pool_raw, y_pool, candidate_features, force_keep_features)
    
    # 筛选后重组数据集
    X_pool = X_pool_raw[best_features]
    X_test = X_test_raw[best_features]
    logger.info(f"✨ 最终进入 Optuna 的精简特征组合 ({len(best_features)}个): {best_features}")

    # 自动化参数优化 (Optuna)
    best_params = optimize_rf(X_pool, y_pool, n_trials=50) # 建议测试时先设 50 试运行
    
    max_trees = 1000
    best_params['n_estimators'] = max_trees
    
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ RF 最优参数已安全保存至: {PARAMS_JSON}")

    # 终极评估阶段：执行热启动训练监控
    logger.info("🏁 进入终极评估阶段：执行热启动训练监控 (warm_start=True)...")

    final_scaler = StandardScaler()
    X_pool_scaled = final_scaler.fit_transform(X_pool.values)
    X_test_scaled = final_scaler.transform(X_test.values)
    
    rf_config = best_params.copy()
    rf_config.pop('n_estimators', None)
    
    final_model = RandomForestRegressor(
        **rf_config, 
        n_estimators=0,     
        warm_start=True,    
        n_jobs=-1, 
        random_state=42, 
        criterion='squared_error'
    )

    step = 50
    logger.info(f"{'Trees':>6} | {'Train R2':>10} | {'Test R2':>10} | 状态")
    logger.info("-" * 50)

    for current_trees in range(step, max_trees + 1, step):
        final_model.n_estimators = current_trees
        final_model.fit(X_pool_scaled, y_pool.values)
        
        cur_pred_train = final_model.predict(X_pool_scaled)
        cur_pred_test  = final_model.predict(X_test_scaled)
        
        cur_train_r2 = r2_score(y_pool.values, cur_pred_train)
        cur_test_r2  = r2_score(y_test, cur_pred_test)
        logger.info(f"{current_trees:>6} | {cur_train_r2:>10.4f} | {cur_test_r2:>10.4f} | 训练中...")

    # 最终指标汇总与百分比重要性提取
    y_pred = final_model.predict(X_test_scaled)
    y_pred_train_final = final_model.predict(X_pool_scaled)

    train_r2_final = r2_score(y_pool.values, y_pred_train_final)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_bias = np.mean(y_pred - y_test)

    # --- 提取特征重要性 (Gini 百分比) ---
    raw_gini = final_model.feature_importances_
    pct_gini = (raw_gini / raw_gini.sum()) * 100
    
    importance_df = pd.DataFrame({
        'Feature': best_features,
        'Importance (%)': pct_gini
    }).sort_values(by='Importance (%)', ascending=False)
    
    logger.info("="*30 + " RANDOM FOREST FINAL REPORT " + "="*30)
    logger.info(f"Train R²  : {train_r2_final:.4f}")
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    
    logger.info("-" * 25 + " Gini 特征贡献度排名 " + "-" * 25)
    for idx, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['Feature']:>22} : {row['Importance (%)']:>6.2f}%")
        
    # --- 提取特征重要性 (Permutation 百分比) ---
    perm_result = permutation_importance(
        final_model, X_test_scaled, y_test, 
        n_repeats=5, random_state=42, n_jobs=-1
    )
    
    # 防止负值导致百分比计算错误，截断到0
    raw_perm = np.maximum(perm_result.importances_mean, 0) 
    pct_perm = (raw_perm / (raw_perm.sum() + 1e-9)) * 100
    
    perm_importance_df = pd.DataFrame({
        'Feature': best_features,
        'Importance (%)': pct_perm
    }).sort_values(by='Importance (%)', ascending=False)
    
    logger.info("-" * 25 + " Permutation 真实重要性排名 " + "-" * 25)
    for idx, row in perm_importance_df.head(15).iterrows():
        logger.info(f"  {row['Feature']:>22} : {row['Importance (%)']:>6.2f}%")
    
    logger.info("-" * 25 + " 开始计算 SHAP 真实归因重要性 (耗时较长请耐心等待) " + "-" * 25)
# ==========================================
# 5. SHAP
# ==========================================
    # 1. 初始化 SHAP 树解释器
    # 注意：为了节省时间，通常在测试集 (X_test_scaled) 上计算 SHAP，而不是全量数据
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # 2. 计算每个特征的平均绝对 SHAP 值 (Mean |SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # 3. 转换为百分比格式
    pct_shap = (mean_abs_shap / (mean_abs_shap.sum() + 1e-9)) * 100
    
    # 4. 构建报告 DataFrame 并按百分比排序
    shap_importance_df = pd.DataFrame({
        'Feature': best_features,
        'Importance (SHAP %)': pct_shap
    }).sort_values(by='Importance (SHAP %)', ascending=False)
    
    for idx, row in shap_importance_df.iterrows():
        logger.info(f"  {row['Feature']:>22} : {row['Importance (SHAP %)']:>6.2f}%")
    logger.info("="*88)

    # 保存最终模型
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(final_model, MODEL_SAVE_PATH)
    joblib.dump(final_scaler, SCALER_SAVE_PATH)
    logger.info(f"✅ 模型与标准化器保存至: {os.path.dirname(MODEL_SAVE_PATH)}")