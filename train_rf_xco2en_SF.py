import pandas as pd
import numpy as np
import optuna
import logging
import os
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 0. 全局配置与路径初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SF_rf_training.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SF_optuna_rf_study.db' 
PARAMS_JSON = '/home/whdong/dl/train_rf_xco2en_SF_best_params.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SF-rf_model.pkl' # 模型保存路径
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SF-rf_scaler.pkl'    # 必须同时保存标准化器

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
# 1. 数据加载与特征工程 (升级时间周期性编码)
# ==========================================
def load_and_preprocess(file_path):
    logger.info(f"📂 正在加载数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    
    # ⚠️ 确保数据严格按时间顺序排列，供 TimeSeriesSplit 使用
    df_clean.sort_values('date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    # --- 升级：时间特征周期性编码 ---     logger.info("🛠️ 执行特征工程: 周期性编码 + 物理交叉项...")
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    
    # --- 物理交叉特征 ---
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    
    return df_clean

# ==========================================
# 2. Optuna + TimeSeriesSplit 深度超参数优化
# ==========================================
def optimize_rf(X_pool, y_pool, n_trials=50):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 20, 80),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.5, 1.0),
            # 新增：控制 Bootstrap 采样率，防止过拟合
            'max_samples': trial.suggest_float('max_samples', 0.5, 0.95),
            'random_state': 42,
            'n_jobs': -1,
            'bootstrap': True 
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        cv_rmses = []
        
        # 使用滑窗交叉验证
        for train_index, val_index in tscv.split(X_pool):
            X_tr_raw, X_va_raw = X_pool.iloc[train_index].values, X_pool.iloc[val_index].values
            y_tr, y_va = y_pool.iloc[train_index].values, y_pool.iloc[val_index].values
            
            # ⚠️ 关键修正：在每一次滑窗内独立进行标准化，防止未来数据泄露
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_va = scaler.transform(X_va_raw)
            
            model = RandomForestRegressor(**param)
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_va)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            
        return np.mean(cv_rmses)

    logger.info("🚀 开始 Random Forest 深度参数模拟 (含时间滑窗CV)...")
    study = optuna.create_study(direction='minimize', storage=DB_FILE, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials) 
    return study.best_params

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/gridded0.25_xco2sf_sif_no2_era5_ndvi_meic_ntl.pkl'
    target = 'xco2_enhanced'
    
    # 加入新的周期性特征
    golden_features = [
        'era5_wind_dir_100m', 'era5_wind_speed_100m', 'era5_tcwv', 'era5_ssrd', 'era5_blh', 
        'era5_v100', 'era5_u100', 'era5_t2m', 'ndvi', 'ndvi_std', 'sif_time_tai93', 'sif_sza', 
        'sif_uncertainty', 'sif_vza', 'sif_raz', 'no2_trop', 'no2_vaa', 'no2_vza', 'no2_sza', 
        'no2_amf_trop', 'no2_trop_std', 'no2_time', 'mean_hour', 'month', 'season', 
        'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 
        'ssrd_t2m_cross', 'ntl_nox_cross', 'ndvi_t2m_cross'
    ]

    # 1. 准备数据
    df = load_and_preprocess(file_path)
    
    train_val_pool = df[df['year'] <= 2021]
    test_df        = df[df['year'] >= 2022]

    X_pool = train_val_pool[golden_features]
    y_pool = train_val_pool[target]
    X_test_raw = test_df[golden_features].values
    y_test = test_df[target].values

    # 2. 执行全自动参数优化
    best_params = optimize_rf(X_pool, y_pool, n_trials=50)

    # 保存参数
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ RF 最优参数已安全保存至: {PARAMS_JSON}")

    # 3. 终极盲测阶段
    logger.info("🏁 执行 Random Forest 盲测集评估与特征重要性分析...")
    logger.info(f"✅ 模型已保存至: {MODEL_SAVE_PATH}")
    logger.info(f"✅ 标准化器已保存至: {SCALER_SAVE_PATH}")

    # ⚠️ 最终训练前，对全体历史数据进行统一标准化
    final_scaler = StandardScaler()
    X_pool_scaled = final_scaler.fit_transform(X_pool.values)
    X_test_scaled = final_scaler.transform(X_test_raw)
    
    final_model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42,criterion='absolute_error')
    final_model.fit(X_pool_scaled, y_pool.values)
    
    y_pred = final_model.predict(X_test_scaled)
    
    # --- 指标计算 (四维体系) ---
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_bias = np.mean(y_pred - y_test)

    # --- 提取特征重要性 ---
    # Random Forest 使用的是基尼不纯度 (Gini Impurity) 减少量来计算特征重要性
    importance_df = pd.DataFrame({
        'Feature': golden_features,
        'Importance (Gini)': final_model.feature_importances_
    }).sort_values(by='Importance (Gini)', ascending=False)
    
    # 输出报告
    logger.info("="*30 + " RANDOM FOREST FINAL REPORT " + "="*30)
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    logger.info("-" * 25 + " 气象与卫星因子贡献度排名前 10 " + "-" * 25)
    
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['Feature']:>20} : {row['Importance (Gini)']:.4f}")
        
    logger.info("="*88)

    # 保存模型实体
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(final_model, MODEL_SAVE_PATH)
    # 保存标准化器（预测新数据时必须使用和训练时一模一样的均值和方差）
    joblib.dump(final_scaler, SCALER_SAVE_PATH)