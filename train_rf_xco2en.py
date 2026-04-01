import pandas as pd
import numpy as np
import optuna
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 全局配置与日志初始化
# ==========================================
LOG_FILE = 'rf_training.log'
# RF 的搜索空间较 GBDT 简单，使用独立数据库
DB_FILE = 'sqlite:///optuna_rf_study.db' 

if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 数据加载与特征工程 (保持与 GBDT 一致)
# ==========================================
def load_and_preprocess(file_path):
    logger.info(f"📂 正在加载数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    
    # 注入交叉特征
    logger.info("🛠️ 执行特征工程: 时间周期 + 物理交叉项...")
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    
    return df_clean

# ==========================================
# 2. Optuna 深度超参数优化 (RF 专属版)
# ==========================================
def optimize_rf(X_train, y_train, X_val, y_val, n_trials=50):
    def objective(trial):
        # 1. 定义 RF 的搜索空间
        param = {
            # 树的数量，通常越多越稳，但增加计算时间
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            
            # 树的最大深度，控制过拟合
            'max_depth': trial.suggest_int('max_depth', 5, 40),
            
            # 节点分裂所需的最小样本数
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            
            # 叶子节点所需的最小样本数 (核心正则化参数)
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            
            # 寻找最佳分裂点时考虑的特征比例 (相当于 colsample)
            'max_features': trial.suggest_float('max_features', 0.5, 1.0),
            
            'random_state': 42,
            'n_jobs': -1, # 随机森林训练天然支持多核并行，速度很快
            'bootstrap': True # 默认开启 Bagging
        }
        
        # 2. 初始化 RF 模型
        model = RandomForestRegressor(**param)
        
        # 3. 训练模型 (RF 没有内建早停回调，通常直接训练全量树)
        model.fit(X_train, y_train)
        
        # 4. 在验证集上评估
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    logger.info("🚀 开始 Random Forest 深度参数模拟 (Optuna)...")
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction='minimize', storage=DB_FILE, sampler=sampler, load_if_exists=True)
    
    # RF 调参通常比 GBDT 快，设 50 次足矣
    study.optimize(objective, n_trials=n_trials) 
    
    logger.info("="*40)
    logger.info(f"🏆 RF 最佳 Val RMSE: {study.best_value:.4f}")
    logger.info(f"最佳参数: {study.best_params}")
    return study.best_params

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/gridded_oco_sif_no2_era5_ndvi_meic_ntl.pkl'
    target = 'xco2_enhanced'
    
    # 🌟 直接使用你之前的 27 个黄金特征
    golden_features = [
        'era5_wind_dir_100m', 'era5_wind_speed_100m', 'era5_tcwv', 'era5_ssrd', 'era5_blh', 
        'era5_v100', 'era5_u100', 'era5_t2m', 'ndvi', 'ndvi_std', 'sif_time_tai93', 'sif_sza', 
        'sif_uncertainty', 'sif_vza', 'sif_raz', 'no2_trop', 'no2_vaa', 'no2_vza', 'no2_sza', 
        'no2_amf_trop', 'no2_trop_std', 'no2_time', 'mean_hour', 'month', 'season', 
        'ssrd_t2m_cross', 'ntl_nox_cross'
    ]

    # 1. 准备数据
    df = load_and_preprocess(file_path)
    
    train_df = df[df['year'] <= 2020]
    val_df   = df[df['year'] == 2021]
    test_df  = df[df['year'] >= 2022]

    X_train_raw = train_df[golden_features].values
    X_val_raw   = val_df[golden_features].values
    X_test_raw  = test_df[golden_features].values

    y_train = train_df[target].values
    y_val   = val_df[target].values
    y_test  = test_df[target].values

    # 虽然 RF 不需要，为了保持处理流一致性进行 Scaling
    logger.info("⚖️ 特征标准化 (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val   = scaler.transform(X_val_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    # 2. 执行全自动参数优化 (试设 50 次)
    best_params = optimize_rf(X_train, y_train, X_val, y_val, n_trials=50)

    # 3. 终极盲测 (Test Set)
    logger.info("🏁 执行 Random Forest 盲测集评估...")
    
    # 结合最佳参数训练最终模型
    final_model = RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)
    final_model.fit(X_train, y_train)
    
    y_pred = final_model.predict(X_test)
    
    logger.info("="*30 + " RANDOM FOREST FINAL REPORT " + "="*30)
    logger.info(f"Test R²  : {r2_score(y_test, y_pred):.4f}")
    logger.info(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    logger.info(f"Test MAE : {mean_absolute_error(y_test, y_pred):.4f}")
    logger.info("="*88)