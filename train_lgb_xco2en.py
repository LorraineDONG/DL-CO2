import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import logging
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 0. 全局配置与日志初始化
# ==========================================
LOG_FILE = 'final_training_log.log'
DB_FILE = 'sqlite:///optuna_study.db'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 数据加载与极速清洗
# ==========================================
def load_and_preprocess(file_path):
    logger.info(f"📂 正在加载数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    
    # --- 特征工程 (领域知识注入) ---
    logger.info("🛠️ 执行特征工程: 时间周期 + 物理交叉项...")
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    
    return df_clean

# ==========================================
# 2. 自动化特征淘汰 (RFE)
# ==========================================
def recursive_feature_elimination(X_train, y_train, X_val, y_val, initial_features):
    logger.info("🕵️‍♂️ 启动特征淘汰赛 (RFE)...")
    current_features = list(initial_features)
    best_r2 = -np.inf
    best_subset = current_features.copy()

    while len(current_features) > 20:
        model = lgb.LGBMRegressor(
                n_estimators=1000,   # 增加树的数量，让重要性评估更稳定
                learning_rate=0.03,  # 稍微降低学习率
                random_state=42, 
                n_jobs=-1, 
                verbose=-1
                )
        model.fit(X_train[current_features], y_train, eval_set=[(X_val[current_features], y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        
        preds = model.predict(X_val[current_features])
        curr_r2 = r2_score(y_val, preds)
        
        if curr_r2 > best_r2:
            best_r2 = curr_r2
            best_subset = current_features.copy()
            
        # 剔除权重最低的特征
        importances = model.booster_.feature_importance(importance_type='gain')
        worst_idx = np.argmin(importances)
        worst_feat = current_features.pop(worst_idx)
        logger.info(f"Count: {len(current_features)+1} | Val R²: {curr_r2:.4f} | 剔除: {worst_feat}")

    logger.info(f"🏆 筛选完成！最佳特征数量: {len(best_subset)}")
    return best_subset

# ==========================================
# 3. Optuna 深度超参数优化
# ==========================================
def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100):
    def objective(trial):
        param = {
            'n_estimators': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            
            # --- 你提到的 5 个核心高级参数 ---
            # 1. 防止无效分裂 (L1 正则化的补充)
            'min_split_gain': trial.suggest_float('min_split_gain', 0, 1.0),
            
            # 2. 增强遥感数据精度 (直方图算法的分箱数)
            'max_bin': trial.suggest_int('max_bin', 128, 512),
            
            # 3. Bagging 频率 (必须配合 subsample < 1.0 使用)
            'subsample': trial.suggest_float('subsample', 0.4, 0.9),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            
            # 4. 决定叶子节点权重的最小和 (对异常值敏感度)
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            
            # 5. 节点级特征随机化 (比 colsample_bytree 更细致)
            'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.5, 1.0),
            
            # 其他正则化
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'metric': 'rmse'
        }
        
        model = lgb.LGBMRegressor(**param)
        pruning_cb = optuna.integration.LightGBMPruningCallback(trial, 'rmse', valid_name='valid_0')
        
        model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                pruning_cb
            ]
        )
        
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    logger.info("🚀 开始深度参数模拟 (Optuna)...")
    study = optuna.create_study(direction='minimize', storage=DB_FILE, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/gridded_oco_sif_no2_era5_ndvi_meic_ntl.pkl'
    target = 'xco2_enhanced'
    raw_features = [
        'era5_wind_dir_100m', 'era5_wind_speed_100m', 'era5_tcwv', 'era5_ssrd', 'era5_blh', 
        'era5_v100', 'era5_u100', 'era5_t2m', 'ntl', 'ndvi', 'ndvi_std', 'sif_time_tai93', 
        'sif_sza', 'sif_740', 'sif_uncertainty', 'sif_vza', 'sif_raz', 'meic_nox', 
        'no2_trop', 'no2_vaa', 'no2_vza', 'no2_sza', 'no2_amf_trop', 'no2_trop_std', 
        'no2_time', 'mean_hour', 'total_obs_count'
    ]

    # 1. 准备数据
    df = load_and_preprocess(file_path)
    all_features = raw_features + ['month', 'season', 'ndvi_t2m_cross', 'ssrd_t2m_cross', 'ntl_nox_cross']
    
    train_df = df[df['year'] <= 2020]
    val_df   = df[df['year'] == 2021]
    test_df  = df[df['year'] >= 2022]

    # 2. 特征淘汰赛
    golden_features = recursive_feature_elimination(
        train_df, train_df[target], val_df, val_df[target], all_features
    )

    # 3. 参数优化
    X_train_final, X_val_final = train_df[golden_features], val_df[golden_features]
    best_params = optimize_hyperparameters(X_train_final, train_df[target], X_val_final, val_df[target], n_trials=200)

    # 4. 终极盲测
    logger.info("🏁 执行盲测集评估...")
    final_model = lgb.LGBMRegressor(**best_params, n_estimators=5000)
    final_model.fit(X_train_final, train_df[target], eval_set=[(X_val_final, val_df[target])],
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    
    X_test_final = test_df[golden_features]
    y_pred = final_model.predict(X_test_final)
    
    logger.info("="*30 + " FINAL REPORT " + "="*30)
    logger.info(f"Test R²: {r2_score(test_df[target], y_pred):.4f}")
    logger.info(f"Test RMSE: {np.sqrt(mean_squared_error(test_df[target], y_pred)):.4f}")
    logger.info(f"Selected Features: {golden_features}")
    logger.info("="*74)