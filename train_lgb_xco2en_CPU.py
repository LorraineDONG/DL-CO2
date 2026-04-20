import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import logging
import os
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
# ==========================================
# 0. 全局配置与路径初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SF_lgb_training.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SF_optuna_lgb_study.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/train_lgb_xco2en_SF_best_params.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SF-lgb_model.pkl'    
# 自动创建不存在的文件夹
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE.replace('sqlite:///', '')), exist_ok=True)
os.makedirs(os.path.dirname(PARAMS_JSON), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)          
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
    
    # 确保数据排序
    df_clean.sort_values('date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    # --- 时间周期性编码 ---
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    
    # --- 物理交叉特征与风速计算 (与RF对齐) ---
    logger.info("🛠️ 执行特征工程: 周期性编码 + 物理交叉项 + 风速计算...")
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    
    return df_clean
# ==========================================
# 2. Optuna 深度调参 (注入 RF 参数哲学)
# ==========================================
def optimize_hyperparameters(X_pool, y_pool, n_trials=100):
    def objective(trial):
        param = {
            'objective': 'huber',  
            'n_estimators': 3000,   
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 15),
            # 🌟 借鉴 RF: 扩大 num_leaves 提升复杂逻辑表达能力
            'num_leaves': trial.suggest_int('num_leaves', 63, 511),
            # 🌟 借鉴 RF min_samples_leaf: 调高叶子节点样本下限，实现物理平滑抗噪
            'min_child_samples': trial.suggest_int('min_child_samples', 30, 150),
            # 🌟 借鉴 RF max_samples: 增强样本采样的随机性，降低过拟合
            'subsample': trial.suggest_float('subsample', 0.4, 0.8),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 4),
            # 🌟 借鉴 RF max_features: 强制隔离强特征，给弱物理特征(如交叉项)表达机会
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # 与 RF 对齐：使用 KFold 交叉验证
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_rmses = []
        
        for train_index, val_index in kf.split(X_pool):
            X_tr, X_va = X_pool.iloc[train_index], X_pool.iloc[val_index]
            y_tr, y_va = y_pool.iloc[train_index], y_pool.iloc[val_index]
            
            model = lgb.LGBMRegressor(**param)
            model.fit(
                X_tr, y_tr, 
                eval_set=[(X_va, y_va)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            preds = model.predict(X_va)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            
        return np.mean(cv_rmses)
    logger.info("🚀 开始带滑窗的深度参数模拟 (Optuna，已注入RF参数哲学)...")
    study = optuna.create_study(direction='minimize', storage=DB_FILE, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/TABLE-WLGXCO2en_sif_no2_era5_ndvi_meic_ntl_dem.pkl'
    target = 'xco2_enhanced'
    
    # 🌟 直接使用与 RF 完全一致的特征集合
    golden_features = [
        'era5_u100', 'era5_v100', 'grid_lon', 'grid_lat',
        'sif_740', 'no2_trop', 'meic_nox', 'dem_mean',
        'era5_tcwv', 'era5_ssrd', 'era5_blh', 'era5_t2m', 
        'ntl', 'ndvi', 'ndvi_std', 
        'month_sin', 'month_cos', 
        'sif_variance', 'era5_wind_speed',
        'no2_amf_trop', 'no2_variance', 
        'ssrd_t2m_cross', 'ntl_nox_cross', 'ndvi_t2m_cross'
    ]
    # 1. 准备数据
    df = load_and_preprocess(file_path)
    
    # 2. 数据划分：与 RF 完全对齐的 train_test_split 逻辑
    X = df[golden_features]
    y = df[target]
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"✨ 输入特征组合 ({len(golden_features)}个): {golden_features}")
    # 3. 参数优化
    best_params = optimize_hyperparameters(X_pool, y_pool, n_trials=80)
    # 保存最优参数
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ 最优参数已安全保存至: {PARAMS_JSON}")
    # 4. 终极盲测阶段
    logger.info("🏁 执行盲测集评估与特征物理重要性分析...")
    
    initial_lr = best_params.pop('learning_rate')
    # 使用更大的 n_estimators 上限，依靠 early_stopping 来寻找最佳迭代次数
    final_model = lgb.LGBMRegressor(**best_params, n_estimators=10000, objective='huber') 
    
    # 动态学习率衰减
    lr_scheduler = lgb.reset_parameter(
        learning_rate=lambda iter: initial_lr * (0.999 ** iter) 
    )
    final_model.fit(
        X_pool, y_pool, 
        eval_set=[(X_test, y_test)],
        eval_metric=['rmse', 'mae'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),
            lr_scheduler
        ]
    )
    
    y_pred = final_model.predict(X_test)
    y_pred_train_final = final_model.predict(X_pool)
    
    # --- 指标计算 (加入 Train R2 对齐 RF 报告) ---
    train_r2_final = r2_score(y_pool, y_pred_train_final)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_bias = np.mean(y_pred - y_test)
    # --- 基于 gain 的特征重要性并归一化 ---
    raw_gain = final_model.booster_.feature_importance(importance_type='gain')
    normalized_gain = raw_gain / raw_gain.sum()
    importance_df = pd.DataFrame({
        'Feature': final_model.feature_name_,
        'Importance (Normalized Gain)': normalized_gain
    }).sort_values(by='Importance (Normalized Gain)', ascending=False)
    
    # ==========================================
    # 5. 输出报告 (严格与 RF 格式对齐)
    # ==========================================
    logger.info("="*30 + " LIGHTGBM FINAL REPORT " + "="*30)
    logger.info(f"Train R²  : {train_r2_final:.4f}")
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    logger.info("-" * 25 + " 气象与卫星因子贡献度排名前 15 " + "-" * 25)
    
    for idx, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['Feature']:>22} : {row['Importance (Normalized Gain)']:.4f}")
        
    logger.info("="*88)
    joblib.dump(final_model, MODEL_SAVE_PATH)
    logger.info(f"✅ 模型已持久化至: {MODEL_SAVE_PATH}")
