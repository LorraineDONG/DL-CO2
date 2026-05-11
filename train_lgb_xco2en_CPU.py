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
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SF_lgb_training(A01-No lead and lag).log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SF_optuna_lgb_study-A01.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/train_lgb_xco2en_SF_best_params-A01.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SF-lgb_model-A01.pkl'    
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
    df_clean['no2_trop_log'] = np.log(df_clean['no2_trop'])
    
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
    file_path = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    target = 'xco2_enhanced'
    
    # 🌟 直接使用与 RF 完全一致的特征集合
    golden_features = [
        'era5_blh', 'era5_d2m', 'era5_sp', 'era5_ssrd', 'era5_t2m', 'era5_tcwv', 
        'era5_u100', 'era5_v100', 'era5_u10', 'era5_v10','era5_wind_speed',
        'grid_lat', 'grid_lon', 'dem_mean', 
        'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'ndvi_t2m_cross', 'ssrd_t2m_cross', 'ntl_nox_cross',
        'meic_nox', 'ntl', 'ndvi','no2_trop_log',
    ]
    
    # 1. 准备全局数据
    df = load_and_preprocess(file_path)
    X_full = df[golden_features]
    y_full = df[target]
    
    # 2. 数据划分：切分出 20% 绝对盲测集 (完全冻结)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )
    logger.info(f"✨ 输入特征组合 ({len(golden_features)}个): {golden_features}")
    
    # 3. 参数优化 (仅在 80% 的 Pool 上进行，杜绝参数泄露)
    best_params = optimize_hyperparameters(X_pool, y_pool, n_trials=500)
    
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ 最优参数已安全保存至: {PARAMS_JSON}")

    # ==========================================
    # 4. 终极盲测阶段 (严禁测试集参与 Early Stopping)
    # ==========================================
    logger.info("🏁 阶段三：执行终极盲测集评估，并对比训练集性能...")
    
    initial_lr = best_params.pop('learning_rate')
    eval_model = lgb.LGBMRegressor(**best_params, n_estimators=10000, objective='huber') 
    
    # 【修复 1】：从 pool 中再切分内部验证集用于 early stopping，对 test 集严格保密
    X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
        X_pool, y_pool, test_size=0.1, random_state=42
    )

    lr_scheduler = lgb.reset_parameter(
        learning_rate=lambda iter: initial_lr * (0.999 ** iter) 
    )
    
    # 训练评估模型
    eval_model.fit(
        X_train_es, y_train_es, 
        eval_set=[(X_val_es, y_val_es)],
        eval_metric=['rmse', 'mae'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),
            lr_scheduler
        ]
    )
    
    # 指标计算与对比
    y_pred_test = eval_model.predict(X_test)
    y_pred_train_final = eval_model.predict(X_pool)
    
    train_r2_final = r2_score(y_pool, y_pred_train_final)
    train_rmse_final = np.sqrt(mean_squared_error(y_pool, y_pred_train_final))

    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_bias = np.mean(y_pred_test - y_test)

    # 提取特征重要性
    raw_gain = eval_model.booster_.feature_importance(importance_type='gain')
    normalized_gain = raw_gain / raw_gain.sum()
    importance_df = pd.DataFrame({
        'Feature': eval_model.feature_name_,
        'Importance (Normalized Gain)': normalized_gain
    }).sort_values(by='Importance (Normalized Gain)', ascending=False)
    
    # 输出统一报告
    logger.info("="*30 + " LIGHTGBM FINAL REPORT " + "="*30)
    logger.info(f"Train R²  : {train_r2_final:.4f}")
    logger.info(f"Train RMSE: {train_rmse_final:.4f} ppm")
    logger.info("-" * 25)
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    
    logger.info("-" * 25 + " 气象与卫星因子贡献度排名前 15 " + "-" * 25)
    for idx, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['Feature']:>22} : {row['Importance (Normalized Gain)']:.4f}")
    logger.info("="*85)

    # ==========================================
    # 5. 训练并保存最终生产模型 (使用 100% 数据)
    # ==========================================
    logger.info("💾 阶段四：使用 100% 数据训练最终生产模型并固化...")
    
    production_model = lgb.LGBMRegressor(**best_params, n_estimators=10000, objective='huber') 
    
    # 【修复 2】：为生产模型重新切分独立的 5% early stopping 验证集
    X_prod_train, X_prod_val, y_prod_train, y_prod_val = train_test_split(
        X_full, y_full, test_size=0.05, random_state=42
    )

    prod_lr_scheduler = lgb.reset_parameter(
        learning_rate=lambda iter: initial_lr * (0.999 ** iter) 
    )

    production_model.fit(
        X_prod_train, y_prod_train, 
        eval_set=[(X_prod_val, y_prod_val)],
        eval_metric=['rmse', 'mae'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),
            prod_lr_scheduler
        ]
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(production_model, MODEL_SAVE_PATH)
    logger.info(f"✅ 生产级模型已持久化至: {MODEL_SAVE_PATH}")