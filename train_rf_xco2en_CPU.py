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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# ==========================================
# 0. 全局配置与路径初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SHP_rf_training.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SHP_optuna_rf_study.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/train_rf_xco2en_SHP_best_params.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-rf_model.pkl' # 模型保存路径
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-rf_scaler.pkl'    # 必须同时保存标准化器

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

    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    
    return df_clean

# ==========================================
# 2. Optuna + TimeSeriesSplit 深度超参数优化
# ==========================================
def optimize_rf(X_pool, y_pool, n_trials=100):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50), # n_estimators=森林中决策树的数量，理论上越多越好，但增加到一定程度后准确率会遇到瓶颈，且会显著增加计算耗时。对于你这种复杂的地球科学数据集，500-1000 树通常是兼顾效率和精度的平衡点。
            'max_depth': trial.suggest_int('max_depth', 10, 30), # max_depth=树的最大深度，如果太深，模型会试图解释每一个异常的观测值，导致过拟合；如果太浅模型无法捕捉复杂非线性关系
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 60), # min_samples_split，一个节点至少要包含多少个样本，才允许被进一步拆分。数值越大，树的生长越保守。
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 4, 60), # min_samples_leaf=一个叶子节点（最末端）至少要包含的样本数。在处理像卫星反演这种含有观测噪声的数据时，适当增大此值，可以过滤掉噪声，使模型更稳健。
            'max_features': trial.suggest_float('max_features', 0.2, 0.6), # max_features=每棵树在拆分节点时，随机选取的特征比例。
            'max_samples': trial.suggest_float('max_samples', 0.3, 0.7), # max_samples=每棵树从训练集中随机抽取的样本比例。如果数据量非常大，减小此值可以加快训练速度并提升泛化能力。
            'random_state': 42,
            'n_jobs': -1,
            'bootstrap': True 
        }
        
        # tscv = TimeSeriesSplit(n_splits=3)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_rmses = []
        
        # 使用滑窗交叉验证
        # 使用 kf.split 替换 tscv.split
        for train_index, val_index in kf.split(X_pool):
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
    file_path = '/home/whdong/dl/TABLE-WLGXCO2en_sif_no2_era5_ndvi_meic_ntl_dem_co.pkl'
    target = 'xco2_enhanced'
    
    # 🌟 完善后的特征列表：包含空间坐标、物理交叉项与风速
    # golden_features = [
    #     'era5_u100', 'era5_v100', 'grid_lon', 'grid_lat',
    #     'sif_740', 'no2_trop', 'meic_nox', 'dem_mean',
    #     'era5_tcwv', 'era5_ssrd', 'era5_blh', 'era5_t2m', 
    #     'ntl', 'ndvi', 'ndvi_std', 
    #     'month_sin', 'month_cos', 
    #     'sif_variance', 'era5_wind_speed',
    #     'no2_amf_trop', 'no2_variance', 
    #     'ssrd_t2m_cross', 'ntl_nox_cross', 'ndvi_t2m_cross'
    # ]
    golden_features = [
    'co', 'co_variance', 'dem_mean', 'dem_std', 
    'era5_blh', 'era5_blh_lag1', 'era5_blh_lag2', 'era5_blh_lag3', 'era5_blh_lead1', 'era5_blh_lead2', 'era5_blh_lead3', 
    'era5_d2m', 'era5_d2m_lag1', 'era5_d2m_lag2', 'era5_d2m_lag3', 'era5_d2m_lead1', 'era5_d2m_lead2', 'era5_d2m_lead3', 
    'era5_sp', 'era5_sp_lag1', 'era5_sp_lag2', 'era5_sp_lag3', 'era5_sp_lead1', 'era5_sp_lead2', 'era5_sp_lead3', 
    'era5_ssrd', 'era5_ssrd_lag1', 'era5_ssrd_lag2', 'era5_ssrd_lag3', 'era5_ssrd_lead1', 'era5_ssrd_lead2', 'era5_ssrd_lead3', 
    'era5_t2m', 'era5_t2m_lag1', 'era5_t2m_lag2', 'era5_t2m_lag3', 'era5_t2m_lead1', 'era5_t2m_lead2', 'era5_t2m_lead3', 
    'era5_tcwv', 'era5_tcwv_lag1', 'era5_tcwv_lag2', 'era5_tcwv_lag3', 'era5_tcwv_lead1', 'era5_tcwv_lead2', 'era5_tcwv_lead3', 
    'era5_u100', 'era5_u100_lag1', 'era5_u100_lag2', 'era5_u100_lag3', 'era5_u100_lead1', 'era5_u100_lead2', 'era5_u100_lead3', 
    'era5_u10', 'era5_u10_lag1', 'era5_u10_lag2', 'era5_u10_lag3', 'era5_u10_lead1', 'era5_u10_lead2', 'era5_u10_lead3', 
    'era5_v100', 'era5_v100_lag1', 'era5_v100_lag2', 'era5_v100_lag3', 'era5_v100_lead1', 'era5_v100_lead2', 'era5_v100_lead3', 
    'era5_v10', 'era5_v10_lag1', 'era5_v10_lag2', 'era5_v10_lag3', 'era5_v10_lead1', 'era5_v10_lead2', 'era5_v10_lead3', 
    'era5_wind_dir_100m', 'era5_wind_dir_100m_lag1', 'era5_wind_dir_100m_lag2', 'era5_wind_dir_100m_lag3', 'era5_wind_dir_100m_lead1', 'era5_wind_dir_100m_lead2', 'era5_wind_dir_100m_lead3', 
    'era5_wind_dir_10m', 'era5_wind_dir_10m_lag1', 'era5_wind_dir_10m_lag2', 'era5_wind_dir_10m_lag3', 'era5_wind_dir_10m_lead1', 'era5_wind_dir_10m_lead2', 'era5_wind_dir_10m_lead3', 
    'era5_wind_speed_100m', 'era5_wind_speed_100m_lag1', 'era5_wind_speed_100m_lag2', 'era5_wind_speed_100m_lag3', 'era5_wind_speed_100m_lead1', 'era5_wind_speed_100m_lead2', 'era5_wind_speed_100m_lead3', 
    'era5_wind_speed_10m', 'era5_wind_speed_10m_lag1', 'era5_wind_speed_10m_lag2', 'era5_wind_speed_10m_lag3', 'era5_wind_speed_10m_lead1', 'era5_wind_speed_10m_lead2', 'era5_wind_speed_10m_lead3', 
    'grid_lat', 'grid_lon', 'meic_nox', 'ndvi', 'ndvi_std', 
    'no2_amf_trop', 'no2_trop', 'no2_variance', 'ntl', 
    'sif_740', 'sif_point_count', 'sif_variance'
]
    
    # 1. 准备数据
    df = load_and_preprocess(file_path)
    
    # 划分训练/验证池 (2021及以前) 和 终极盲测集 (2022及以后)
    # train_val_pool = df[df['year'] <= 2021]
    # test_df        = df[df['year'] >= 2022]

    # X_pool = train_val_pool[golden_features]
    # y_pool = train_val_pool[target] 
    # X_test_raw = test_df[golden_features].values
    # y_test = test_df[target].values

    X = df[golden_features]
    y = df[target]
    X_pool, X_test_raw, y_pool, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    # 2. 执行全自动参数优化 (Optuna)
    # 起步阶段建议 n_trials 设为 30-50 以保证速度
    best_params = optimize_rf(X_pool, y_pool, n_trials=100)
    
    # 💡 调参结束后，强制将最终模型的树数量设定为 1000 以获得最佳泛化性能
    max_trees = 1000
    best_params['n_estimators'] = max_trees
    
    # 保存最优参数
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ RF 最优参数已安全保存至: {PARAMS_JSON}")

    # 3. 终极盲测阶段与阶段性监控
    logger.info("🏁 进入终极评估阶段：执行热启动训练监控 (warm_start=True)...")

    # ⚠️ 最终训练前，对全体历史数据进行统一标准化
    final_scaler = StandardScaler()
    X_pool_scaled = final_scaler.fit_transform(X_pool.values)
    X_test_scaled = final_scaler.transform(X_test_raw)
    
    # 🌟 初始化具有“热启动”能力的模型
    # 剥离 n_estimators，因为我们将通过循环手动增加它
    rf_config = best_params.copy()
    rf_config.pop('n_estimators', None)
    
    final_model = RandomForestRegressor(
        **rf_config, 
        n_estimators=0,      # 从 0 开始
        warm_start=True,     # 开启热启动，允许在原有树的基础上继续生长
        n_jobs=-1, 
        random_state=42, 
        criterion='squared_error'
    )

    # 🌟 循环训练：每 50 棵树打印一次 Train/Test R2 进度
    step = 50
    logger.info(f"{'Trees':>6} | {'Train R2':>10} | {'Test R2':>10} | 状态")
    logger.info("-" * 50)

    for current_trees in range(step, max_trees + 1, step):
        final_model.n_estimators = current_trees
        final_model.fit(X_pool_scaled, y_pool.values)
        
        # 实时计算当前轮次的表现
        cur_pred_train = final_model.predict(X_pool_scaled)
        cur_pred_test  = final_model.predict(X_test_scaled)
        
        cur_train_r2 = r2_score(y_pool.values, cur_pred_train)
        cur_test_r2  = r2_score(y_test, cur_pred_test)
        
        logger.info(f"{current_trees:>6} | {cur_train_r2:>10.4f} | {cur_test_r2:>10.4f} | 训练中...")

    # 4. 最终指标汇总
    y_pred = final_model.predict(X_test_scaled)
    y_pred_train_final = final_model.predict(X_pool_scaled)

    train_r2_final = r2_score(y_pool.values, y_pred_train_final)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_bias = np.mean(y_pred - y_test)

    # --- 提取特征重要性 ---
    importance_df = pd.DataFrame({
        'Feature': golden_features,
        'Importance (Gini)': final_model.feature_importances_
    }).sort_values(by='Importance (Gini)', ascending=False)
    
    # 输出最终评估报告
    logger.info("="*30 + " RANDOM FOREST FINAL REPORT " + "="*30)
    logger.info(f"Train R²  : {train_r2_final:.4f}")
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    logger.info("-" * 25 + " 气象与卫星因子贡献度排名前 15 " + "-" * 25)
    
    for idx, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['Feature']:>22} : {row['Importance (Gini)']:.4f}")
        
    logger.info("="*88)

    # 5. 保存模型与标准化器
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(final_model, MODEL_SAVE_PATH)
    joblib.dump(final_scaler, SCALER_SAVE_PATH)
    logger.info(f"✅ 模型与标准化器已持久化至: {os.path.dirname(MODEL_SAVE_PATH)}")