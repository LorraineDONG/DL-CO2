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
# 版本更新记录：A02 -> A04 核心逻辑修正说明
# ==========================================
# 
# 1. 修复 n_estimators 与早停机制的冲突（防止严重过拟合）：
#    - [旧逻辑 A02] Optuna 直接搜索 n_estimators 并在有早停的情况下评估，但在最终集成训练时去掉了早停，导致模型跑满设定树深，强行记住训练集噪声。
#    - [新逻辑 A04] 从 Optuna 搜索空间中移除 n_estimators 并固定为一个极大值 (3000)。利用早停截断后，记录每一折真实的 best_iteration，取 3 折平均值后动态传给最终的集成模型。
# 
# 2. 修复 Optuna 内部的验证集泄漏（保证参数寻优的客观性）：
#    - [旧逻辑 A02] 验证集 X_va 被同时用作早停的 eval_set 和 Optuna 目标函数的打分集，导致模型对验证集“隐式调参”（又当裁判又当运动员）。
#    - [新逻辑 A04] 在 KFold 内部，从训练折再次切分 15% 作为绝对独立的早停监控集 (X_es)。X_va 彻底剥离，仅在模型训练结束后用于计算纯洁的 RMSE 反馈给 Optuna。
# 
# 3. 引入 Bootstrap 重采样（实现具有物理统计意义的不确定性量化）：
#    - [旧逻辑 A02] 10 个集成模型使用完全相同的全局训练集，仅仅修改随机种子。由于 XGBoost 极其稳定，导致预测结果高度同质化，计算出的不确定性（标准差）失真。
#    - [新逻辑 A04] 实现了真正的 Bagging（自举汇聚法）。在外部循环中加入 np.random.choice(replace=True) 进行有放回重采样。每个模型拟合的训练集分布产生微小差异，从而真实反映模型对未知样本的预测方差。
#
# * 注：按照对照实验需求，保留了原始全局 train_test_split 随机划分机制，未强制干预时空自相关性。
# ==========================================

# ==========================================
# 0. 全局配置与路径初始化 (更新为 A04)
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SHP_xgb_training(A07).log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SHP_optuna_xgb_study-A07.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/train_xgb_xco2en_SHP_best-A07.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A07.pkl' 
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A7.pkl' 
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A07.json'   

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
  
    # 筛选，去除极端值对模型的干扰
    df_clean = df_clean[df_clean[target] <= 10].reset_index(drop=True)
    df_clean['no2_trop_log'] = np.log(df_clean['no2_trop'])

    # 特征工程
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
# 3. Optuna 深度优化 (修复验证集泄漏与早停冲突)
# ==========================================
def optimize_xgb(X_pool, y_pool, n_trials=50):
    def objective(trial):
        param = {
            # 【修复 1】：不再让 Optuna 搜索 n_estimators，固定一个大值，靠 early_stopping 截断
            'n_estimators': 3000, 
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 3e-2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8), 
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7), 
            'subsample': trial.suggest_float('subsample', 0.4, 0.75),             
            'min_child_weight': trial.suggest_int('min_child_weight', 20, 150), 
            'gamma': trial.suggest_float('gamma', 0.1, 20.0, log=True), 
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 50.0, log=True), 
            'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 100.0, log=True), 
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42,
            'n_jobs': -1
        }

        kf = KFold(n_splits=3, shuffle=False)
        cv_rmses = []
        best_iterations = [] # 用于记录每一折跑出的最佳树数量
        
        for train_index, val_index in kf.split(X_pool):
            X_tr_raw_full, X_va_raw = X_pool[train_index], X_pool[val_index]
            y_tr_full, y_va = y_pool.iloc[train_index].values, y_pool.iloc[val_index].values

            # 【修复 2】：切分独立的早停监控集(X_es)，确保测试集(X_va)的绝对纯洁，防止隐式过拟合
            X_tr_raw, X_es_raw, y_tr, y_es = train_test_split(
                X_tr_raw_full, y_tr_full, test_size=0.15, random_state=42
            )

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_es = scaler.transform(X_es_raw)
            X_va = scaler.transform(X_va_raw)
            
            model = xgb.XGBRegressor(**param)
            model.fit(
                X_tr, y_tr, 
                eval_set=[(X_es, y_es)], # 使用独立的 early stopping 集合
                early_stopping_rounds=100, 
                verbose=False
            )
            
            # 使用纯洁的 X_va 计算得分，反馈给 Optuna
            preds = model.predict(X_va)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            best_iterations.append(model.best_iteration)
            
        # 【修复 1 延续】：将 3 折平均的最优树数量保存下来，留给外层提取
        mean_best_iteration = int(np.mean(best_iterations))
        trial.set_user_attr("best_n_estimators", mean_best_iteration)
        
        return np.mean(cv_rmses)

    logger.info("🚀 阶段二：开始 XGBoost Optuna 强正则化参数搜索...")
    study = optuna.create_study(
        direction='minimize', 
        storage=DB_FILE, 
        load_if_exists=True,
        study_name='xco2en_gpu_xgboost_A07' 
    )
    study.optimize(objective, n_trials=n_trials) 
    
    # 提取 Optuna 找到的最佳参数，并将最优树数量硬编码进去
    final_best_params = study.best_params
    final_best_params['n_estimators'] = study.best_trial.user_attrs["best_n_estimators"]
    
    return final_best_params

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    target = 'xco2_enhanced'

    initial_features = [
    'era5_blh', 'era5_blh_lag1', 'era5_blh_lead1', 
    'era5_d2m', 'era5_d2m_lag1', 'era5_d2m_lead1', 
    'era5_sp', 'era5_sp_lag1', 'era5_sp_lead1', 
    'era5_ssrd', 'era5_ssrd_lag1', 'era5_ssrd_lead1', 
    'era5_t2m', 'era5_t2m_lag1', 'era5_t2m_lead1', 
    'era5_tcwv', 'era5_tcwv_lag1', 'era5_tcwv_lead1', 
    'era5_u100', 'era5_u100_lag1', 'era5_u100_lead1', 
    'era5_u10', 'era5_u10_lag1', 'era5_u10_lead1', 
    'era5_v100', 'era5_v100_lag1', 'era5_v100_lead1', 
    'era5_v10', 'era5_v10_lag1', 'era5_v10_lead1', 
    'no2_trop_log','no2_trop_mean',
    'ndvi_t2m_cross','ssrd_t2m_cross','ntl_nox_cross',
    'meic_nox', 'ndvi', 'ntl', 'dem_mean',
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos', 
    'grid_lon', 'grid_lat', 
]
    
    # 1. 准备全局数据并严格切分 (遵循用户要求，保留原始的切分方式)
    df = load_and_preprocess(file_path)
    X_full_raw = df[initial_features].values
    y_full = df[target]
    
    X_pool_raw, X_test_raw, y_pool, y_test = train_test_split(
        X_full_raw, y_full, test_size=0.2, random_state=42
    )
 
    # 2. 执行 SHAP 特征筛选 (仅使用训练池数据 X_pool)
    temp_scaler = StandardScaler()
    X_pool_scaled_for_shap = temp_scaler.fit_transform(X_pool_raw) 

    selected_feature_names = perform_shap_feature_selection(
        X_pool_scaled_for_shap, y_pool, initial_features, top_n=25  
    )
    selected_indices = [initial_features.index(f) for f in selected_feature_names]
    
    X_pool_selected_raw = X_pool_raw[:, selected_indices]
    X_test_selected_raw = X_test_raw[:, selected_indices] 

    # 3. 执行参数寻优 (仅使用训练池数据 X_pool)
    best_params = optimize_xgb(X_pool_selected_raw, y_pool, n_trials=200) # 建议跑至少50次
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ XGB 最优参数 (包含动态锁定的 n_estimators: {best_params['n_estimators']}) 已保存至: {PARAMS_JSON}")

    # ==========================================
    # 4 & 5. 终极盲测评估与 10 个集成模型生成
    # ==========================================
    logger.info("🏁 阶段三：执行终极盲测集评估，并生成真正的 Bootstrap 集成模型...")
    
    eval_scaler = StandardScaler()
    X_pool_selected_scaled = eval_scaler.fit_transform(X_pool_selected_raw)
    X_test_selected_scaled = eval_scaler.transform(X_test_selected_raw)
    
    joblib.dump(eval_scaler, SCALER_SAVE_PATH)
    logger.info(f"✅ 标准化器 (Scaler) 已成功保存至: {SCALER_SAVE_PATH}")

    ensemble_seeds = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]
    
    logger.info("="*30 + " XGBOOST ENSEMBLE REPORT " + "="*30)
    
    # 将 y_pool 统一转为 numpy array 以方便后续的下标索引
    y_pool_arr = y_pool.values if isinstance(y_pool, pd.Series) else y_pool

    for i, seed in enumerate(ensemble_seeds):
        logger.info(f"⚙️ 正在训练第 {i+1}/10 个模型 (Seed: {seed})...")
        
        # 【修复 3】：实现 Bootstrap 重采样，让模型具有物理意义的方差
        np.random.seed(seed)
        n_samples = X_pool_selected_scaled.shape[0]
        # 有放回重采样
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True) 
        
        X_boot = X_pool_selected_scaled[boot_indices]
        y_boot = y_pool_arr[boot_indices]

        model_params = best_params.copy()
        model_params['random_state'] = seed
        model_params['n_jobs'] = -1
        
        # 训练当前种子模型 (不再使用 Early Stopping，直接跑满最优树数量)
        eval_model = xgb.XGBRegressor(**model_params)
        eval_model.fit(X_boot, y_boot, verbose=False)
        
        # 对全部训练池(X_pool)和测试集进行评估
        train_preds = eval_model.predict(X_pool_selected_scaled)
        test_preds = eval_model.predict(X_test_selected_scaled)

        train_r2 = r2_score(y_pool, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
        logger.info(f"   📊 [Seed {seed:^4}] Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")
        
        base_path, ext = os.path.splitext(MODEL_SAVE_PATH)
        current_model_path = f"{base_path}_seed{seed}{ext}"
        
        joblib.dump(eval_model, current_model_path)
        
    logger.info("="*85)
    logger.info("💾 10 个 Bootstrap 集成模型已全部生成并保存完毕，准备用于真实的不确定性量化！")
    logger.info("🎉 全部训练与评估流程圆满结束！")