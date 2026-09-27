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
from sklearn.utils import resample

# ==============================================================================
# 版本演进说明 (Version Evolution Notes): 当前版本 vs A01
# ==============================================================================
# 1. 核心科学问题 (Scientific Rationale):
#    在初始版本中，集成模型仅通过扰动 XGBoost 的 'random_state' (随机种子) 来生成。
#    由于 Optuna 进行了强正则化约束（限制了最大树深并显著提升了叶子节点最小样本权重），
#    导致各子模型在结构和决策边界上高度同质化（Homogenized Decision Boundaries）。
#    这种同质化导致模型“过度自信” (Overconfident)，原生输出的预测标准差 (1σ) 无法真实
#    反映模型对未知观测的认知不确定性 (Epistemic Uncertainty)，在盲测集上表现为预测区间
#    覆盖率 (PICP) 严重偏低 (仅 ~15.7%)。
#
# 2. 当前版本的改进机制 (Substantive Methodology Improvement):
#    当前版本引入了基于统计学加法公理的真正的 Bagging (Bootstrap 自助重采样) 机制：
#    - 数据扰动 (Data Perturbation): 在训练循环中，利用 `sklearn.utils.resample` 对
#      训练池进行有放回的随机抽样（采样率 100%，含重复样本）。每次迭代各子模型看到的
#      数据流略有不同（独特样本比例约为 63.2%）。
#    - 拓宽物理方差 (Expanding Physical Variance): 这种机制强制拉开了不同随机种子模型
#      在面对高排放热点 (Hotspots) 或复杂气象/地表交叉特征（如 ntl_nox_cross, ndvi_t2m_cross）
#      时的预测分歧，从而在空间和时间尺度上物理性地撑开了集成标准差 (Ensemble STD)。
#
# 3. 预期学术价值 (Academic & Publication Value for RSE/ES&T):
#    - 显著提升了不确定性量化 (UQ) 的稳健性，使得原生预测标准差能够真正作为实际预测误差的
#      可靠代理 (Reliable Proxy for Absolute Prediction Error)，大幅拉升 95% 置信区间覆盖率 (PICP)。
#    - 避免了纯后处理校准 (Post-hoc Calibration) 带来的物理意义缺失，在空间制图 (Spatial Mapping) 
#      中能够更真实地反映复杂环境强迫下的像素级置信度评估。
# ==============================================================================


# ==========================================
# 0. 全局配置与路径初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SHP_xgb_training(A06).log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SHP_optuna_xgb_study-A06.db' 
PARAMS_JSON = '/home/whdong/dl/best_params/train_xgb_xco2en_SHP_best-A06.json'
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A06.pkl' 
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A6.pkl' 
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A06.json'   

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
    # A01不开启<=10的样本，数据集里的我全都要
    # df_clean = df_clean[df_clean[target] <= 10].reset_index(drop=True)
    # logger.info(f"🧹 经过筛选 (xco2_enhanced <= 10) 后，当前样本量为: {len(df_clean)}")

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
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100), 
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 3e-2, log=True),
            # 严格限制树深！XGBoost 超过 8 极易过拟合
            'max_depth': trial.suggest_int('max_depth', 3, 8), 
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7), # 增加随机性
            'subsample': trial.suggest_float('subsample', 0.4, 0.75),             # 增加随机性
            # 显著提升叶子节点最小样本权重，强制限制分裂
            'min_child_weight': trial.suggest_int('min_child_weight', 20, 150), 
            # 拓宽 gamma 边界，更激进地剪枝
            'gamma': trial.suggest_float('gamma', 0.1, 20.0, log=True), 
            # 继续保持强 L1/L2 正则
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 50.0, log=True), 
            'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 100.0, log=True), 
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42,
            'n_jobs': -1
        }

        # 寻优阶段用 3-Fold
        kf = KFold(n_splits=3, shuffle=False)
        cv_rmses = []
        
        for train_index, val_index in kf.split(X_pool):
            X_tr_raw, X_va_raw = X_pool[train_index], X_pool[val_index]
            y_tr, y_va = y_pool.iloc[train_index].values, y_pool.iloc[val_index].values

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_va = scaler.transform(X_va_raw)
            
            model = xgb.XGBRegressor(**param)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, verbose=False)
            
            preds = model.predict(X_va)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            
        return np.mean(cv_rmses)

    logger.info("🚀 阶段二：开始 XGBoost Optuna 强正则化参数搜索...")
    study = optuna.create_study(
        direction='minimize', 
        storage=DB_FILE, 
        load_if_exists=True,
        study_name='xco2en_gpu_xgboost_A06' # 启用全新命名，隔离先前被污染的超参数历史
    )
    study.optimize(objective, n_trials=n_trials) 
    return study.best_params

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    target = 'xco2_enhanced'

    initial_features = [
    'era5_blh', 
    'era5_blh_lag1',# 'era5_blh_lag2', 'era5_blh_lag3', 
    'era5_blh_lead1', #'era5_blh_lead2', 'era5_blh_lead3', 
    'era5_d2m', 
    'era5_d2m_lag1', #'era5_d2m_lag2', 'era5_d2m_lag3', 
    'era5_d2m_lead1', #'era5_d2m_lead2', 'era5_d2m_lead3', 
    'era5_sp',  
    'era5_sp_lag1', #'era5_sp_lag2', 'era5_sp_lag3', 
    'era5_sp_lead1', #'era5_sp_lead2', 'era5_sp_lead3', 
    'era5_ssrd',
    'era5_ssrd_lag1',# 'era5_ssrd_lag2', 'era5_ssrd_lag3', 
    'era5_ssrd_lead1', #'era5_ssrd_lead2', 'era5_ssrd_lead3', 
    'era5_t2m', 
    'era5_t2m_lag1',# 'era5_t2m_lag2', 'era5_t2m_lag3', 
    'era5_t2m_lead1', #'era5_t2m_lead2', 'era5_t2m_lead3', 
    'era5_tcwv',
    'era5_tcwv_lag1', #'era5_tcwv_lag2', 'era5_tcwv_lag3', 
    'era5_tcwv_lead1',# 'era5_tcwv_lead2', 'era5_tcwv_lead3', 
    'era5_u100', 
    'era5_u100_lag1', #'era5_u100_lag2', 'era5_u100_lag3', 
    'era5_u100_lead1',# 'era5_u100_lead2', 'era5_u100_lead3', 
    'era5_u10', 
    'era5_u10_lag1',# 'era5_u10_lag2', 'era5_u10_lag3', 
    'era5_u10_lead1', #'era5_u10_lead2', 'era5_u10_lead3', 
    'era5_v100', 
    'era5_v100_lag1', #'era5_v100_lag2', 'era5_v100_lag3',
    'era5_v100_lead1', #'era5_v100_lead2', 'era5_v100_lead3', 
    'era5_v10', 
    'era5_v10_lag1', #'era5_v10_lag2', 'era5_v10_lag3', 
    'era5_v10_lead1',# 'era5_v10_lead2', 'era5_v10_lead3', 
    'no2_trop_log','no2_trop_mean',
    'ndvi_t2m_cross','ssrd_t2m_cross','ntl_nox_cross',
    'meic_nox', 'ndvi', 'ntl', 'dem_mean',
    #'no2_amf_trop', 'no2_variance', 'ndvi_std', #'dem_std',
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos', 
    'grid_lon', 'grid_lat', 
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
        X_pool_scaled_for_shap, y_pool, initial_features, top_n=20 # A01 只严格保留了贡献度最高的 20 个特征
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
    # 4 & 5. 终极盲测评估与 10 个集成模型生成
    # ==========================================
    logger.info("🏁 阶段三：执行终极盲测集评估，并生成 10 个集成模型...")
    
    # 严格的标准化逻辑：fit训练池，transform测试集 (标准化器全局只需一个)
    eval_scaler = StandardScaler()
    X_pool_selected_scaled = eval_scaler.fit_transform(X_pool_selected_raw)
    X_test_selected_scaled = eval_scaler.transform(X_test_selected_raw)
    
    # 保存全局唯一的标准化器
    joblib.dump(eval_scaler, SCALER_SAVE_PATH)
    logger.info(f"✅ 标准化器 (Scaler) 已成功保存至: {SCALER_SAVE_PATH}")

    # 定义 10 个不同的随机种子
    ensemble_seeds = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]
    
    logger.info("="*30 + " XGBOOST ENSEMBLE REPORT " + "="*30)
    
    for i, seed in enumerate(ensemble_seeds):
        logger.info(f"⚙️ 正在训练第 {i+1}/10 个模型 (Seed: {seed})...")
        
        # 【核心修改】：实施真正的 Bagging (Bootstrap 有放回抽样)
        # 利用当前的随机种子进行抽样，保证实验可复现
        X_boot, y_boot = resample(
            X_pool_selected_scaled, y_pool, 
            replace=True, 
            n_samples=len(y_pool), 
            random_state=seed
        )
        
        # 使用 Optuna 找到的最佳参数，但替换随机种子
        model_params = best_params.copy()
        model_params['random_state'] = seed
        model_params['n_jobs'] = -1
        
        # 训练当前种子模型 
        # 注意：这里喂给模型的是刚刚抽样得到的 X_boot 和 y_boot
        eval_model = xgb.XGBRegressor(**model_params)
        eval_model.fit(X_boot, y_boot, verbose=False)
        
        # 预测并评估 
        # 注意：评估时依然使用完整的训练池和测试集，以保证 R2 和 RMSE 计算的公平性
        train_preds = eval_model.predict(X_pool_selected_scaled)
        test_preds = eval_model.predict(X_test_selected_scaled)

        train_r2 = r2_score(y_pool, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
        logger.info(f"   📊 [Seed {seed:^4}] Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")
        
        # 动态生成模型保存路径
        base_path, ext = os.path.splitext(MODEL_SAVE_PATH)
        current_model_path = f"{base_path}_seed{seed}{ext}"
        
        # 保存当前模型
        joblib.dump(eval_model, current_model_path)
        
    logger.info("="*85)
    logger.info("💾 10 个集成模型已全部生成并保存完毕，准备用于应用期推理与不确定性量化！")
    logger.info("🎉 全部训练与评估流程圆满结束！")