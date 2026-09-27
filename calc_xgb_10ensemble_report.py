import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# 0. 日志配置
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. 全局配置与路径初始化
# ==========================================
DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
target = 'xco2_enhanced'

FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'
SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
MODEL_BASE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01.pkl'

ensemble_seeds = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]

# ==========================================
# 2. 数据加载与特征工程 (补全缺失的逻辑)
# ==========================================
def load_and_preprocess(file_path):
    logger.info(f"📂 正在加载并加工数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    
    # 动态构建特征工程 (A01, A05需要注释下一行; 其他模型需要打开)
    # df_clean = df_clean[df_clean[target] <= 10].reset_index(drop=True)
    df_clean['no2_trop_log'] = np.log(df_clean['no2_trop'])
    
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

# 调用函数，拿到加工好的 DataFrame
df_clean = load_and_preprocess(DATA_FILE)

# ==========================================
# 3. 加载特征列表、划分数据集与缩放
# ==========================================
logger.info("📂 正在加载选定的特征列表...")
with open(FEATURES_JSON, 'r', encoding='utf-8') as f:
    selected_features = json.load(f)

# 【重点修改】：提取“全量数据”
X_raw_full = df_clean[selected_features].values
y_full = df_clean[target].values

# 【重点修改】：用一模一样的 random_state=42，严格剥离出 20% 测试集
_, X_test_raw, _, y_test = train_test_split(X_raw_full, y_full, test_size=0.2, random_state=42)

logger.info("📐 正在加载 StandardScaler 并进行数据缩放...")
scaler = joblib.load(SCALER_PATH)

# 分别对【全量数据】和【测试集数据】进行 transform
X_scaled_full = scaler.transform(X_raw_full)
X_scaled_test = scaler.transform(X_test_raw)

# ==========================================
# 4. 加载 10 个 XGBoost 模型并执行独立预测
# ==========================================
# 准备两个矩阵，分别存储全量预测和测试集预测
all_preds_full = np.zeros((len(ensemble_seeds), len(y_full)))
all_preds_test = np.zeros((len(ensemble_seeds), len(y_test)))

logger.info(f"🚀 开始加载并执行 {len(ensemble_seeds)} 个集成模型的推理...")
for i, seed in enumerate(ensemble_seeds):
    base_path, ext = os.path.splitext(MODEL_BASE_PATH)
    current_model_path = f"{base_path}_seed{seed}{ext}"
    
    if not os.path.exists(current_model_path):
        logger.error(f"❌ 找不到模型文件 {current_model_path}，请检查路径！")
        continue
        
    model = joblib.load(current_model_path) 
    
    # 模型同时对两波数据进行打分
    all_preds_full[i, :] = model.predict(X_scaled_full)
    all_preds_test[i, :] = model.predict(X_scaled_test)
    
    logger.info(f"  ✅ 模型 [Seed {seed:^4}] 推理完成.")

# ==========================================
# 5. 计算集成均值与不确定性 (Uncertainty)
# ==========================================
logger.info("📊 计算集合预测均值与不确定性 (Standard Deviation)...")

# 计算全量数据的均值和标准差 (用于拼回原始 DataFrame 制图)
ensemble_mean_full = np.mean(all_preds_full, axis=0) 
ensemble_std_full = np.std(all_preds_full, axis=0) 

df_clean['predicted_mean'] = ensemble_mean_full
df_clean['uncertainty_std'] = ensemble_std_full

# 计算测试集数据的均值和标准差 (用于报告真实的泛化误差)
ensemble_mean_test = np.mean(all_preds_test, axis=0)
ensemble_std_test = np.std(all_preds_test, axis=0)

# ==========================================
# 6. 输出终极评估报告 (双路对比)
# ==========================================
# --- 统计全量指标 ---
full_r2 = r2_score(y_full, ensemble_mean_full)
full_rmse = np.sqrt(mean_squared_error(y_full, ensemble_mean_full))
full_mae = mean_absolute_error(y_full, ensemble_mean_full)

# --- 统计测试集指标 ---
test_r2 = r2_score(y_test, ensemble_mean_test)
test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_mean_test))
test_mae = mean_absolute_error(y_test, ensemble_mean_test)

logger.info("=" * 55)
logger.info("          XGBOOST ENSEMBLE FINAL REPORT          ")
logger.info("=" * 55)

logger.info("【1】全量数据评估 (含训练集 - 仅作空间分布制图参考)：")
logger.info(f" -> 伪 R² (Full R²)  : {full_r2:.4f}")
logger.info(f" -> Full RMSE        : {full_rmse:.4f} ppm")
logger.info(f" -> Full MAE         : {full_mae:.4f} ppm")
logger.info(f" -> 全局平均不确定性 : {np.mean(ensemble_std_full):.4f} ppm")
logger.info("-" * 55)

logger.info("【2】严苛测试集评估 (20% Unseen Data - 真实模型性能)：")
logger.info(f" -> 真 R² (Test R²)  : {test_r2:.4f}")
logger.info(f" -> Test RMSE        : {test_rmse:.4f} ppm")
logger.info(f" -> Test MAE         : {test_mae:.4f} ppm")
logger.info(f" -> 测试集平均不确定性: {np.mean(ensemble_std_test):.4f} ppm")
logger.info("=" * 55)