import pandas as pd
import numpy as np
import os
import json
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ==========================================
# 0. 路径配置 (与你的训练脚本保持一致)
# ==========================================
DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
TARGET = 'xco2_enhanced'

# XGBoost 路径
XGB_PARAMS_JSON = '/home/whdong/dl/best_params/train_xgb_xco2en_SHP_best-A01.json'
XGB_FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'
XGB_SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
XGB_MODEL_BASE = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01'

# Random Forest 路径
RF_FEATURES_JSON = '/home/whdong/dl/best_params/selected_features_rf_baseline-A01.json'
RF_SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-rf_baseline_scaler-A01.pkl'
RF_MODEL_BASE = '/home/whdong/dl/models/XCO2en_SHP-rf_baseline_model-A01'

OUTPUT_CSV = '/home/whdong/dl/ML-prediction-output_result/A01/figures/Spatiotemporal_Validation_Table.csv'

ENSEMBLE_SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]

# ==========================================
# 1. 还原数据预处理与严格测试集切分
# ==========================================
def load_and_preprocess(file_path):
    print(f"📂 加载原始数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    
    df_clean['no2_trop_log'] = np.log(df_clean['no2_trop'])
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)
    
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    return df_clean

df_clean = load_and_preprocess(DATA_FILE)

# 【核心】：直接对整个 DataFrame 进行 train_test_split，完美保留辅助列(lon, lat, month)
df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
y_test = df_test[TARGET].values

# ==========================================
# 2. 计算 XGBoost 集成预测值
# ==========================================
print("🚀 正在计算 10-XGBoost Ensemble 预测值...")
with open(XGB_FEATURES_JSON, 'r') as f: xgb_features = json.load(f)
xgb_scaler = joblib.load(XGB_SCALER_PATH)

X_test_xgb_raw = df_test[xgb_features].values
X_test_xgb = xgb_scaler.transform(X_test_xgb_raw)

xgb_preds_all = []
for seed in ENSEMBLE_SEEDS:
    model_path = f"{XGB_MODEL_BASE}_seed{seed}.pkl"
    model = joblib.load(model_path)
    xgb_preds_all.append(model.predict(X_test_xgb))
# 均值作为集成最终预测
pred_xgb = np.mean(xgb_preds_all, axis=0)

# ==========================================
# 3. 计算 Random Forest 集成预测值
# ==========================================
print("🌲 正在计算 10-RF Baseline 预测值...")
with open(RF_FEATURES_JSON, 'r') as f: rf_features = json.load(f)
rf_scaler = joblib.load(RF_SCALER_PATH)

X_test_rf_raw = df_test[rf_features].values
X_test_rf = rf_scaler.transform(X_test_rf_raw)

rf_preds_all = []
for seed in ENSEMBLE_SEEDS:
    model_path = f"{RF_MODEL_BASE}_seed{seed}.pkl"
    model = joblib.load(model_path)
    rf_preds_all.append(model.predict(X_test_rf))
# 均值作为集成最终预测
pred_rf = np.mean(rf_preds_all, axis=0)

# ==========================================
# 4. 训练一个多元线性回归 (MLR) 作为 Baseline 2
# ==========================================
print("📈 正在即时训练 MLR Baseline...")
# 我们使用 XGB 筛选出的重要特征来训练线性模型
X_train_xgb_raw = df_train[xgb_features].values
X_train_xgb = xgb_scaler.transform(X_train_xgb_raw)
y_train = df_train[TARGET].values

mlr = LinearRegression()
mlr.fit(X_train_xgb, y_train)
pred_mlr = mlr.predict(X_test_xgb)

# ==========================================
# 5. 定义时空维度的切分 Mask
# ==========================================
# 注意：你需要根据你的实际研究边界微调以下的 lat/lon 范围！
masks = {
    "Global Test Set": pd.Series(True, index=df_test.index),
    
    # 季节切分
    "Spring (MAM)": df_test['month'].isin([3, 4, 5]),
    "Summer (JJA)": df_test['month'].isin([6, 7, 8]),
    "Autumn (SON)": df_test['month'].isin([9, 10, 11]),
    "Winter (DJF)": df_test['month'].isin([12, 1, 2]),
    
    # 空间区域切分 (请根据实际情况修改经纬度框)
    "North China Plain": (df_test['grid_lat'] >= 34.0) & (df_test['grid_lat'] <= 41.0) & 
                         (df_test['grid_lon'] >= 113.0) & (df_test['grid_lon'] <= 120.0),
                         
    "Yangtze River Delta": (df_test['grid_lat'] >= 29.0) & (df_test['grid_lat'] <= 34.0) & 
                           (df_test['grid_lon'] >= 118.0) & (df_test['grid_lon'] <= 123.0),
                           
    "Pearl River Delta": (df_test['grid_lat'] >= 21.0) & (df_test['grid_lat'] <= 25.0) & 
                         (df_test['grid_lon'] >= 112.0) & (df_test['grid_lon'] <= 115.0),
}

# 增加一个 Background / Others
masks["Others (Background)"] = ~(masks["North China Plain"] | masks["Yangtze River Delta"] | masks["Pearl River Delta"])

# ==========================================
# 6. 计算指标并生成表格
# ==========================================
print("📊 正在计算验证统计指标...")

def calc_metrics(y_true, y_pred):
    if len(y_true) < 5: # 样本太少不计算
        return np.nan, np.nan, np.nan, np.nan
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true) # Bias 定义为 Pred - True
    return r2, rmse, mae, bias

results = []

for dimension_name, mask in masks.items():
    idx = mask.values
    y_true_sub = y_test[idx]
    n_samples = len(y_true_sub)
    
    # XGB 指标
    xgb_r2, xgb_rmse, xgb_mae, xgb_bias = calc_metrics(y_true_sub, pred_xgb[idx])
    # RF 指标
    rf_r2, rf_rmse, rf_mae, rf_bias = calc_metrics(y_true_sub, pred_rf[idx])
    # MLR 指标
    mlr_r2, mlr_rmse, mlr_mae, mlr_bias = calc_metrics(y_true_sub, pred_mlr[idx])
    
    results.append({
        "Validation Domain": dimension_name,
        "N (Samples)": n_samples,
        
        "10-XGBoost R2": xgb_r2,
        "10-XGBoost RMSE": xgb_rmse,
        "10-XGBoost MAE": xgb_mae,
        "10-XGBoost Bias": xgb_bias,
        
        "10-RF R2": rf_r2,
        "10-RF RMSE": rf_rmse,
        "10-RF MAE": rf_mae,
        "10-RF Bias": rf_bias,
        
        "MLR R2": mlr_r2,
        "MLR RMSE": mlr_rmse,
        "MLR MAE": mlr_mae,
        "MLR Bias": mlr_bias
    })

# 构建 DataFrame 并格式化小数位数
df_results = pd.DataFrame(results)
cols_to_round = df_results.columns.drop(["Validation Domain", "N (Samples)"])
df_results[cols_to_round] = df_results[cols_to_round].round(3)

# 导出 CSV
df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"==================================================")
print(f"🎉 验证统计表格计算完成！")
print(f"💾 CSV 已保存至: {OUTPUT_CSV}")
print(df_results[["Validation Domain", "N (Samples)", "10-XGBoost RMSE", "10-RF RMSE"]].head(5))
print(f"==================================================")