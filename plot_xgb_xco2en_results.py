import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# 尝试导入地图库
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️ 未安装 Cartopy，空间地图将以普通散点图形式呈现。")

# ==========================================
# 1. 路径与配置加载 (与训练脚本对齐)
# ==========================================
DATA_PATH = '/home/whdong/dl/xco2_sif_no2_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
PARAMS_JSON = '/home/whdong/dl/best_params/train_xgb_xco2en_SHP_10fold_best.json'
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features_10fold.json'
MODEL_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_10fold_model.pkl' 
SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_10fold_scaler.pkl' 
FIG_SAVE_DIR = '/home/whdong/dl/figures/'
os.makedirs(FIG_SAVE_DIR, exist_ok=True)

# 必须与训练脚本中完全一致的 32 个初始特征
INITIAL_FEATURES = [
    'era5_blh', 'era5_d2m', 'era5_sp', 'era5_ssrd', 'era5_t2m', 'era5_tcwv', 
    'era5_u100', 'era5_v100', 'era5_u10', 'era5_v10', 'era5_wind_speed',
    'grid_lat', 'grid_lon', 'dem_mean', 'dem_std', 
    'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    'ndvi_t2m_cross', 'ssrd_t2m_cross', 'ntl_nox_cross',
    'meic_nox', 'ntl', 'ndvi', 'ndvi_std', 'sif_740', 'sif_variance',
    'no2_amf_trop', 'no2_trop', 'no2_variance', 'no2_trop_log'
]

def load_data_and_tools():
    with open(FEATURES_JSON, 'r') as f:
        selected_features = json.load(f)
    with open(PARAMS_JSON, 'r') as f:
        best_params = json.load(f)
    
    saved_bundle = joblib.load(MODEL_PATH)
    production_model = saved_bundle['model']
    scaler = joblib.load(SCALER_PATH)
    
    df = pd.read_pickle(DATA_PATH).dropna()
    df['date'] = pd.to_datetime(df['date'])
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12.0)
    df['doy_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)
    df['ndvi_t2m_cross'] = df['ndvi'] * df['era5_t2m']
    df['ssrd_t2m_cross'] = df['era5_ssrd'] * df['era5_t2m']
    df['ntl_nox_cross'] = df['ntl'] * df['meic_nox']
    df['era5_wind_speed'] = np.sqrt(df['era5_u100']**2 + df['era5_v100']**2)
    df['no2_trop_log'] = np.log1p(np.maximum(df['no2_trop'], 0))
    # df['no2_co_cross'] = df['no2_trop'] / (df['co'] + 1e-8)
    
    return df, selected_features, best_params, production_model, scaler

# ==========================================
# 2. 重新生成 OOF 预测
# ==========================================
def get_oof_predictions(X_raw, y, params):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    
    print("🔄 正在重新计算 10-Fold OOF 预测以获取评估指标 (严谨的内部归一化)...")
    for train_idx, test_idx in kf.split(X_raw):
        # 1. 划分原始数据
        X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # 2. 在折内部进行标准化
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
        
        # 删除了多余的 tree_method，避免报错
        model = xgb.XGBRegressor(**params, n_jobs=-1, random_state=42)
        model.fit(X_tr, y_tr)
        oof_preds[test_idx] = model.predict(X_te)
        
    return oof_preds, y

# ==========================================
# 3. 绘图函数
# ==========================================
def plot_all(df, y_true, y_pred, features, model, X_scaled):
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    print("🎨 绘图中: 密度散点图...")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    idx = np.random.choice(len(y_true), 50000, replace=False) if len(y_true)>50000 else np.arange(len(y_true))
    xt, yp = y_true[idx], y_pred[idx]
    xy = np.vstack([xt, yp])
    z = gaussian_kde(xy)(xy)
    
    sc = ax.scatter(xt, yp, c=z, s=5, cmap='viridis')
    ax.plot([xt.min(), xt.max()], [xt.min(), xt.max()], 'k--', label='1:1 Line')
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_title(f'OOF Evaluation (R²={r2:.3f}, RMSE={rmse:.3f})')
    plt.colorbar(sc, label='Density')
    plt.savefig(f"{FIG_SAVE_DIR}density_scatter.png")
    
    print("🎨 绘图中: SHAP 解释图...")
    explainer = shap.TreeExplainer(model)
    X_sample = X_scaled[:5000] if len(X_scaled)>5000 else X_scaled
    shap_v = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_v, X_sample, feature_names=features, show=False)
    plt.savefig(f"{FIG_SAVE_DIR}shap_summary.png")

    print("🎨 绘图中: 空间误差地图...")
    df_err = pd.DataFrame({'lon': df['grid_lon'], 'lat': df['grid_lat'], 'err': y_pred - y_true})
    grid_err = df_err.groupby(['lon', 'lat'])['err'].mean().reset_index()

    if HAS_CARTOPY:
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        sc = ax.scatter(grid_err['lon'], grid_err['lat'], c=grid_err['err'], 
                        cmap='RdBu_r', vmin=-2, vmax=2, s=10, transform=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(grid_err['lon'], grid_err['lat'], c=grid_err['err'], cmap='RdBu_r', vmin=-2, vmax=2, s=10)
    
    plt.colorbar(sc, label='Bias (Pred - Obs) [ppm]')
    plt.title('Spatial Distribution of Residuals')
    plt.savefig(f"{FIG_SAVE_DIR}spatial_bias_map.png")

if __name__ == "__main__":
    df, selected_features, params, model, loaded_scaler = load_data_and_tools()
    
    # 1. 提取未缩放的全部原始特征
    X_initial_raw = df[INITIAL_FEATURES].values
    
    # 2. 找到 SHAP 筛选出的那 20 个特征的索引
    selected_indices = [INITIAL_FEATURES.index(f) for f in selected_features]
    
    # 3. 切片获取这 20 个用于训练的原始特征矩阵
    X_selected_raw = X_initial_raw[:, selected_indices]
    
    y_true_all = df['xco2_enhanced'].values
    
    # 4. 获取 OOF 预测结果（传入这 20 个未缩放的数据，内部会做标准的 10Fold 分布缩放）
    y_pred, y_true = get_oof_predictions(X_selected_raw, y_true_all, params)
    
    # 5. 【核心修复】使用加载的生产级 Scaler，去缩放那 20 个特征 (而不是 32 个)
    X_all_selected_scaled = loaded_scaler.transform(X_selected_raw)
    
    # 6. 开始绘图
    plot_all(df, y_true_all, y_pred, selected_features, model, X_all_selected_scaled)
    print(f"✅ 绘图完成！图片已保存至: {FIG_SAVE_DIR}")