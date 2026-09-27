# -*- coding: utf-8 -*-
"""
fig13.py — SHAP Variance Decomposition (Standalone Panels)

源自 fig13.py，按独立输出要求重构：
  - 输出1: 特征重要度 vs 模型分歧度 (原 panel b)   → FIG13-Importance_vs_Disagreement.png
  - 输出2: 主导不确定性来源空间分布 (原 panel c)    → FIG13-Dominant_Uncertainty_Source.png
  - 输出3: Top-8 特征 SHAP 方差剖面 (4x2 子图)     → FIG13-Feature_SHAP_Profiles.png

所有输出：不显示主标题、不显示图编号。
8 子图面板按 (a)-(h) 标注。
"""

import os, json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import shap
import cartopy.crs as ccrs, cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import shape

# ==========================================
# 0. 全局样式
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.unicode_minus': False,
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'axes.linewidth': 1.2, 'lines.linewidth': 1.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.top': True, 'ytick.right': True,
    'axes.grid': True, 'grid.linestyle': '--', 'grid.linewidth': 0.6,
    'grid.alpha': 0.4, 'grid.color': '#B0B0B0',
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05
})

# ==========================================
# 1. 路径 & 辅助
# ==========================================
DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'
SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
MODEL_BASE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01'
OUTPUT_DIR = '/home/whdong/dl/figures/scatter_density_A01'
WORLD_SHP = '/home/whdong/shapefile/world/world.shp'
CHINA_PROV_SHP = '/home/whdong/shapefile/china/province.shp'
TARGET = 'xco2_enhanced'
SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]
os.makedirs(OUTPUT_DIR, exist_ok=True)

def feature_display_name(name):
    m = {'no2_trop_mean':'NO2 Mean','no2_trop_log':'Log(NO2)','era5_blh':'PBLH',
         'era5_blh_lag1':'PBLH Lag1','era5_blh_lead1':'PBLH Lead1',
         'era5_d2m':'Dewpt T2m','era5_d2m_lag1':'Dewpt T2m Lag1',
         'era5_d2m_lead1':'Dewpt T2m Lead1','era5_sp':'Surface Press',
         'era5_sp_lag1':'SP Lag1','era5_sp_lead1':'SP Lead1',
         'era5_ssrd':'SSRD','era5_ssrd_lag1':'SSRD Lag1','era5_ssrd_lead1':'SSRD Lead1',
         'era5_t2m':'T2m','era5_t2m_lag1':'T2m Lag1','era5_t2m_lead1':'T2m Lead1',
         'era5_tcwv':'TCWV','era5_tcwv_lag1':'TCWV Lag1','era5_tcwv_lead1':'TCWV Lead1',
         'era5_u100':'U100','era5_u100_lag1':'U100 Lag1','era5_u100_lead1':'U100 Lead1',
         'era5_u10':'U10','era5_u10_lag1':'U10 Lag1','era5_u10_lead1':'U10 Lead1',
         'era5_v100':'V100','era5_v100_lag1':'V100 Lag1','era5_v100_lead1':'V100 Lead1',
         'era5_v10':'V10','era5_v10_lag1':'V10 Lag1','era5_v10_lead1':'V10 Lead1',
         'meic_nox':'MEIC NOx','ndvi':'NDVI','ntl':'NTL','dem_mean':'Elevation',
         'ndvi_t2m_cross':'NDVI x T2m','ssrd_t2m_cross':'SSRD x T2m','ntl_nox_cross':'NTL x NOx',
         'doy_sin':'DOY sin','doy_cos':'DOY cos','month_sin':'Month sin','month_cos':'Month cos',
         'grid_lon':'Longitude','grid_lat':'Latitude'}
    return m.get(name, name)

def feature_units(name):
    if 'no2' in name: return '[$10^{16}$ mol/cm$^2$]'
    if 'temp' in name or 't2m' in name or 'd2m' in name: return '[K]'
    if 'blh' in name: return '[m]'
    if 'sp' in name: return '[hPa]'
    if 'ssrd' in name: return '[MJ/m$^2$]'
    if 'wind' in name or 'u10' in name or 'u100' in name or 'v10' in name or 'v100' in name: return '[m/s]'
    if 'tcwv' in name: return '[kg/m$^2$]'
    if 'ndvi' in name or 'doy' in name or 'month' in name: return ''
    if 'dem' in name: return '[m]'
    if 'lon' in name or 'lat' in name: return '[deg]'
    if 'meic' in name: return '[mol/m$^2$/s]'
    if 'ntl' in name and 'cross' not in name: return '[W/m$^2$/sr]'
    return ''

def add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333')
    if os.path.exists(WORLD_SHP):
        try:
            ax.add_feature(cfeature.ShapelyFeature(
                shpreader.Reader(WORLD_SHP).geometries(), ccrs.PlateCarree(),
                facecolor='none', edgecolor='#333333', linewidth=0.6, linestyle=':'))
        except: ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=':')
    if os.path.exists(CHINA_PROV_SHP):
        try:
            import shapefile as pyshp
            sf = pyshp.Reader(CHINA_PROV_SHP, encoding='gbk')
            provinces = cfeature.ShapelyFeature(
                [shape(s.__geo_interface__) for s in sf.shapes()],
                ccrs.PlateCarree(), facecolor='none', edgecolor='#777777',
                linewidth=0.3, linestyle=':')
            ax.add_feature(provinces)
        except: pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray',
                      alpha=0.4, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 9}; gl.ylabel_style = {'size': 9}


# ==========================================
# 2. 输出1: 重要度 vs 分歧度 (单独)
# ==========================================
def save_fig1(shap_values_all, shap_std, feature_names, save_path):
    mean_abs_shap = np.mean(np.abs(shap_values_all), axis=(0, 1))
    mean_shap_std = np.mean(shap_std, axis=0)
    eps = 1e-10
    imp = (mean_abs_shap - mean_abs_shap.min()) / (mean_abs_shap.max() - mean_abs_shap.min() + eps)
    dis = (mean_shap_std - mean_shap_std.min()) / (mean_shap_std.max() - mean_shap_std.min() + eps)

    sort_idx = np.argsort(imp)
    y_pos = np.arange(len(feature_names))
    labels = [feature_display_name(feature_names[i]) for i in sort_idx]
    bh = 0.35

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y_pos - bh/2, imp[sort_idx], bh, color='#4C72B0', alpha=0.85,
            edgecolor='white', linewidth=0.5, label='Prediction Importance (|SHAP|)')
    ax.barh(y_pos + bh/2, dis[sort_idx], bh, color='#C44E52', alpha=0.85,
            edgecolor='white', linewidth=0.5, label='Model Disagreement (SHAP Std)')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Normalized Contribution (0-1)', fontsize=12)
    ax.set_xlim(0, 1.15)

    rho, p_val = spearmanr(imp, dis)
    p_txt = f'p = {p_val:.2e}' if p_val < 0.001 else f'p = {p_val:.3f}'
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='silver')
    ax.text(0.95, 0.15, f'Spearman r = {rho:.3f}\n{p_txt}',
            transform=ax.transAxes, fontsize=11, ha='right', va='bottom', bbox=props)
    ax.legend(loc='lower right', fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出1: {save_path}")


# ==========================================
# 3. 输出2: 主导不确定性来源空间分布 (单独)
# ==========================================
def save_fig2(lons, lats, dominant_idx, feature_names, save_path, top_n=4):
    from collections import Counter
    top_features = [idx for idx, _ in Counter(dominant_idx).most_common(top_n)]
    cat_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#95a5a6']
    cat_array = np.full(len(dominant_idx), len(top_features), dtype=int)
    cat_labels = []
    for i, feat_idx in enumerate(top_features):
        cat_array[dominant_idx == feat_idx] = i
        cat_labels.append(feature_display_name(feature_names[feat_idx]))
    cat_labels.append('Other')

    grid = pd.DataFrame({'lon': lons, 'lat': lats, 'cat': cat_array})
    grid_mode = grid.groupby(['lon', 'lat'])['cat'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else len(top_features)
    ).reset_index()

    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
    add_map_features(ax)

    for ci in range(len(cat_labels)):
        sub = grid_mode[grid_mode['cat'] == ci]
        if len(sub) > 0:
            ax.scatter(sub['lon'], sub['lat'], color=cat_colors[ci], s=14,
                       label=cat_labels[ci], transform=ccrs.PlateCarree(),
                       zorder=3, edgecolor='none')
    ax.legend(loc='lower left', fontsize=8, markerscale=0.8,
              framealpha=0.85, edgecolor='silver')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出2: {save_path}")


# ==========================================
# 4. 输出3: Top-8 特征 SHAP 方差剖面 (4x2 子图)
# ==========================================
def save_fig3(X_raw, shap_std, feature_names, save_path):
    mean_contrib = np.mean(shap_std, axis=0)
    top8_idx = np.argsort(mean_contrib)[-8:]  # 最大的8个
    top8_idx = top8_idx[::-1]  # 从大到小排列 (a) = 最大

    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    axes_flat = axes.flatten()
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    colors = plt.cm.Set2(np.linspace(0, 0.85, 8))

    for i, (ax, idx) in enumerate(zip(axes_flat, top8_idx)):
        fv = X_raw[:, idx]
        sv = shap_std[:, idx]

        p_low, p_high = np.percentile(fv, 2), np.percentile(fv, 98)
        bins = np.linspace(p_low, p_high, 15)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        bm, bs = [], []
        for j in range(len(bins) - 1):
            mask = (fv >= bins[j]) & (fv < bins[j + 1])
            n_pts = np.sum(mask)
            bm.append(np.mean(sv[mask]) if n_pts >= 20 else np.nan)
            bs.append(np.std(sv[mask], ddof=1)/np.sqrt(n_pts) if n_pts >= 20 else np.nan)

        bm = np.array(bm); bs = np.array(bs)
        valid = ~np.isnan(bm)
        ax.errorbar(bin_centers[valid], bm[valid], yerr=bs[valid],
                    fmt='o-', color=colors[i], ecolor='gray',
                    elinewidth=1, capsize=2, markersize=4, linewidth=1.5)

        feat_disp = feature_display_name(feature_names[idx])
        unit = feature_units(feature_names[idx])
        ax.set_xlabel(f'{feat_disp} {unit}', fontsize=9)
        ax.set_ylabel('SHAP Std [ppm]', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.text(0.03, 0.95, panel_labels[i], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    plt.subplots_adjust(hspace=0.35, wspace=0.30)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"输出3: {save_path}")


# ==========================================
# 5. 主执行流
# ==========================================
def main():
    print("=" * 55)
    print("  fig13: SHAP Variance Decomposition (3 outputs)")
    print("=" * 55)

    # 1. 加载
    print("1/5: 加载数据...")
    df = pd.read_pickle(DATA_FILE)
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
    with open(FEATURES_JSON, 'r') as f:
        selected_features = json.load(f)

    # 2. 切分
    print("2/5: 切分数据...")
    df_pool, _ = train_test_split(df_clean, test_size=0.2, random_state=42)
    X_pool_raw = df_pool[selected_features].values
    y_pool = df_pool[TARGET].values
    eval_scaler = joblib.load(SCALER_PATH)
    X_pool_scaled = eval_scaler.transform(X_pool_raw)
    np.random.seed(42)
    shap_sample_size = min(5000, len(X_pool_scaled))
    shap_idx = np.random.choice(len(X_pool_scaled), shap_sample_size, replace=False)
    X_shap_raw = X_pool_raw[shap_idx]
    X_shap_scaled = X_pool_scaled[shap_idx]
    pool_lons = df_pool['grid_lon'].values[shap_idx]
    pool_lats = df_pool['grid_lat'].values[shap_idx]

    # 3. 10模型推理 + SHAP
    print("3/5: 10模型推理与SHAP计算...")
    shap_values_all = []
    for seed in SEEDS:
        model_path = f"{MODEL_BASE_PATH}_seed{seed}.pkl"
        model = joblib.load(model_path)
        shap_values_all.append(shap.TreeExplainer(model).shap_values(
            eval_scaler.transform(X_shap_raw)))  # 注意: 每次重新transform保证scaler一致
    shap_values_all = np.array(shap_values_all)
    # 重新获取正确的scaled数据
    X_shap_scaled = eval_scaler.transform(X_shap_raw)
    # 重新推理获取std
    test_preds_all = []
    for seed in SEEDS:
        model_path = f"{MODEL_BASE_PATH}_seed{seed}.pkl"
        model = joblib.load(model_path)
        test_preds_all.append(model.predict(X_shap_scaled))
    test_preds_all = np.array(test_preds_all)
    ensemble_std_raw = np.std(test_preds_all, axis=0)
    shap_std = np.std(shap_values_all, axis=0)
    dominant_idx = np.argmax(shap_std, axis=1)

    # 4. 三路输出
    print("4/4: 渲染3个输出...")
    p1 = os.path.join(OUTPUT_DIR, 'FIG13-Importance_vs_Disagreement.png')
    p2 = os.path.join(OUTPUT_DIR, 'FIG13-Dominant_Uncertainty_Source.png')
    p3 = os.path.join(OUTPUT_DIR, 'FIG13-Feature_SHAP_Profiles.png')

    save_fig1(shap_values_all, shap_std, selected_features, p1)
    save_fig2(pool_lons, pool_lats, dominant_idx, selected_features, p2)
    save_fig3(X_shap_raw, shap_std, selected_features, p3)

    print("fig13 全部完成！")

if __name__ == "__main__":
    main()
