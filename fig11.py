# -*- coding: utf-8 -*-
"""
fig11.py (Supplementary) — 预测误差的空间相关图 (Correlogram)

计算测试集预测误差的经验相关函数 (correlogram)，
展示误差的空间自相关随距离的衰减。

发现：在所有距离上误差相关性 ≈ 0，
说明对于 GEOS-Chem 同化网格 (0.25 deg x 0.3125 deg, ~40 km)，
观测误差协方差矩阵 R 可合理近似为对角阵。
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.4,
    'grid.color': '#B0B0B0',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'
SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
MODEL_BASE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01'
OUTPUT_DIR = '/home/whdong/dl/figures/scatter_density_A01'
TARGET = 'xco2_enhanced'
SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]
os.makedirs(OUTPUT_DIR, exist_ok=True)


def haversine_matrix(lons, lats):
    """向量化计算点对球面距离矩阵 (km)"""
    lon1, lat1 = np.radians(lons[:, None]), np.radians(lats[:, None])
    lon2, lat2 = np.radians(lons[None, :]), np.radians(lats[None, :])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0 * c


def compute_correlogram(lons, lats, errors, n_sample=2000, max_dist_km=400):
    """
    计算经验相关函数 (correlogram).
    采样 n_sample 个点，计算全配对的误差相关性和距离。
    """
    n_total = len(errors)
    rng = np.random.RandomState(42)
    idx = rng.choice(n_total, min(n_total, n_sample), replace=False)
    lons_s, lats_s = lons[idx], lats[idx]
    errors_s = errors[idx]
    print(f"   采样 {len(idx)} 个点, 计算全配对距离矩阵...")

    dist_matrix = haversine_matrix(lons_s, lats_s)
    diff_matrix = errors_s[:, None] - errors_s[None, :]
    gamma_matrix = 0.5 * diff_matrix ** 2
    total_var = np.var(errors_s, ddof=1)

    triu_idx = np.triu_indices(len(idx), k=1)
    dists = dist_matrix[triu_idx]
    gamma_vals = gamma_matrix[triu_idx]
    valid = dists <= max_dist_km
    dists, gamma_vals = dists[valid], gamma_vals[valid]
    print(f"   有效配对: {len(dists):,}")

    # 变距分箱: 短距离加密
    bin_edges = np.array([0, 5, 10, 15, 20, 30, 40, 50, 75, 100,
                          150, 200, 300, 400])
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    results = []
    for i in range(n_bins):
        mask = (dists >= bin_edges[i]) & (dists < bin_edges[i + 1])
        n_pts = np.sum(mask)
        if n_pts < 50:
            continue
        g_mean = np.mean(gamma_vals[mask])
        g_sem = np.std(gamma_vals[mask], ddof=1) / np.sqrt(n_pts)
        corr = 1.0 - g_mean / total_var
        results.append({
            'd': bin_centers[i],
            'gamma': g_mean,
            'gamma_sem': g_sem,
            'corr': corr,
            'n_pairs': n_pts,
        })

    return results, total_var, len(idx)


def main():
    print("=" * 55)
    print("  Fig 11 (Supp): Error Correlogram")
    print("=" * 55)

    print("1/4: 加载数据...")
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
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    with open(FEATURES_JSON, 'r') as f:
        selected_features = json.load(f)

    print("2/4: 切分测试集...")
    _, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
    X_test_raw = df_test[selected_features].values
    y_test = df_test[TARGET].values
    lons = df_test['grid_lon'].values
    lats = df_test['grid_lat'].values

    print("3/4: 推理...")
    eval_scaler = joblib.load(SCALER_PATH)
    X_test_scaled = eval_scaler.transform(X_test_raw)
    test_preds = []
    for seed in SEEDS:
        model_path = f"{MODEL_BASE_PATH}_seed{seed}.pkl"
        model = joblib.load(model_path)
        test_preds.append(model.predict(X_test_scaled))
    test_preds = np.array(test_preds)
    errors = np.mean(test_preds, axis=0) - y_test

    print("4/4: 计算相关图...")
    results, total_var, n_pts = compute_correlogram(lons, lats, errors)

    # —— 绘图 ——
    fig, ax = plt.subplots(figsize=(9, 6.5))

    ds = [r['d'] for r in results]
    corrs = [r['corr'] for r in results]

    # 散点 + 连线
    ax.plot(ds, corrs, marker='o', markersize=8, linestyle='-',
            color='#2c3e50', linewidth=2, markerfacecolor='#e74c3c',
            markeredgecolor='white', markeredgewidth=1.2,
            label='Empirical Correlogram')

    # r=0 参考线
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.6)

    # GEOS-Chem 网格尺度参考线 (~40 km, 0.25 deg x 0.3125 deg 对角线)
    gc_scale = 40
    ax.axvline(x=gc_scale, color='#2980b9', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(gc_scale + 3, ax.get_ylim()[1] * 0.85,
            f'GEOS-Chem Grid Scale\n({gc_scale} km)',
            color='#2980b9', fontsize=13, fontweight='bold',
            va='top', ha='left')

    # 文本框: 关键推论
    text_lines = [
        f'Sampled Points: {n_pts}',
        f'Total Pairs: {len(ds):,}',
        f'Error Variance: {total_var:.3f} ppm$^2$',
        # r'$\blacktriangleright$ Prediction errors show negligible',
        # '   spatial correlation at all scales.',
        # r'$\blacktriangleright$ For GEOS-Chem assimilation',
        # '   (0.25$^\circ$ $\\times$ 0.3125$^\circ$),',
        # '   $\\mathbf{R}$ can be treated as diagonal.',
    ]
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                 alpha=0.92, edgecolor='silver')
    ax.text(0.97, 0.97, '\n'.join(text_lines),
            transform=ax.transAxes, fontsize=13,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_xlabel('Distance Between Grid Cells [km]', fontsize=13)
    ax.set_ylabel(r'Error Spatial Correlation $\rho(d)$', fontsize=13)
    ax.set_xlim(0, 355)
    ax.set_ylim(-0.5, 0.5)
    ax.legend(loc='lower right', fontsize=13)
    ax.set_title('Error Spatial Correlogram', fontsize=14, pad=15)

    # 在 r=0 线附近加一个细长的灰色带状区标记 "noise floor"
    ax.axhspan(-0.05, 0.05, color='gray', alpha=0.08, zorder=0)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'FIG11-Error_Correlogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"相关图已保存至: {save_path}")
    print("Fig 11 全部完成！")


if __name__ == "__main__":
    main()
