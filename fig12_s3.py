# -*- coding: utf-8 -*-
"""
fig12_s3.py (Supplementary) — OOD 分层相空间误差对比图

逻辑完全继承 fig12.py，但分别对 Low OOD (P<33) 和 High OOD (P>67)
独立输出 3x2 相空间面板。
两张图共享同一色带范围 (full-test-set percentile)，确保可公平比较。

六组驱动因子对 (与 fig12 完全一致):
  (a) 风速 x NO2   (b) PBLH x NO2   (c) NDVI x NO2
  (d) 海拔 x 风速  (e) T2M x PBLH   (f) NO2 x T2M

输出:
  FigS3_LowOOD_Phase_Space.png
  FigS3_HighOOD_Phase_Space.png
"""

import os, json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'axes.labelweight': 'normal',
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 9,
    'axes.linewidth': 1.5, 'lines.linewidth': 1.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.top': True, 'ytick.right': True,
    'axes.grid': True, 'grid.linestyle': '--', 'grid.linewidth': 0.6,
    'grid.alpha': 0.35, 'grid.color': '#B0B0B0',
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05
})

DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'
SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
MODEL_BASE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01'
OUTPUT_DIR = '/home/whdong/dl/figures/scatter_density_A01'
TARGET = 'xco2_enhanced'
SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_phase_panel(ax, x, y, sigma, x_label, y_label, panel_tag,
                     gridsize=35, vmin=None, vmax=None, text_anchor='tl'):
    """hexbin 展示 2D 相空间 (与 fig12.py 完全一致)"""
    valid = (x < np.percentile(x, 99.5)) & (y < np.percentile(y, 99.5))
    valid &= (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(sigma))
    x_sub, y_sub, s_sub = x[valid], y[valid], sigma[valid]
    n = len(x_sub)
    if vmin is None: vmin = np.percentile(s_sub, 5)
    if vmax is None: vmax = np.percentile(s_sub, 95)

    num_bins = 5
    discrete_cmap = plt.get_cmap('Spectral_r', num_bins)

    hb = ax.hexbin(x_sub, y_sub, C=s_sub, reduce_C_function=np.mean,
                   gridsize=gridsize, cmap=discrete_cmap,
                   vmin=vmin, vmax=vmax, mincnt=5,
                   alpha=0.85, edgecolors='none')

    r_x, p_x = spearmanr(x_sub, s_sub)
    r_y, p_y = spearmanr(y_sub, s_sub)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    def p_fmt(p):
        return 'p < 0.001' if p < 0.001 else f'p = {p:.4f}'
    txt = (f'N = {n:,}\n'
           f'R(x) = {r_x:.3f} ({p_fmt(p_x)})\n'
           f'R(y) = {r_y:.3f} ({p_fmt(p_y)})')
    props = dict(boxstyle='round,pad=0.35', facecolor='white',
                 alpha=0.85, edgecolor='silver')
    if text_anchor == 'tr':
        ax.text(0.96, 0.96, txt, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                horizontalalignment='right', bbox=props)
    else:
        ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=props)
    ax.text(0.96, 0.04, panel_tag, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='right')
    return hb


def create_ood_phase_space(ood_mask, ood_label, df_test, sigma,
                           vmin, vmax, save_path):
    """对指定 OOD 子集绘制 3x2 相空间 (与 fig12 同结构)"""
    idx = np.where(ood_mask)[0]

    no2 = df_test['no2_trop'].values[idx]
    ws  = df_test['era5_wind_speed'].values[idx]
    blh = df_test['era5_blh'].values[idx]
    ndvi_val = df_test['ndvi'].values[idx]
    dem = df_test['dem_mean'].values[idx]
    t2m = df_test['era5_t2m'].values[idx]
    s_sub = sigma[idx]

    panels = [
        (ws, no2, r'Wind Speed [m/s]', r'NO$_2$ VCD [molecules/cm$^2$]', '(a)'),
        (blh, no2, r'PBLH [m]', r'NO$_2$ VCD [molecules/cm$^2$]', '(b)'),
        (ndvi_val, no2, r'NDVI', r'NO$_2$ VCD [molecules/cm$^2$]', '(c)'),
        (dem, ws, r'Elevation [m]', r'Wind Speed [m/s]', '(d)'),
        (t2m, blh, r'T$_{2m}$ [K]', r'PBLH [m]', '(e)'),
        (no2, t2m, r'NO$_2$ VCD [molecules/cm$^2$]', r'T$_{2m}$ [K]', '(f)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.flatten()
    hb_cbar = None

    for idx_p, (ax, panel) in enumerate(zip(axes_flat, panels)):
        x, y, xl, yl, tag = panel
        kwargs = dict(vmin=vmin, vmax=vmax)
        if idx_p == 5:
            kwargs['text_anchor'] = 'tr'
        hb = draw_phase_panel(ax, x, y, s_sub, xl, yl, tag, **kwargs)
        if idx_p == 0:
            hb_cbar = hb
        if idx_p >= 3:
            ax.set_xlabel(xl, fontsize=13)
        if idx_p in [0, 3]:
            ax.set_ylabel(yl, fontsize=13)

    fig.suptitle(ood_label, fontsize=14, fontweight='bold', y=0.98)

    cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(hb_cbar, cax=cbar_ax)
    cbar.set_label(r'Mean Prediction Error $\sigma$ [ppm]',
                   fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.06, right=0.92, bottom=0.08, top=0.93,
                        wspace=0.25, hspace=0.28)
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close()
    print(f"  {save_path}")


def main():
    print("=" * 55)
    print("  Fig S3: OOD-stratified Phase Space")
    print("=" * 55)

    # 1. 加载数据 (与 fig12 完全一致)
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
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    with open(FEATURES_JSON, 'r') as f:
        selected_features = json.load(f)

    # 2. 切分
    print("2/5: 切分数据...")
    df_pool, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
    X_pool_raw = df_pool[selected_features].values
    X_test_raw = df_test[selected_features].values
    y_test = df_test[TARGET].values

    # 3. 缩放与推理
    print("3/5: 缩放与推理...")
    eval_scaler = joblib.load(SCALER_PATH)
    X_pool_scaled = eval_scaler.transform(X_pool_raw)
    X_test_scaled = eval_scaler.transform(X_test_raw)
    test_preds = []
    for seed in SEEDS:
        model = joblib.load(f"{MODEL_BASE_PATH}_seed{seed}.pkl")
        test_preds.append(model.predict(X_test_scaled))
    test_preds = np.array(test_preds)
    ensemble_pred = np.mean(test_preds, axis=0)
    sigma = np.abs(ensemble_pred - y_test)

    # 4. 计算 OOD 分数
    print("4/5: 计算 OOD 分数...")
    nn = NearestNeighbors(n_neighbors=100, metric='euclidean', n_jobs=-1)
    nn.fit(X_pool_scaled)
    dists, _ = nn.kneighbors(X_test_scaled)
    ood_raw = dists.mean(axis=1)
    ood_score = (ood_raw - ood_raw.mean()) / ood_raw.std()

    low_mask = ood_score < np.percentile(ood_score, 33)
    high_mask = ood_score > np.percentile(ood_score, 67)
    print(f"   Low OOD: {np.sum(low_mask):,}  High OOD: {np.sum(high_mask):,}")

    # 统一的色带范围 (基于全样本)
    vmin = np.percentile(sigma, 5)
    vmax = np.percentile(sigma, 98)

    # 5. 输出两张图
    print("5/5: 输出 Low OOD 和 High OOD 相空间图...")
    p1 = os.path.join(OUTPUT_DIR, 'FigS3_LowOOD_Phase_Space.png')
    p2 = os.path.join(OUTPUT_DIR, 'FigS3_HighOOD_Phase_Space.png')

    create_ood_phase_space(low_mask, 'Low OOD (P<33)', df_test, sigma,
                           vmin, vmax, p1)
    create_ood_phase_space(high_mask, 'High OOD (P>67)', df_test, sigma,
                           vmin, vmax, p2)

    print("Fig S3 全部完成！")


if __name__ == "__main__":
    main()
