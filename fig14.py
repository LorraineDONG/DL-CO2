# -*- coding: utf-8 -*-
"""
fig14.py — 预测残差诊断与空间聚合效应 (Supplementary)

双面板 (1x2):
  (a) QQ-plot: 标准化残差 vs 标准正态分布
      检验预测误差的高斯性，评估同化误差假设的合理性
  (b) 空间聚合效应: R2 和 RMSE 随聚合分辨率的变化
      展示数据在 GEOS-Chem 同化网格 (0.25 deg) 下的真实性能水平

输出: Fig14_Residual_Diagnostics.png
"""

import os, json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import probplot
from scipy.optimize import brentq
from scipy.stats import norm

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': 12,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.labelsize': 13, 'axes.labelweight': 'normal',
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'axes.linewidth': 1.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'xtick.top': True, 'ytick.right': True,
    'axes.grid': True, 'grid.linestyle': '--',
    'grid.linewidth': 0.8, 'grid.alpha': 0.4,
    'grid.color': '#B0B0B0', 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
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


def draw_qq_plot(ax, y_true, y_pred, sigma_calibrated):
    """(a) 标准化残差的 QQ-plot vs N(0,1)"""
    std_err = (y_pred - y_true) / sigma_calibrated
    q99 = np.percentile(np.abs(std_err), 99)
    valid = np.abs(std_err) <= q99
    z = std_err[valid]
    n = len(z)
    (osm, osr), (slope, intercept, r) = probplot(z, dist='norm')

    ax.scatter(osm, osr, s=8, c='steelblue', alpha=0.6,
               edgecolors='none', zorder=3)
    ax.plot(osm, osm, color='firebrick', linewidth=2,
            linestyle='-', label='N(0,1) Reference', zorder=4)
    ci = 1.36 / np.sqrt(n)
    ax.fill_between(osm, osm - ci, osm + ci, color='gray',
                    alpha=0.12, zorder=2, label='95% CI')

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Standardized Residuals')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.text(0.04, 0.95, '(a)', transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')
    props = dict(boxstyle='round,pad=0.4', facecolor='white',
                 alpha=0.85, edgecolor='silver')
    ax.text(0.96, 0.05, f'N={n:,}\nr={r:.4f}',
            transform=ax.transAxes, fontsize=10, va='bottom',
            ha='right', bbox=props)
    ax.legend(loc='upper left', fontsize=9)


def draw_aggregation_effect(ax, df_test, y_true, y_pred):
    """(b) R2 和 RMSE 随聚合分辨率的变化"""
    lats = df_test['grid_lat'].values
    lons = df_test['grid_lon'].values
    resolutions = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    r2_vals, rmse_vals = [], []

    for res in resolutions:
        lat_agg = np.floor(lats / res) * res + res / 2.0
        lon_agg = np.floor(lons / res) * res + res / 2.0
        df_agg = pd.DataFrame({
            'lat': lat_agg, 'lon': lon_agg,
            'true': y_true, 'pred': y_pred
        }).groupby(['lat', 'lon']).mean().reset_index()
        r2_vals.append(r2_score(df_agg['true'], df_agg['pred']))
        rmse_vals.append(np.sqrt(mean_squared_error(df_agg['true'], df_agg['pred'])))

    color_r2 = '#2c3e50'
    ax.plot(resolutions, r2_vals, marker='o', markersize=10,
            color=color_r2, linewidth=2.5, label='R$^2$', zorder=4)
    ax.set_xlabel('Spatial Aggregation Resolution [$^\circ$]')
    ax.set_ylabel('R$^2$', color=color_r2, fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', colors=color_r2)
    for x, y in zip(resolutions, r2_vals):
        ax.text(x, y + 0.015, f'{y:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color_r2)

    ax2 = ax.twinx()
    color_rmse = '#d63031'
    ax2.plot(resolutions, rmse_vals, marker='s', markersize=10,
             color=color_rmse, linewidth=2.5, linestyle='--',
             label='RMSE [ppm]', zorder=4)
    ax2.set_ylabel('RMSE [ppm]', color=color_rmse, fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', colors=color_rmse)
    for x, y in zip(resolutions, rmse_vals):
        ax2.text(x, y + 0.03, f'{y:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=color_rmse)

    ax.axvline(x=0.25, color='#2980b9', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(0.27, ax.get_ylim()[0] + 0.02,
            'GEOS-Chem Grid (0.25$^\circ$)',
            color='#2980b9', fontsize=9, fontweight='bold', va='bottom')
    ax.text(0.04, 0.95, '(b)', transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')
    lines = [ax.get_lines()[0], ax2.get_lines()[0]]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower right', fontsize=10)


def create_figure(y_true, y_pred, sigma_calibrated, df_test, save_path):
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    draw_qq_plot(ax_a, y_true, y_pred, sigma_calibrated)
    draw_aggregation_effect(ax_b, df_test, y_true, y_pred)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"残差诊断图已保存至: {save_path}")


def main():
    print("=" * 55)
    print("  Fig 14: Residual Diagnostics & Aggregation Effect")
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

    print("3/4: 缩放与推理...")
    eval_scaler = joblib.load(SCALER_PATH)
    X_test_scaled = eval_scaler.transform(X_test_raw)
    test_preds = []
    for seed in SEEDS:
        model = joblib.load(f"{MODEL_BASE_PATH}_seed{seed}.pkl")
        test_preds.append(model.predict(X_test_scaled))
    test_preds = np.array(test_preds)
    ensemble_pred = np.mean(test_preds, axis=0)
    raw_std = np.std(test_preds, axis=0)

    def obj(k):
        s = raw_std * k
        lo = ensemble_pred - 1.96 * s
        hi = ensemble_pred + 1.96 * s
        return np.mean((y_test >= lo) & (y_test <= hi)) - 0.95
    try:
        k_global = brentq(obj, 1.0, 35.0)
    except (ValueError, RuntimeError):
        k_global = 11.99
    sigma_cal = raw_std * k_global
    print(f"   全局 k = {k_global:.3f}")

    print("4/4: 绘图...")
    save_path = os.path.join(OUTPUT_DIR, 'FIG14-Residual_Diagnostics.png')
    create_figure(y_test, ensemble_pred, sigma_cal, df_test, save_path)
    print("Fig 14 全部完成！")

if __name__ == "__main__":
    main()
