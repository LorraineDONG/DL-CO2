# -*- coding: utf-8 -*-
"""
fig9.py — 分层可靠性图 (Stratified Reliability Diagram)

将测试集按 XCO2en 预测量级、季节、模型分歧度（raw std）分层，
对每个子集独立计算 optimal_k 和校准前后的 PICP 覆盖率曲线。
输出 2x3 面板拼图。

【v2 修改】
  - 面板排列: (a)低 (b)中 (c)高 | (d)夏季 (e)冬季 (f)分歧
  - 图例仅保留在 (a), 去除背景框, 置于文本下方
  - 每个子图纵轴隐藏 0% 刻度
  - 仅 (a)(d) 保留左纵轴, 仅 (d)(e)(f) 保留底轴
  - 文本框去除全局 k 引用
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import brentq
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter, FuncFormatter

# ==========================================
# 0. 全局学术级字体与样式配置
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 8,
    'axes.linewidth': 1.5,
    'lines.linewidth': 1.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
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

# ==========================================
# 1. 路径配置
# ==========================================
DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'
SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
MODEL_BASE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01'
OUTPUT_DIR = '/home/whdong/dl/figures/scatter_density_A01'
TARGET = 'xco2_enhanced'
SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. 核心可靠性计算函数
# ==========================================

def calculate_coverage(y_true, y_pred, std_raw, k_factor, expected_prob):
    z_score = norm.ppf(0.5 + expected_prob / 2.0)
    calibrated_std = std_raw * k_factor
    lower = y_pred - z_score * calibrated_std
    upper = y_pred + z_score * calibrated_std
    return np.mean((y_true >= lower) & (y_true <= upper))


def find_optimal_k(y_true, y_pred, std_raw, target_picp=0.95, k_range=(0.5, 35.0)):
    def objective(k):
        calibrated_std = std_raw * k
        lower = y_pred - 1.96 * calibrated_std
        upper = y_pred + 1.96 * calibrated_std
        picp = np.mean((y_true >= lower) & (y_true <= upper))
        return picp - target_picp
    try:
        return brentq(objective, k_range[0], k_range[1])
    except (ValueError, RuntimeError):
        return np.nan


def compute_reliability_curve(y_true, y_pred, std_raw, k_factor):
    expected_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    coverages = np.array([calculate_coverage(y_true, y_pred, std_raw, k_factor, p)
                          for p in expected_probs])
    return expected_probs, coverages


def calibration_error(probs, coverages):
    """校准误差：校准后曲线与 1:1 线的平均绝对偏差 (百分比)"""
    return np.mean(np.abs(coverages - probs)) * 100


# ==========================================
# 3. 分层 Mask 生成 (新排序)
# ==========================================
def build_stratification_masks(df_test, ensemble_pred, ensemble_std_raw):
    """6 组分层: (a)<2 (b)2-5 (c)>5 | (d)夏 (e)冬 (f)Top10%分歧"""
    y_pred = ensemble_pred
    raw_std = ensemble_std_raw
    months = df_test['month'].values
    n = len(y_pred)

    labels = []
    masks = []

    # (a) XCO2en 低值 (< 2 ppm)
    labels.append('ΔXCO$_2$ Low (<2 ppm)')
    masks.append(y_pred < 2.0)
    # (b) XCO2en 中间 (2-5 ppm)
    labels.append('ΔXCO$_2$ Mid (2-5 ppm)')
    masks.append((y_pred >= 2.0) & (y_pred <= 5.0))
    # (c) XCO2en 高值 (> 5 ppm)
    labels.append('ΔXCO$_2$ High (>5 ppm)')
    masks.append(y_pred > 5.0)
    # (d) 夏季
    labels.append('Season: Summer (JJA)')
    masks.append(np.isin(months, [6, 7, 8]))
    # (e) 冬季
    labels.append('Season: Winter (DJF)')
    masks.append(np.isin(months, [12, 1, 2]))
    # (f) 高模型分歧度 (Top 10%)
    p90 = np.percentile(raw_std, 90)
    labels.append('Raw Ensemble Std: Top 10%')
    masks.append(raw_std >= p90)

    return labels, masks
# def build_stratification_masks(df_test, ensemble_pred, ensemble_std_raw):
#     """6 组分层: (a)低 (b)中 (c)高 | (d)夏 (e)冬 (f)高分歧"""
#     y_pred = ensemble_pred
#     raw_std = ensemble_std_raw
#     months = df_test['month'].values
#     n = len(y_pred)

#     labels = []
#     masks = []

#     # 计算 33% 和 67% 的实际浓度分位数
#     p33 = np.percentile(y_pred, 33)
#     p67 = np.percentile(y_pred, 67)

#     # (a) XCO2en 低值 (动态填入具体浓度，保留 1 位小数)
#     labels.append(f'XCO$_2$en Low (<{p33:.1f} ppm)')
#     masks.append(y_pred < p33)
    
#     # (b) XCO2en 中间
#     labels.append(f'XCO$_2$en Mid ({p33:.1f}-{p67:.1f} ppm)')
#     masks.append((y_pred >= p33) & (y_pred < p67))
    
#     # (c) XCO2en 高值
#     labels.append(f'XCO$_2$en High (>={p67:.1f} ppm)')
#     masks.append(y_pred >= p67)
    
#     # (d) 夏季
#     labels.append('Season: Summer (JJA)')
#     masks.append(np.isin(months, [6, 7, 8]))
    
#     # (e) 冬季
#     labels.append('Season: Winter (DJF)')
#     masks.append(np.isin(months, [12, 1, 2]))
    
#     # (f) 高模型分歧度
#     p80 = np.percentile(raw_std, 80)
#     labels.append(f'Raw Ensemble Std: Top 20%')
#     masks.append(raw_std >= p80)

#     return labels, masks


# ==========================================
# 4. 单个子图绘制函数
# ==========================================

def plot_single_reliability(ax, panel_tag, label, e_probs, raw_cov, cal_cov,
                            k_opt, n_samples,
                            show_legend=False,
                            show_ylabel=False,
                            show_xlabel=False):
    """绘制一个可靠性子图"""

    # 面板标签 (a)-(f) 置于左上角
    ax.text(0.03, 0.97, panel_tag, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)

    # 1:1 完美校准线
    ax.plot([0, 1], [0, 1], color='black', linestyle='--',
            linewidth=1.5, label='Perfect (1:1)')
    # 原始 (未校准) 曲线
    ax.plot(e_probs, raw_cov, marker='o', markersize=5,
            linestyle='-', color='steelblue', linewidth=1.8,
            label='Raw (k=1)')
    # 校准后曲线
    cal_ce = calibration_error(e_probs, cal_cov)
    ax.plot(e_probs, cal_cov, marker='s', markersize=5,
            linestyle='-', color='firebrick', linewidth=1.8,
            label='Calibrated')

    # 坐标轴范围与刻度
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    # 纵轴: 隐藏 0% 刻度 (通用格式器)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: "" if np.isclose(x, 0) else f"{int(round(x*100))}%"))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    # 显示/隐藏坐标轴标签
    if show_ylabel:
        ax.set_ylabel('Observed Coverage (PICP)', fontsize=13)
    else:
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xlabel('Expected Confidence Level', fontsize=13)
    else:
        ax.set_xticklabels([])

    # 文本框 (标签下方)
    text_lines = [
        label,
        f'N = {n_samples:,}',
        f'k = {k_opt:.2f}',
        f'CE = {cal_ce:.1f}%'
    ]
    props = dict(boxstyle='round,pad=0.4', facecolor='white',
                 alpha=0.85, edgecolor='silver')
    ax.text(0.05, 0.85, '\n'.join(text_lines),
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top')#, bbox=props)

    # 图例: 仅 (a) 显示, 置于文本下方, 无背景框
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(0.03, 0.7),
                  fontsize=11, frameon=False)


# ==========================================
# 5. 主绘图函数
# ==========================================

def plot_stratified_reliability(y_test, ensemble_pred, ensemble_raw_std,
                                global_k, df_test, save_path):
    """生成 2x3 分层可靠性图"""
    labels, masks = build_stratification_masks(df_test, ensemble_pred,
                                                ensemble_raw_std)

    panel_data = []
    for mask in masks:
        y_t = y_test[mask]
        y_p = ensemble_pred[mask]
        s_r = ensemble_raw_std[mask]
        n = np.sum(mask)
        if n < 100:
            panel_data.append({
                'e_probs': np.array([0.1, 0.5, 0.95]),
                'raw_cov': np.array([0.1, 0.5, 0.95]),
                'cal_cov': np.array([0.1, 0.5, 0.95]),
                'k_opt': np.nan, 'n': n
            })
            continue
        e_probs, raw_cov = compute_reliability_curve(y_t, y_p, s_r, 1.0)
        k_opt = find_optimal_k(y_t, y_p, s_r)
        if np.isnan(k_opt):
            k_opt = global_k
        _, cal_cov = compute_reliability_curve(y_t, y_p, s_r, k_opt)
        panel_data.append({
            'e_probs': e_probs, 'raw_cov': raw_cov,
            'cal_cov': cal_cov, 'k_opt': k_opt, 'n': n
        })

    # 2x3 布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    panel_tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for idx, (ax, data, lbl) in enumerate(zip(axes_flat, panel_data, labels)):
        plot_single_reliability(
            ax, panel_tags[idx], lbl,
            data['e_probs'], data['raw_cov'], data['cal_cov'],
            data['k_opt'], data['n'],
            show_legend=(idx == 0),
            show_ylabel=(idx in [0, 3]),
            show_xlabel=(idx in [3, 4, 5])
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"分层可靠性图已保存至: {save_path}")


# ==========================================
# 6. 主执行流
# ==========================================

def main():
    print("=" * 55)
    print("  Fig 9: Stratified Reliability Diagram")
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

    print("2/4: 切分测试集 (20%)...")
    _, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
    X_test_raw = df_test[selected_features].values
    y_test = df_test[TARGET].values

    print("3/4: 加载标准化器并缩放...")
    eval_scaler = joblib.load(SCALER_PATH)
    X_test_scaled = eval_scaler.transform(X_test_raw)

    print("4/4: 加载 10 个模型并推理...")
    test_preds = []
    for seed in SEEDS:
        model_path = f"{MODEL_BASE_PATH}_seed{seed}.pkl"
        model = joblib.load(model_path)
        test_preds.append(model.predict(X_test_scaled))
    test_preds = np.array(test_preds)
    ensemble_pred = np.mean(test_preds, axis=0)
    ensemble_raw_std = np.std(test_preds, axis=0)

    global_k = find_optimal_k(y_test, ensemble_pred, ensemble_raw_std)
    if np.isnan(global_k):
        global_k = 11.99
    print(f"   全局 k = {global_k:.3f}")

    save_path = os.path.join(OUTPUT_DIR, 'FIG9-Stratified_Reliability.png')
    plot_stratified_reliability(y_test, ensemble_pred, ensemble_raw_std,
                                global_k, df_test, save_path)
    print("Fig 9 全部完成！")

if __name__ == "__main__":
    main()
