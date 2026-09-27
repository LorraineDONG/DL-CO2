# -*- coding: utf-8 -*-
"""
tab2.py — 伪观测误差驱动因子归因统计表 (Extended Attribution Table)

量化模型前 20 个入选特征对校准后伪观测误差 sigma 的独立影响强度。

输出:
  1. tab2_Driver_Attribution.csv — 完整的统计表
  2. tab2_Driver_Effect_Sizes.png — 效应量水平条形图 (补充材料)
  3. 控制台打印的格式化表格副本

统计指标:
  - Spearman 秩相关系数 R
  - 显著性 p 值
  - 低四分位 (Q1) sigma 均值
  - 高四分位 (Q3) sigma 均值
  - 效应比 Q3/Q1 (高/低四分位 sigma 比值)
  - 样本量 N

为了消除量纲差异，所有驱动变量在分箱前均进行 Z-score 标准化。
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

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
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.35,
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
# 2. 定义驱动变量 (前 20 个入选特征)
# ==========================================

DRIVER_VARS = [
    ('doy_sin',           r'DOY (sin)',              True),
    ('no2_trop_mean',     r'NO$_2$ Trop Mean',       True),
    ('no2_trop_log',      r'log(NO$_2$ Trop)',       True),
    ('month_sin',         r'Month (sin)',            True),
    ('grid_lat',          r'Latitude',               True),
    ('doy_cos',           r'DOY (cos)',              True),
    ('era5_t2m_lead1',    r'T$_{2m}$ Lead 1',       True),
    ('ssrd_t2m_cross',    r'SSRD $\times$ T$_{2m}$', True),
    ('era5_ssrd_lead1',   r'SSRD Lead 1',            True),
    ('era5_t2m',          r'T$_{2m}$',               True),
    ('grid_lon',          r'Longitude',              True),
    ('era5_ssrd_lag1',    r'SSRD Lag 1',             True),
    ('ndvi_t2m_cross',    r'NDVI $\times$ T$_{2m}$', True),
    ('era5_blh',          r'PBL Height',             True),
    ('era5_blh_lead1',    r'PBLH Lead 1',            True),
    ('era5_sp',           r'Surface Pressure',       True),
    ('dem_mean',          r'Elevation',              True),
    ('era5_d2m_lag1',     r'D$_{2m}$ Lag 1',         True),
    ('ndvi',              r'NDVI',                   True),
    ('era5_blh_lag1',     r'PBLH Lag 1',             True),
]

# ==========================================
# 3. 核心统计计算
# ==========================================

def compute_driver_stats(sigma, driver_col, col_name, higher_is_better=True):
    """
    对单个驱动变量计算与 sigma 的关联统计。
    返回一个有序字典，包含所有指标。
    """
    valid = (~np.isnan(sigma)) & (~np.isnan(driver_col))
    s = sigma[valid]
    d = driver_col[valid]
    n = len(s)

    if n < 100:
        return None

    # Spearman 秩相关
    r_val, p_val = spearmanr(d, s)

    # 按驱动变量分四组
    q1 = np.percentile(d, 25)
    q3 = np.percentile(d, 75)

    mask_low = d <= q1
    mask_high = d >= q3

    sigma_low_mean = np.mean(s[mask_low])
    sigma_high_mean = np.mean(s[mask_high])
    sigma_low_median = np.median(s[mask_low])
    sigma_high_median = np.median(s[mask_high])

    # 效应比: 高值区 sigma / 低值区 sigma
    if sigma_low_mean > 0:
        ratio = sigma_high_mean / sigma_low_mean
    else:
        ratio = np.nan

    return {
        'Driver': col_name,
        'N': n,
        'Spearman_R': round(r_val, 4),
        'p_value': p_val,
        'Sigma_Low_Q_Mean': round(sigma_low_mean, 4),
        'Sigma_High_Q_Mean': round(sigma_high_mean, 4),
        'Sigma_Low_Q_Median': round(sigma_low_median, 4),
        'Sigma_High_Q_Median': round(sigma_high_median, 4),
        'Effect_Ratio_High_Low': round(ratio, 3),
    }


# ==========================================
# 4. 驱动因子效应量条形图
# ==========================================

def plot_effect_size_barchart(results, save_path):
    """绘制驱动因子效应量排序水平条形图"""
    df = pd.DataFrame(results)
    df = df.sort_values('Effect_Ratio_High_Low', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 9))

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df['Driver'], df['Effect_Ratio_High_Low'],
                   color=colors, edgecolor='#333333', linewidth=0.8,
                   height=0.5)

    # 在条形末端标注 Spearman R
    for i, (_, row) in enumerate(df.iterrows()):
        r_str = f'R = {row["Spearman_R"]:.3f}'
        sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01
               else '*' if row['p_value'] < 0.05 else '')
        ax.text(row['Effect_Ratio_High_Low'] + 0.02, i,
                f'{r_str} {sig}', va='center', fontsize=10,
                fontweight='bold', color='#333333')

    ax.axvline(x=1.0, color='firebrick', linestyle='--',
               linewidth=1.5, alpha=0.7, zorder=0)
    ax.set_xlabel(r'Effect Ratio ($\sigma_{Q3}$ / $\sigma_{Q1}$)',
                  fontsize=12, fontweight='bold')
    ax.set_title('Physical Driver Impact on Prediction Uncertainty',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(left=0.5)
    ax.grid(True, axis='x', linestyle=':', alpha=0.4)
    ax.grid(False, axis='y')

    props = dict(boxstyle='round,pad=0.4', facecolor='white',
                 alpha=0.85, edgecolor='silver')
    ax.text(0.98, 0.04,
            r'Ratio > 1: High driver $\rightarrow$ larger $\sigma$'
            '\nSignificance: *** p<0.001, ** p<0.01, * p<0.05',
            transform=ax.transAxes, fontsize=9, ha='right',
            va='bottom', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"驱动因子效应量条形图已保存至: {save_path}")


# ==========================================
# 5. 格式化输出
# ==========================================

def print_table_to_console(results):
    """在控制台打印格式化表格"""
    print("\n" + "=" * 90)
    print("  Table: Physical Driver Attribution of Prediction Uncertainty")
    print("=" * 90)
    print(f"{'Driver':<22} {'N':>8} {'Spearman_R':>12} {'p-value':>10} "
          f"{'Sigma_Q1_mean':>14} {'Sigma_Q3_mean':>14} {'Ratio Q3/Q1':>12}")
    print("-" * 90)
    for r in results:
        p_str = f'{r["p_value"]:.2e}' if r['p_value'] < 0.001 else f'{r["p_value"]:.4f}'
        print(f"{r['Driver']:<22} {r['N']:>8,} {r['Spearman_R']:>12.4f} {p_str:>10} "
              f"{r['Sigma_Low_Q_Mean']:>14.3f} {r['Sigma_High_Q_Mean']:>14.3f} "
              f"{r['Effect_Ratio_High_Low']:>12.3f}")
    print("=" * 90)
    print(f"  Ratio Interpretation: Values > 1 indicate that higher driver values "
          f"are associated with larger sigma.")
    print()


# ==========================================
# 6. 主执行流
# ==========================================

def compute_top10_attribution(sigma, driver_dict, results):
    """Top 10% 高 sigma 格点归因"""
    import os
    thresh_sigma = np.percentile(sigma, 90)
    high_mask = sigma >= thresh_sigma
    n_high = np.sum(high_mask)
    n_total = len(sigma)

    print(f"\n===== Top 10%% 高 sigma 归因分析 =====")
    print(f"阈值: {thresh_sigma:.3f} ppm, 高 sigma 格点数: {n_high:,} / {n_total:,}")

    attribution = {}
    driver_names = []
    for r in results:
        name = r['Driver']
        col_name = None
        for dv_name, dv_label, _ in DRIVER_VARS:
            if dv_label == name:
                col_name = dv_name
                break
        if col_name is None or col_name not in driver_dict:
            continue
        driver_names.append(name)
        vals = driver_dict[col_name]
        spearman_r = r['Spearman_R']
        if spearman_r > 0:
            thr = np.percentile(vals, 80)
            cnt = int(np.sum(vals[high_mask] >= thr))
        else:
            thr = np.percentile(vals, 20)
            cnt = int(np.sum(vals[high_mask] <= thr))
        pct = float(cnt / n_high * 100)
        attribution[name] = {'threshold': thr, 'direction': '+' if spearman_r > 0 else '-', 'count': cnt, 'pct': pct}
        print(f"  {name:<20} {attribution[name]['direction']} {cnt:>6,} ({pct:.1f}%%)")

    n_factors = np.zeros(n_high, dtype=int)
    high_idx = np.where(high_mask)[0]
    for i in range(n_high):
        cnt = 0
        for r in results:
            name = r['Driver']
            col_name = None
            for dv_name, dv_label, _ in DRIVER_VARS:
                if dv_label == name:
                    col_name = dv_name
                    break
            if col_name is None or col_name not in driver_dict:
                continue
            vals = driver_dict[col_name]
            spearman_r = r['Spearman_R']
            if spearman_r > 0:
                if vals[high_idx[i]] >= attribution[name]['threshold']:
                    cnt += 1
            else:
                if vals[high_idx[i]] <= attribution[name]['threshold']:
                    cnt += 1
        n_factors[i] = cnt

    print(f"\n  高 sigma 格点归因于 N 个因子:")
    for n in range(1, 5):
        c = int(np.sum(n_factors >= n))
        print(f"    >= {n} 因子: {c:,} ({c/n_high*100:.1f}%%)")

    rows = []
    for name in driver_names:
        a = attribution[name]
        rows.append({'Attribution_Driver': name, 'Direction': a['direction'], 'HighSigma_Grids': a['count'], 'Pct_of_Top10': round(a['pct'], 1)})
    rows.append({})
    rows.append({'Attribution_Driver': 'Summary --- Cumulative attribution by N factors'})
    for n in range(1, 5):
        c = int(np.sum(n_factors >= n))
        rows.append({'Attribution_Driver': f'  >= {n} factors', 'HighSigma_Grids': c, 'Pct_of_Top10': round(c/n_high*100, 1)})
    rows.append({'Attribution_Driver': 'Total high-sigma grids', 'HighSigma_Grids': n_high, 'Pct_of_Top10': 100.0})

    out_path = os.path.join(OUTPUT_DIR, 'tab2_Top10_Attribution.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Top 10%% 归因表已保存至: {out_path}")
    print()

def main():
    print("=" * 55)
    print("  Tab 2: Driver Attribution Table")
    print("=" * 55)

    # 1. 加载与特征工程
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

    # 2. 切分测试集
    print("2/4: 切分测试集...")
    _, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
    X_test_raw = df_test[selected_features].values
    y_test = df_test[TARGET].values

    # 3. 缩放与推理
    print("3/4: 缩放与推理...")
    eval_scaler = joblib.load(SCALER_PATH)
    X_test_scaled = eval_scaler.transform(X_test_raw)

    test_preds = []
    for seed in SEEDS:
        model_path = f"{MODEL_BASE_PATH}_seed{seed}.pkl"
        model = joblib.load(model_path)
        test_preds.append(model.predict(X_test_scaled))
    test_preds = np.array(test_preds)
    ensemble_pred = np.mean(test_preds, axis=0)

    # sigma = 校准后的不确定性
    sigma = np.abs(ensemble_pred - y_test)

    # 4. 计算每个驱动变量的统计
    print("4/4: 计算驱动因子归因统计...")
    results = []
    for col_name, label, _ in DRIVER_VARS:
        driver_data = df_test[col_name].values
        r = compute_driver_stats(sigma, driver_data, label)
        if r is not None:
            results.append(r)
            print(f"   {label:<22} R={r['Spearman_R']:.4f}, "
                  f"Ratio={r['Effect_Ratio_High_Low']:.3f}")

    # 4b. Top 10% 高 sigma 归因
    driver_dict = {col_name: df_test[col_name].values for col_name, _, _ in DRIVER_VARS}
    compute_top10_attribution(sigma, driver_dict, results)

    # 5. 输出表格
    print_table_to_console(results)

    # 6. 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'tab2_Driver_Attribution.csv')
    df_out = pd.DataFrame(results)
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV 表格已保存至: {csv_path}")

    # 7. 绘制效应量条形图
    fig_path = os.path.join(OUTPUT_DIR, 'FigS1_Driver_Effect_Sizes.png')
    plot_effect_size_barchart(results, fig_path)

    print("\nTab 2 全部完成！")


if __name__ == "__main__":
    main()
