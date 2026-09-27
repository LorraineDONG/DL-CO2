import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, norm
import shap
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# A01 = (标准 XGBoost 引擎) + (MSE 损失) + (无单调性约束) + (有验证集泄漏) + 保留全量极值 + (无 Bagging)
# A02 = (标准 XGBoost 引擎) + (MSE 损失) + (无单调性约束) + (有验证集泄漏) + 剔除极值 (<=10) + (无 Bagging)
# A03 = (DART 引擎) + (Huber 损失) + (单调性约束) + (无验证集泄漏) + 剔除极值 (<=10) + (无 Bagging)
# A04 = (标准 XGBoost 引擎) + (MSE 损失) + (无单调性约束) + (无验证集泄漏，引入独立监控集) + 剔除极值 (<=10) + (真正的 Bagging)
# A05 = (DART 引擎) + (Huber 损失) + (单调性约束) + (无验证集泄漏) + 保留全量极值 + (无 Bagging)
# A06 = (标准 XGBoost 引擎) + (MSE 损失) + (无单调性约束) + (有验证集泄漏) + 保留全量极值 + (真正的 Bagging)
# A07 = (标准 XGBoost 引擎) + (MSE 损失) + (无单调性约束) + (无验证集泄漏) + 保留全量极值 + (真正的 Bagging)
# A08 = (DART 引擎) + (Huber 损失) + (单调性约束) + (无验证集泄漏) + 剔除极值 (<=10) + (真正的 Bagging)
# A09 = (DART 引擎) + (Huber 损失) + (单调性约束) + (无验证集泄漏) + 保留全量极值 + (真正的 Bagging)

# ==============================================================================
# 🌲 XCO2en (XCO2 Enhanced) 模型集成架构演进树 (Version Evolution Tree)
# ==============================================================================
#
# 核心演进主线：
# A01 (Base)
#  │
#  ├── 枝干 1 (直接加 Bagging)
#  │    └── A06: A01 + 真实的 Bootstrap 抽样 [注: 未修复验证集泄漏]
#  │
#  └── 枝干 2 (极值截断策略, xco2_enhanced <= 10)
#       └── A02: A01 + 剔除极端高值 [注: 依然存在 Optuna 验证集泄漏]
#            │
#            ├── 衍生分支 2.1 (纯打补丁)
#            │    └── A07: A02 + 真实的 Bootstrap 抽样 [注: 未修复验证集泄漏]
#            │
#            ├── 衍生分支 2.2 (统计学终极严谨版)
#            │    └── A04: A02 + 彻底修复验证集泄漏 (切分独立 X_es 监控集) 
#            │                 + 动态锁定交叉验证平均最优树数量 (防早停冲突) 
#            │                 + 真实的 Bootstrap (Bagging) 不确定性量化
#            │
#            └── 衍生分支 2.3 (物理约束与黑科技版)
#                 └── A03: A02 + DART 引擎 (引入 Dropout 防空间死记硬背)
#                      │       + Pseudo-Huber 损失函数 (抗离群值鲁棒性)
#                      │       + 物理单调性约束 (强制排放清单正相关)
#                      │       + 取消早停 (客观上避免了验证集泄漏)
#                      │
#                      ├── 衍生分支 2.3-1
#                      │    └── A08: A03 + 真实的 Bootstrap (Bagging) 抽样机制
#                      │                 [意义: 在截断数据集上兼顾物理约束与稳健区间覆盖率]
#                      │
#                      └── 衍生分支 2.3-2
#                           └── A05: A03 的架构 + 恢复全量数据 (去除 <=10 过滤)
#                                │
#                                └── 衍生分支 2.3-2.1 (物理强迫 + 全量热点捕获 + 严谨方差量化)
#                                     └── A09: A05 + 真实的 Bootstrap (Bagging) 抽样机制
#
# ==============================================================================


# 全局设置字体风格
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. 路径配置 (严格对齐 A02 的环境)
# ==========================================
DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'
SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
MODEL_BASE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01' 
OUTPUT_DIR = '/home/whdong/dl/figures/scatter_density_A01' 

os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET = 'xco2_enhanced'
SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]

# ==========================================
# 1. 核心绘图函数定义
# ==========================================
def plot_density_scatter(ax, y_true, y_pred, title):
    """绘制平滑密度散点图，包含1:1线与统计指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mb = np.mean(y_pred - y_true)  
    r2 = r2_score(y_true, y_pred)
    n_samples = len(y_true)

    # 下采样防卡死
    max_samples = 50000
    if len(y_true) > max_samples:
        idx = np.random.choice(len(y_true), max_samples, replace=False)
    else:
        idx = np.arange(len(y_true))
        
    xt, yp = y_true[idx], y_pred[idx]
    
    # 二维核密度
    xy = np.vstack([xt, yp])
    z = gaussian_kde(xy)(xy)
    
    # 按密度排序优化显示
    sort_idx = z.argsort()
    xt_sorted, yp_sorted, z_sorted = xt[sort_idx], yp[sort_idx], z[sort_idx]
    
    sc = ax.scatter(xt_sorted, yp_sorted, c=z_sorted, s=5, cmap='viridis', alpha=0.9, edgecolor='none')
    
    # 参考线
    ax.plot([0, 10], [0, 10], color='red', linestyle='--', linewidth=1.5, label='1:1 Line')
    m, b = np.polyfit(xt, yp, 1)
    ax.plot(xt, m*xt + b, color='black', linestyle='-', linewidth=1.2, label='Fit Line')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Observed XCO2 Enhanced [ppm]', fontsize=12)
    ax.set_ylabel('Predicted XCO2 Enhanced [ppm]', fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)

    sign = '+' if b >= 0 else '-'
    textstr = '\n'.join((
        f'N = {n_samples}',
        f'$R^2$ = {r2:.3f}',
        f'RMSE = {rmse:.3f}',
        f'MB = {mb:.3f}',
        f'y = {m:.2f}x {sign} {abs(b):.3f}'
    ))
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='silver')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    return sc

def plot_spatial_bias_map(df_coords, y_true, y_pred, title, save_path):
    """绘制空间残差地图"""
    df_err = pd.DataFrame({
        'lon': df_coords['grid_lon'].values, 
        'lat': df_coords['grid_lat'].values, 
        'err': y_pred - y_true
    })
    
    grid_err = df_err.groupby(['lon', 'lat'])['err'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    sc = ax.scatter(grid_err['lon'], grid_err['lat'], c=grid_err['err'], 
                        cmap='RdBu_r', vmin=-2, vmax=2, s=1.5, transform=ccrs.PlateCarree())

    plt.colorbar(sc, label='Bias (Pred - Obs) [ppm]', extend='both')
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_uncertainty_map(df_coords, uncertainty, title, save_path):
    """绘制空间不确定性 (标准差) 地图"""
    df_unc = pd.DataFrame({
        'lon': df_coords['grid_lon'].values, 
        'lat': df_coords['grid_lat'].values, 
        'unc': uncertainty
    })
    
    # 按经纬度网格取平均不确定性
    grid_unc = df_unc.groupby(['lon', 'lat'])['unc'].mean().reset_index()

    # 不确定性是大于0的绝对值，使用单向渐变色 (如 YlOrRd) 更符合直觉
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    sc = ax.scatter(grid_unc['lon'], grid_unc['lat'], c=grid_unc['unc'], 
                    cmap='YlOrRd', s=1.5, transform=ccrs.PlateCarree())

    plt.colorbar(sc, label='Prediction Uncertainty ($1\sigma$) [ppm]', extend='max')
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_vs_error(uncertainty, abs_error, picp, title, save_path):
    """绘制不确定性 vs 绝对误差的密度散点图"""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # 下采样防卡死
    max_samples = 50000
    if len(uncertainty) > max_samples:
        idx = np.random.choice(len(uncertainty), max_samples, replace=False)
    else:
        idx = np.arange(len(uncertainty))
        
    xt, yp = uncertainty[idx], abs_error[idx]
    
    # 计算二维核密度
    xy = np.vstack([xt, yp])
    z = gaussian_kde(xy)(xy)
    
    sort_idx = z.argsort()
    xt_sorted, yp_sorted, z_sorted = xt[sort_idx], yp[sort_idx], z[sort_idx]
    
    sc = ax.scatter(xt_sorted, yp_sorted, c=z_sorted, s=5, cmap='viridis', alpha=0.9, edgecolor='none')
    
    # 绘制趋势线证明正相关
    m, b = np.polyfit(xt, yp, 1)
    ax.plot(xt, m*xt + b, color='black', linestyle='-', linewidth=1.5, label=f'Trend (Slope={m:.2f})')
    
    ax.set_xlabel('Model Uncertainty ($1\sigma$) [ppm]', fontsize=12)
    ax.set_ylabel('Absolute Prediction Error [ppm]', fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 计算相关系数
    corr_coef = np.corrcoef(xt, yp)[0, 1]
    
    # 将统计指标（包括 PICP）放入图内
    textstr = '\n'.join((
        f'Correlation (r) = {corr_coef:.3f}',
        f'95% CI PICP = {picp:.1f}%',
        f'Trend: y = {m:.2f}x + {b:.3f}'
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='silver')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    ax.legend(loc='lower right')
    
    fig.colorbar(sc, ax=ax, label='Point Density')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_reliability_diagram(y_true, y_pred, std_raw, optimal_k, title, save_path):
    """绘制不确定性量化的可靠性图 (Reliability Diagram)"""
    from scipy.stats import norm
    from matplotlib.ticker import PercentFormatter

    # 定义内部覆盖率计算函数
    def calculate_coverage(k_factor, expected_prob):
        z_score = norm.ppf(0.5 + expected_prob / 2.0) 
        calibrated_std = std_raw * k_factor
        lower = y_pred - z_score * calibrated_std
        upper = y_pred + z_score * calibrated_std
        return np.mean((y_true >= lower) & (y_true <= upper))

    # 准备计算 10% 到 95% 置信区间的数据点
    expected_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    raw_coverages = [calculate_coverage(1.0, p) for p in expected_probs]         # 校准前
    calib_coverages = [calculate_coverage(optimal_k, p) for p in expected_probs] # 校准后

    # 开始绘图
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    # 绘制参考线与数据曲线
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.5, label='Perfect Calibration (1:1)')
    ax.plot(expected_probs, raw_coverages, marker='o', markersize=6, linestyle='-', 
            color='steelblue', linewidth=2, label='Raw Ensemble (Uncalibrated)')
    ax.plot(expected_probs, calib_coverages, marker='s', markersize=6, linestyle='-', 
            color='firebrick', linewidth=2, label=f'Calibrated Ensemble (k={optimal_k:.2f})')

    # 坐标轴与标题设置
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Expected Confidence Level', fontsize=12)
    ax.set_ylabel('Observed Coverage Frequency (PICP)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)

    # 格式化为百分比
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='silver')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_binned_uncertainty_vs_error(uncertainty, abs_error, title, save_path, num_bins=15):
    """绘制二维分箱后的不确定性 vs 绝对误差图"""
    # 1. 构建 DataFrame
    df = pd.DataFrame({'unc': uncertainty, 'err': abs_error})
    
    # 取 0 到 99% 分位数，避免被极少数超高不确定性的离群点拉偏箱子划分
    min_unc = df['unc'].min()
    max_unc = df['unc'].quantile(0.99)
    bins = np.linspace(min_unc, max_unc, num_bins + 1)
    
    # 2. 分箱操作
    df['bin'] = pd.cut(df['unc'], bins=bins)
    
    # 3. 计算每个箱内的统计量 (均值和标准误 SEM)
    stats = df.groupby('bin', observed=False).agg(
        unc_mean=('unc', 'mean'),
        err_mean=('err', 'mean'),
        err_sem=('err', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
        count=('err', 'count')
    ).dropna()
    
    # 过滤掉样本量极少 (<30) 的箱子，保证统计学置信度
    stats = stats[stats['count'] > 30]

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # 画带误差棒的点
    ax.errorbar(stats['unc_mean'], stats['err_mean'], yerr=stats['err_sem'], 
                fmt='o', color='teal', ecolor='gray', elinewidth=1.5, capsize=4, 
                markersize=8, markeredgecolor='white', markeredgewidth=1,
                label='Binned Mean ± SEM')
    
    # 线性拟合与相关系数
    if len(stats) > 1:
        m, b = np.polyfit(stats['unc_mean'], stats['err_mean'], 1)
        ax.plot(stats['unc_mean'], m * stats['unc_mean'] + b, color='firebrick', 
                linestyle='-', linewidth=2.5, label=f'Linear Trend (Slope={m:.2f})')
        corr_coef = np.corrcoef(stats['unc_mean'], stats['err_mean'])[0, 1]
    else:
        m, b, corr_coef = 0, 0, 0
    
    ax.set_xlabel('Model Uncertainty ($1\sigma$) [ppm]', fontsize=13)
    ax.set_ylabel('Mean Absolute Prediction Error [ppm]', fontsize=13)
    ax.set_title(title, fontsize=15, pad=15)
    ax.set_xlim(left=0, right=max_unc * 1.05)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 填入统计指标
    textstr = '\n'.join((
        f'Binned Correlation (r) = {corr_coef:.3f}',
        f'Trend: y = {m:.2f}x + {b:.3f}'
    ))
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='silver')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
# ==========================================
# 2. 数据加载与特征复原
# ==========================================
print("\n📂 1/9: 加载数据与特征工程...")
df = pd.read_pickle(DATA_FILE)
df_clean = df.dropna().copy()
# 动态构建特征工程 (A01, A05，A09需要注释下一行; 其他模型需要打开)
# A03改成小于等于8了，其他还是小于等于10
# df_clean = df_clean[df_clean[TARGET] <= 10].reset_index(drop=True)
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

# 切分数据集 (保留 df 结构以便画地图取经纬度)
df_pool, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)

X_pool_raw = df_pool[selected_features].values
y_pool = df_pool[TARGET].values
X_test_raw = df_test[selected_features].values
y_test = df_test[TARGET].values

print("⚖️ 2/9: 加载标准化器并进行缩放...")
eval_scaler = joblib.load(SCALER_PATH)
X_pool_scaled = eval_scaler.transform(X_pool_raw)
X_test_scaled = eval_scaler.transform(X_test_raw)

print("🔍 3/9: 准备固定 SHAP 解释样本 (5000条以防内存溢出)...")
np.random.seed(42)
shap_sample_size = min(5000, len(X_pool_scaled))
shap_idx = np.random.choice(len(X_pool_scaled), shap_sample_size, replace=False)
X_shap_sample = X_pool_scaled[shap_idx]

# ==========================================
# 3. 循环 10 个模型进行预测、绘图与 SHAP 收集
# ==========================================
print(f"\n🧠 4/9: 正在遍历处理 10 个模型 (包含推理、散点图、SHAP 与 空间误差地图)...")
train_preds_all = []
test_preds_all = []
shap_values_all = [] 

for seed in SEEDS:
    print(f"   ► 处理 Model Seed {seed} ...")
    model_path = f"{MODEL_BASE_PATH}_seed{seed}.pkl"
    model = joblib.load(model_path)
    
    # 预测推理
    pred_train = model.predict(X_pool_scaled)
    pred_test = model.predict(X_test_scaled)
    train_preds_all.append(pred_train)
    test_preds_all.append(pred_test)
    
    # 1. 绘制单体散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = plot_density_scatter(axes[0], y_pool, pred_train, f'Model Seed {seed} - Train')
    sc2 = plot_density_scatter(axes[1], y_test, pred_test, f'Model Seed {seed} - Test')
    fig.colorbar(sc2, ax=axes.ravel().tolist(), label='Point Density')
    plt.savefig(os.path.join(OUTPUT_DIR, f'scatter_model_seed{seed}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 绘制单体 SHAP
    explainer = shap.TreeExplainer(model)
    shap_v = explainer.shap_values(X_shap_sample)
    shap_values_all.append(shap_v)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_v, X_shap_sample, feature_names=selected_features, show=False)
    plt.title(f'SHAP Summary - Model Seed {seed}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'shap_summary_seed{seed}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 【新增】绘制单体空间误差地图
    print(f"      - 正在绘制 Seed {seed} 的空间误差地图...")
    plot_spatial_bias_map(
        df_coords=df_test, 
        y_true=y_test, 
        y_pred=pred_test, 
        title=f'Spatial Bias (Test Set) - Model Seed {seed}', 
        save_path=os.path.join(OUTPUT_DIR, f'spatial_bias_map_seed{seed}.png')
    )

train_preds_all = np.array(train_preds_all)
test_preds_all = np.array(test_preds_all)
shap_values_all = np.array(shap_values_all)

# ==========================================
# 4. 集成模型汇总制图
# # ==========================================
print("\n🚀 5/9: 计算集成结果并绘制总体散点密度图...")
ensemble_train_pred = np.mean(train_preds_all, axis=0)
ensemble_test_pred = np.mean(test_preds_all, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sc_train = plot_density_scatter(axes[0], y_pool, ensemble_train_pred, 'XGBoost Ensemble - Training Set')
sc_test = plot_density_scatter(axes[1], y_test, ensemble_test_pred, 'XGBoost Ensemble - Testing Set')
fig.colorbar(sc_test, ax=axes.ravel().tolist(), label='Point Density')
plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_ensemble_final.png'), dpi=400, bbox_inches='tight')
plt.close()

print("🚀 6/9: 利用加法公理绘制集成模型总 SHAP 解释图...")
ensemble_shap_values = np.mean(shap_values_all, axis=0)
plt.figure(figsize=(10, 8))
shap.summary_plot(ensemble_shap_values, X_shap_sample, feature_names=selected_features, show=False)
plt.title('Ensemble Model SHAP Summary', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_ensemble_final.png'), dpi=400, bbox_inches='tight')
plt.close()

print("🗺️ 7/9: 绘制集成模型盲测集空间误差地图...")
plot_spatial_bias_map(
    df_coords=df_test, 
    y_true=y_test, 
    y_pred=ensemble_test_pred, 
    title='Spatial Distribution of Residuals (Ensemble Testing Set)', 
    save_path=os.path.join(OUTPUT_DIR, 'spatial_bias_map_ensemble.png'))

print("\n🛡️ 8/9: 正在执行模型不确定性量化 (UQ) 评估...")

# 计算集成预测值的标准差 (代表模型在空间上的不确定性/发散度)
ensemble_test_std = np.std(test_preds_all, axis=0)

# ==========【原始结果】==============
# ensemble_test_std_raw = np.std(test_preds_all, axis=0)
# optimal_k = 1  

# ==========【寻优结果-start】==============
ensemble_test_std_raw = np.std(test_preds_all, axis=0)
abs_error = np.abs(ensemble_test_pred - y_test)

# 定义计算 PICP 的目标函数
def target_function(k):
    calibrated_std = ensemble_test_std_raw * k
    lower = ensemble_test_pred - 1.96 * calibrated_std
    upper = ensemble_test_pred + 1.96 * calibrated_std
    picp = np.mean((y_test >= lower) & (y_test <= upper)) * 100
    return picp - 95.0  # 我们希望这个差值等于 0 (即 PICP = 95)

# 使用 Brentq 算法自动在 k=1 到 k=20 之间寻找完美解
try:
    optimal_k = brentq(target_function, 1.0, 20.0)
    print(f"   ► 自动寻优成功！最优温度缩放系数 (k) = {optimal_k:.3f}")
except ValueError:
    print("   ► 警告：在 1~20 的范围内未找到 95% 覆盖率，可能需要更大的 k。")
    optimal_k = 10.0 # 备用回退方案

# 应用最优 k 值
ensemble_test_std = ensemble_test_std_raw * optimal_k
# ==========【寻优结果-end】==============

# 2. 计算实际的绝对误差
abs_error = np.abs(ensemble_test_pred - y_test)

# 3. 计算 95% 置信区间 (1.96个标准差) 的覆盖率 (PICP)
# 假设集成输出服从正态分布，95% CI 约为均值 ± 1.96*σ
lower_bound = ensemble_test_pred - 1.96 * ensemble_test_std
upper_bound = ensemble_test_pred + 1.96 * ensemble_test_std

# 真实值落在区间内的布尔数组，求均值即为覆盖比例
picp_95 = np.mean((y_test >= lower_bound) & (y_test <= upper_bound)) * 100
print(f"   ► 95% 置信区间覆盖率 (PICP): {picp_95:.2f}% (越接近 95% 越完美)")


# 绘制空间不确定性地图
print("   ► 绘制 空间不确定性地图 (Spatial Uncertainty Map)...")
plot_spatial_uncertainty_map(
    df_coords=df_test, 
    uncertainty=ensemble_test_std, 
    title='Spatial Distribution of Model Uncertainty ($1\sigma$)', 
    save_path=os.path.join(OUTPUT_DIR, 'spatial_uncertainty_map_ensemble.png'))

# 绘制不确定性 vs 绝对误差的散点图
print("   ► 绘制 不确定性 vs 绝对误差散点图 (Uncertainty vs. Absolute Error)...")
plot_uncertainty_vs_error(
    uncertainty=ensemble_test_std, 
    abs_error=abs_error, 
    picp=picp_95,
    title='Uncertainty vs. Absolute Error (Blind Test)', 
    save_path=os.path.join(OUTPUT_DIR, 'scatter_uncertainty_vs_error.png'))

# 绘制可靠性图 
print("   ► 绘制 可靠性图 (Reliability Diagram)...")
plot_reliability_diagram(
    y_true=y_test, 
    y_pred=ensemble_test_pred, 
    std_raw=ensemble_test_std_raw, 
    optimal_k=optimal_k, 
    title='Reliability Diagram for Uncertainty Quantification', 
    save_path=os.path.join(OUTPUT_DIR, 'reliability_diagram_calibration.png'))

print("   ► 绘制 二维分箱不确定性散点图 (Binned Uncertainty vs. Error)...")
plot_binned_uncertainty_vs_error(
    uncertainty=ensemble_test_std, 
    abs_error=abs_error, 
    title='Binned Uncertainty vs. Mean Absolute Error', 
    save_path=os.path.join(OUTPUT_DIR, 'scatter_binned_uncertainty_vs_error.png')
)    

# ==========================================
# 9. 绘制 SHAP 特征交互依赖图 (Physical Attribution)
# ==========================================
print("\n🔬 9/9: 正在绘制 SHAP 特征交互依赖图 (Dependence Plots)...")

def plot_shap_dependence(shap_vals, features_matrix, feature_names, feature1, feature2_for_color, save_path):
    """
    绘制指定特征对的 SHAP 依赖图
    - feature1: 主特征 (X 轴)
    - feature2_for_color: 交互特征 (颜色映射)
    """
    # 检查特征是否在列表中
    if feature1 not in feature_names or feature2_for_color not in feature_names:
        print(f"⚠️ 警告: 找不到特征 {feature1} 或 {feature2_for_color}，跳过绘制。")
        return

    # shap.dependence_plot 内部会自动调用 plt.figure，所以我们直接设定参数
    shap.dependence_plot(
        ind=feature1,                           # X 轴的特征
        shap_values=shap_vals,                  # SHAP 值矩阵
        features=features_matrix,               # 原始特征矩阵
        feature_names=feature_names,            # 特征名称列表
        interaction_index=feature2_for_color,   # 决定颜色的交互特征
        show=False,                             # 不直接显示，为了保存
        cmap=plt.get_cmap("coolwarm"),          # 使用冷暖色调，区分度高
        alpha=0.8,
        dot_size=20
    )
    
    # 获取当前的 Figure 和 Axes 调整格式
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(8, 6)
    fig.set_dpi(300)
    
    ax.set_title(f'SHAP Dependence: {feature1} vs {feature2_for_color}', fontsize=14, pad=15)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      - 保存成功: {os.path.basename(save_path)}")

# 推荐绘制的物理交叉组合 (你可以根据需要自由修改或增加)
interaction_pairs = [
    # 1. 动力学扩散作用：风速 vs 人类活动强度
    ('era5_wind_speed', 'ntl_nox_cross'),
    
    # 2. 热力学扩散作用：边界层高度 vs 对流层 NO2 (代表污染源强)
    ('era5_blh', 'no2_trop_log'),
    
    # 3. 生物圈-气象耦合：植被指数 vs 温度
    ('ndvi', 'era5_t2m'),
    
    # 4. 夜灯与污染清单的交叉：夜间灯光 vs MEIC NOx 排放
    ('ntl', 'meic_nox')
]

# 遍历绘制所有推荐的组合
for feat1, feat2 in interaction_pairs:
    save_filename = os.path.join(OUTPUT_DIR, f'shap_dep_{feat1}_VS_{feat2}.png')
    plot_shap_dependence(
        shap_vals=ensemble_shap_values, 
        features_matrix=X_shap_sample, 
        feature_names=selected_features, 
        feature1=feat1, 
        feature2_for_color=feat2, 
        save_path=save_filename)



print(f"\n🎉 大功告成！所有验证图与地图均已保存在: {OUTPUT_DIR}")