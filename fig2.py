import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, norm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.ticker import FuncFormatter, PercentFormatter

# ==========================================
# 全局设置字体与学术级样式
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 11,               
    'axes.titlesize': 14,          
    'axes.titleweight': 'bold',    
    'axes.labelsize': 12,          
    'axes.labelweight': 'normal',
    'xtick.labelsize': 11,         
    'ytick.labelsize': 11,
    'legend.fontsize': 10,         
    'legend.title_fontsize': 11,   
    'axes.linewidth': 1.5,         
    'lines.linewidth': 1.8,        
    'lines.markersize': 6,         
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
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,      
    'legend.framealpha': 0.9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',       
    'savefig.pad_inches': 0.05
})

# ==========================================
# 1. 统计绘图函数族 (用于 2x2 拼图)
# ==========================================

from matplotlib.ticker import FuncFormatter, PercentFormatter

def draw_reliability_diagram(ax, y_true, y_pred, std_raw, optimal_k, title):
    def calculate_coverage(k_factor, expected_prob):
        z_score = norm.ppf(0.5 + expected_prob / 2.0) 
        calibrated_std = std_raw * k_factor
        lower = y_pred - z_score * calibrated_std
        upper = y_pred + z_score * calibrated_std
        return np.mean((y_true >= lower) & (y_true <= upper))

    expected_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    raw_coverages = [calculate_coverage(1.0, p) for p in expected_probs]         
    calib_coverages = [calculate_coverage(optimal_k, p) for p in expected_probs] 
    calib_ce = np.mean(np.abs(np.array(calib_coverages) - expected_probs)) * 100

    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.5, label='Perfect Calibration (1:1)')
    ax.plot(expected_probs, raw_coverages, marker='o', markersize=6, linestyle='-', 
            color='steelblue', linewidth=2, label='Raw Ensemble (Uncalibrated)')
    ax.plot(expected_probs, calib_coverages, marker='s', markersize=6, linestyle='-', 
            color='firebrick', linewidth=2, label=f'Calibrated Ensemble')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Expected Confidence Level')
    ax.set_ylabel('Observed Coverage Frequency (PICP)')
    
    # 标签置于左上角
    ax.text(0.04, 0.95, title, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left', zorder=10)
    
    # 【修改点】过滤纵坐标起始刻度 0
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0) else f"{int(round(x*100))}%"))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    
    ax.grid(True, linestyle=':', alpha=0.6)
    # 图例向下移动，避免遮挡左上角标签
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.86), framealpha=0.9, edgecolor='silver')

    ax.text(0.04, 0.62, f'k = {optimal_k:.2f}\nCE = {calib_ce:.1f}%', transform=ax.transAxes, fontsize=12)
    
def draw_binned_uncertainty_vs_error(ax, uncertainty, abs_error, title, num_bins=15):
    df = pd.DataFrame({'unc': uncertainty, 'err': abs_error})
    min_unc = df['unc'].min()
    max_unc = df['unc'].quantile(0.99)
    bins = np.linspace(min_unc, max_unc, num_bins + 1)
    df['bin'] = pd.cut(df['unc'], bins=bins)
    
    stats = df.groupby('bin', observed=False).agg(
        unc_mean=('unc', 'mean'),
        err_mean=('err', 'mean'),
        err_sem=('err', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
        count=('err', 'count')
    ).dropna()
    stats = stats[stats['count'] > 30]

    ax.errorbar(stats['unc_mean'], stats['err_mean'], yerr=stats['err_sem'], 
                fmt='o', color='teal', ecolor='gray', elinewidth=1.5, capsize=4, 
                markersize=7, markeredgecolor='white', markeredgewidth=1,
                label='Binned Mean ± SEM')
    
    if len(stats) > 1:
        m, b = np.polyfit(stats['unc_mean'], stats['err_mean'], 1)
        ax.plot(stats['unc_mean'], m * stats['unc_mean'] + b, color='firebrick', 
                linestyle='-', linewidth=2, label=f'Linear Trend (Slope={m:.2f})')
        corr_coef = np.corrcoef(stats['unc_mean'], stats['err_mean'])[0, 1]
    else:
        m, b, corr_coef = 0, 0, 0
    
    ax.set_xlabel('Model Uncertainty ($1\sigma$) [ppm]')
    ax.set_ylabel('Mean Absolute Prediction Error [ppm]')
    
    # 标签置于左上角
    ax.text(0.04, 0.95, title, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left', zorder=10)
    ax.set_xlim(left=0, right=max_unc * 1.05)
    ax.set_ylim(bottom=0)
    
    # 【修改点】过滤纵坐标起始刻度 0
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0) else f"{x:.1f}"))
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    textstr = f'Binned Correlation (r) = {corr_coef:.3f}\nTrend: y = {m:.2f}x + {b:.3f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='silver')
    
    # 统计信息文本框向下移动，避免遮挡标签
    ax.text(0.04, 0.86, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)
    ax.legend(loc='lower right')

def draw_density_scatter(ax, y_true, y_pred, title, std=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mb = np.mean(y_pred - y_true)  
    r2 = r2_score(y_true, y_pred)
    n_samples = len(y_true)

    max_samples = 50000
    if len(y_true) > max_samples:
        idx = np.random.choice(len(y_true), max_samples, replace=False)
    else:
        idx = np.arange(len(y_true))
        
    xt, yp = y_true[idx], y_pred[idx]
    xy = np.vstack([xt, yp])
    z = gaussian_kde(xy)(xy)
    
    sort_idx = z.argsort()
    xt_sorted, yp_sorted, z_sorted = xt[sort_idx], yp[sort_idx], z[sort_idx]
    
    # 绘制散点
    sc = ax.scatter(xt_sorted, yp_sorted, c=z_sorted, s=5, cmap='viridis', alpha=0.9, edgecolor='none')
    
    # 绘制 1:1 参考线 (红色虚线)
    ax.plot([0, 10], [0, 10], color='#D62728', linestyle='--', linewidth=2.0, label='1:1 Line')
    
    # 【修改点 1】提前计算回归参数
    m, b = np.polyfit(xt, yp, 1)
    
    # 【修改点 2】实际绘制出线性拟合线 (深灰色实线，以便与 1:1 线区分)
    x_fit = np.array([0, 10])
    y_fit = m * x_fit + b
    ax.plot(x_fit, y_fit, color='#333333', linestyle='-', linewidth=2.0, label='Linear Fit')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    ax.set_xlabel('Observed $XCO_2$ Enhanced [ppm]')
    ax.set_ylabel('Predicted $XCO_2$ Enhanced [ppm]')
    
    # 标签置于左上角
    ax.text(0.04, 0.95, title, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left', zorder=10)
    
    # 过滤纵坐标起始刻度 0
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0) else f"{int(x)}"))
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 【修改点 3】绘制图例（此时会自动包含刚才画的 1:1 Line 和 Linear Fit）
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='silver')

    # 生成统计文本框
    sign = '+' if b >= 0 else '-'
    textstr = f'N = {n_samples}\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMB = {mb:.3f}\ny = {m:.2f}x {sign} {abs(b):.3f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='silver')
    
    # 统计信息文本框向下移动，避免遮挡标签
    ax.text(0.04, 0.86, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)

    return sc

def draw_binned_prediction_vs_relative_rmse(ax, y_true, y_pred, title, num_bins=12):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'sq_err': (y_pred - y_true) ** 2})
    
    min_pred, max_pred = df['pred'].min(), df['pred'].max()
    bins = np.linspace(min_pred, max_pred, num_bins + 1)
    df['bin'] = pd.cut(df['pred'], bins=bins)
    
    stats = df.groupby('bin', observed=False).agg(
        pred_mean=('pred', 'mean'),
        true_mean=('true', 'mean'),
        mse=('sq_err', 'mean'),
        count=('pred', 'count')
    ).dropna()
    
    stats = stats[stats['count'] > 10]
    stats['rmse'] = np.sqrt(stats['mse'])
    stats['rel_rmse'] = (stats['rmse'] / stats['true_mean']) * 100.0

    bin_widths = np.diff(bins)[0] * 0.6
    bars = ax.bar(stats['pred_mean'], stats['rmse'], width=bin_widths, 
                  color='#4C72B0', alpha=0.75, edgecolor='#333333', linewidth=1.0,
                  label='Absolute RMSE')
    
    ax.set_xlabel('Predicted $XCO_2$ Enhanced [ppm]')
    ax.set_ylabel('Absolute RMSE [ppm]', color='#4C72B0')
    ax.tick_params(axis='y', colors='#4C72B0')
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    
    # 标签置于左上角
    ax.text(0.04, 0.95, title, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left', zorder=10)

    # 【修改点】过滤左纵坐标起始刻度 0
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0) else f"{x:.1f}"))

    ax_right = ax.twinx()
    line = ax_right.plot(stats['pred_mean'], stats['rel_rmse'], 
                         marker='^', markersize=7, color='firebrick', linewidth=2.0, 
                         markeredgecolor='white', markeredgewidth=1,
                         label='Relative RMSE')
    
    ax_right.set_ylabel('Relative RMSE (%)', color='firebrick')
    ax_right.set_ylim(bottom=0)
    ax_right.set_ylim(top=ax_right.get_ylim()[1] * 1.15)
    
    # 【修改点】显式激活并硬编码强制开启右轴的刻度线与标签渲染，确保其不丢失
    ax_right.tick_params(axis='y', which='both', right=True, labelright=True, direction='in', colors='firebrick')
    
    # 【修改点】过滤右纵坐标起始刻度 0 并对其进行规范化格式化
    ax_right.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0) else f"{int(round(x))}%"))

    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, color='#B0B0B0')
    ax_right.grid(False)

    lines_labels = [ax.get_legend_handles_labels(), ax_right.get_legend_handles_labels()]
    handles = lines_labels[0][0] + lines_labels[1][0]
    labels = lines_labels[0][1] + lines_labels[1][1]
    
    # 【修改点】将子图 4 (d) 的图例移到右上角 (upper right)
    ax.legend(handles, labels, loc='upper right', framealpha=0.9, edgecolor='silver')

# ==========================================
# 2. 空间分布绘图函数族 (用于 1x2 地图)
# ==========================================

def add_map_gridlines(ax, hide_left=False):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False   
    gl.right_labels = False 
    
    # 【新增】控制是否隐藏左侧刻度
    if hide_left:
        gl.left_labels = False
        
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

def draw_spatial_map(fig, ax, df_coords, values, title, cmap, vmin, vmax, cbar_label, is_bias=False, hide_left=False):
    df_val = pd.DataFrame({'lon': df_coords['grid_lon'].values, 'lat': df_coords['grid_lat'].values, 'val': values})
    grid_val = df_val.groupby(['lon', 'lat'])['val'].mean().reset_index()
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#333333', zorder=2)
    
    world_shp_path = '/home/whdong/shapefile/world/world.shp'
    if os.path.exists(world_shp_path):
        world_borders = cfeature.ShapelyFeature(shpreader.Reader(world_shp_path).geometries(), ccrs.PlateCarree(),     
                                                facecolor='none', edgecolor='#333333', linewidth=0.8, linestyle=':')
        ax.add_feature(world_borders, zorder=2)

    china_prov_shp_path = '/home/whdong/shapefile/china/province.shp'
    if os.path.exists(china_prov_shp_path):
        try:
            import shapefile
            from shapely.geometry import shape
            with shapefile.Reader(china_prov_shp_path, encoding='gbk') as sf:
                geometries = [shape(s.__geo_interface__) for s in sf.shapes()]
            china_prov_borders = cfeature.ShapelyFeature(geometries, ccrs.PlateCarree(),
                                                         facecolor='none', edgecolor='#777777', linewidth=0.4, linestyle=':')
            ax.add_feature(china_prov_borders, zorder=2)
        except Exception:
            pass 

    sc = ax.scatter(grid_val['lon'], grid_val['lat'], c=grid_val['val'], 
                    cmap=cmap, vmin=vmin, vmax=vmax, s=10, alpha=0.9, edgecolor='none', transform=ccrs.PlateCarree(), zorder=1)
    
    add_map_gridlines(ax, hide_left=hide_left)
    ax.text(0.03, 0.97, title, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)
    
    # 颜色条置于底部
    cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.75, aspect=40, pad=0.05, extend='both')
    cbar.set_label(cbar_label, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(size=0)

# ==========================================
# 3. 版面渲染引擎：生成 2 张独立的完美组合图
# ==========================================

def create_figure1_statistical(y_test, ensemble_test_pred, ensemble_test_std_raw, ensemble_test_std, optimal_k, abs_error, save_path):
    print("🎨 正在生成 Figure 1 (2x2 统计验证组合图 - 像素级硬锁大字版)...")
    
    # 备份原有 rc 基础设置，防止影响后续的 1x2 地图
    orig_rc = plt.rcParams.copy()
    
    # 【修改点】全面单独调大 2x2 统计图组的字体，显著增强图片的学术可读性
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 13,               
        'axes.titlesize': 18,          
        'axes.labelsize': 15,          
        'xtick.labelsize': 13,         
        'ytick.labelsize': 13,
        'legend.fontsize': 11.5,         
        'legend.title_fontsize': 12.5,
        'axes.linewidth': 1.5,         
        'lines.linewidth': 2.0,        
        'lines.markersize': 7,         
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
        'grid.color': '#B0B0B0'
    })
    
    # 建立 14x14 英寸正方形画布
    fig = plt.figure(figsize=(15, 14), dpi=300)
    
    # 采用底层绝对坐标 hard code 定位。主图框宽、高设为相等的 0.36
    # 彻底杜绝排版引擎对 1:1 正方形网格完整度的破坏
    w, h = 0.36, 0.36
    left_col = 0.11     # 左列轴线左起位置
    right_col = 0.57    # 右列轴线左起位置（中间留出 0.08 的充裕距离给坐标轴文本）
    bottom_row = 0.11   # 下排轴线底部位置
    top_row = 0.52      # 上排轴线底部位置
    
    ax_a = fig.add_axes([left_col, top_row, w, h])
    ax_c = fig.add_axes([right_col, top_row, w, h])
    ax_b = fig.add_axes([left_col, bottom_row, w, h])
    ax_d = fig.add_axes([right_col, bottom_row, w, h])
    
    # (a) Reliability Diagram
    # draw_reliability_diagram(ax_a, y_test, ensemble_test_pred, ensemble_test_std_raw, optimal_k, '(a)')
    sc_scatter = draw_density_scatter(ax_a, y_test, ensemble_test_pred, '(a)')
    # 散点密度颜色条也要跟随移动到 ax_a
    cbaxes = ax_a.inset_axes([1.015, 0.0, 0.03, 1.0])
    cbar3 = fig.colorbar(sc_scatter, cax=cbaxes)
    # cbar3.set_label('Point Density', fontsize=14, labelpad=8)
    cbar3.ax.tick_params(labelsize=12, size=0)

    # (b) 右上图：Reliability Diagram
    draw_reliability_diagram(
        ax_c, y_test, ensemble_test_pred,
        ensemble_test_std_raw, optimal_k, '(b)')
    
    # (c) Binned Uncertainty vs. MAE
    draw_binned_uncertainty_vs_error(ax_b, ensemble_test_std, abs_error, '(c)')
    
    # (b) XGBoost Ensemble Testing Set与外挂 Colorbar 渲染
    # sc_scatter = draw_density_scatter(ax_c, y_test, ensemble_test_pred, '(b)')
    # cbaxes = ax_c.inset_axes([1.04, 0.0, 0.03, 1.0]) 
    # cbar3 = fig.colorbar(sc_scatter, cax=cbaxes)
    # cbar3.set_label('Point Density', fontsize=14, labelpad=8)
    # cbar3.ax.tick_params(labelsize=12, size=0)

    # (d) Binned Prediction vs Rel RMSE
    draw_binned_prediction_vs_relative_rmse(ax_d, y_test, ensemble_test_pred, '(d)')

    # 输出并保存（绝对坐标系下不使用任何 tight_layout）
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 还原环境配置，确保后面的 1x2 空间图依旧保持原有精美字号
    plt.rcParams.update(orig_rc)
    print(f"🎉 Figure 1 成功生成至: {save_path}")

def create_figure2_spatial(df_test, y_test, ensemble_test_pred, ensemble_test_std, save_path):
    print("🌍 正在生成 Figure 2 (1x2 空间分布地图)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    
    # (a) 左图：默认不隐藏刻度
    draw_spatial_map(fig, axes[0], df_test, ensemble_test_std, 
                     title='(a)', 
                     cmap='rainbow', vmin=0, vmax=2, cbar_label='Prediction Uncertainty ($1\sigma$) [ppm]', is_bias=False)
    
    # (b) 右图：传入 hide_left=True 隐藏左刻度
    bias = ensemble_test_pred - y_test
    draw_spatial_map(fig, axes[1], df_test, bias, 
                     title='(b)', 
                     cmap='RdBu_r', vmin=-2, vmax=2, cbar_label='Bias (Pred - Obs) [ppm]', is_bias=True, hide_left=True)

    plt.subplots_adjust(wspace=0.03) # 这里 0.03 的间距现在会非常紧凑且好看
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"🎉 Figure 2 成功生成至: {save_path}")

# ==========================================
# 4. 主执行流 
# ==========================================
def main():
    
    DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'
    SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
    MODEL_BASE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01' 
    OUTPUT_DIR = '/home/whdong/dl/figures/scatter_density_A01' 
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TARGET = 'xco2_enhanced'
    SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]

    print("\n📂 1/5: 加载数据与特征工程复原...")
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

    df_pool, df_test = train_test_split(df_clean, test_size=0.2, random_state=42)
    X_test_raw = df_test[selected_features].values
    y_test = df_test[TARGET].values

    print("⚖️ 2/5: 加载标准化器进行 Test Set 特征缩放...")
    eval_scaler = joblib.load(SCALER_PATH)
    X_test_scaled = eval_scaler.transform(X_test_raw)

    print(f"\n🧠 3/5: 正在并行遍历处理 10 个模型进行推理验证...")
    test_preds_all = []
    for seed in SEEDS:
        model_path = f"{MODEL_BASE_PATH}_seed{seed}.pkl"
        model = joblib.load(model_path)
        test_preds_all.append(model.predict(X_test_scaled))
    test_preds_all = np.array(test_preds_all)

    print("\n🛡️ 4/5: 执行模型集成与不确定性量化 (UQ) 参数寻优...")
    ensemble_test_pred = np.mean(test_preds_all, axis=0)
    ensemble_test_std_raw = np.std(test_preds_all, axis=0)
    abs_error = np.abs(ensemble_test_pred - y_test)

    def target_function(k):
        calibrated_std = ensemble_test_std_raw * k
        lower, upper = ensemble_test_pred - 1.96 * calibrated_std, ensemble_test_pred + 1.96 * calibrated_std
        return np.mean((y_test >= lower) & (y_test <= upper)) * 100 - 95.0 

    try:
        optimal_k = brentq(target_function, 1.0, 20.0)
        print(f"   ► 自动寻优成功！最优缩放系数 (k) = {optimal_k:.3f}")
    except ValueError:
        print("   ► 警告：在 1~20 的范围内未找到 95% 覆盖率，采用备选策略。")
        optimal_k = 10.0 
    ensemble_test_std = ensemble_test_std_raw * optimal_k

    print("\n🎨 5/5: 分阶段渲染 2 组学术拼图...")
    fig1_path = os.path.join(OUTPUT_DIR, 'Figure2A_Statistical_Validation.png')
    fig2_path = os.path.join(OUTPUT_DIR, 'Figure2B_Spatial_Distribution.png')
    
    # 生成图 1
    create_figure1_statistical(y_test, ensemble_test_pred, ensemble_test_std_raw, ensemble_test_std, optimal_k, abs_error, fig1_path)
    # 生成图 2
    create_figure2_spatial(df_test, y_test, ensemble_test_pred, ensemble_test_std, fig2_path)

if __name__ == "__main__":
    main()