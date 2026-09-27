import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

cmap = 'WhGrYlRd'
if cmap == 'WhGrYlRd':
    clrs = np.genfromtxt(r'/home/whdong/WhGrYlRd.txt', delimiter=' ')
    cmap = mpl.colors.ListedColormap(clrs / 255.0)

# ==========================================
# 0. 路径与全局配置
# ==========================================
# 指向你的大批量预测结果所在的子目录 (请核对模型版本，如 A07)
ACTIVE_VERSION = 'A01'
PKL_DIR = f'/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}'
OUTPUT_DIR = f'/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🚀 核心控制台：你想要画哪些图？
PLOT_TASKS = ['annual', 'seasonal', 'monthly']

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def get_season(month):
    """气象学标准季节划分"""
    if month in [3, 4, 5]: return 'Spring (MAM)'
    elif month in [6, 7, 8]: return 'Summer (JJA)'
    elif month in [9, 10, 11]: return 'Autumn (SON)'
    else: return 'Winter (DJF)'

def format_map(ax, title, show_left=True, show_bottom=True):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8)
    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree()) # 中国区域
    ax.set_title(title, fontsize=16, pad=10, fontweight='bold')
    
    # 绘制网格线与刻度
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
    gl.top_labels = False   # 永远关闭顶部刻度
    gl.right_labels = False # 永远关闭右侧刻度
    
    gl.left_labels = show_left
    gl.bottom_labels = show_bottom
    
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

# ==========================================
# 1. 批量读取 PKL 并打上时间标签
# ==========================================
print(f"📂 正在读取 {PKL_DIR} 中的 .pkl 文件...")
pkl_files = glob.glob(os.path.join(PKL_DIR, "pred_0.1deg_*.pkl"))
pkl_files.sort()

if not pkl_files:
    raise FileNotFoundError("没有找到任何 .pkl 文件，请检查路径！")

df_list = []
for file in pkl_files:
    try:
        # 直接读取 Pickle 文件
        df_day = pd.read_pickle(file)
        
        # 确保 date 列为 datetime 格式，提取月份和季节
        if not np.issubdtype(df_day['date'].dtype, np.datetime64):
            df_day['date'] = pd.to_datetime(df_day['date'])
            
        df_day['month'] = df_day['date'].dt.month
        df_day['season'] = df_day['month'].apply(get_season)
        
        # 只保留画图需要的核心列
        keep_cols = ['grid_lat', 'grid_lon', 'pred_xco2_enhanced', 'pred_uncertainty_1sigma', 'month', 'season']
        df_list.append(df_day[keep_cols])
        
    except Exception as e:
        print(f"  ⚠️ 读取 {os.path.basename(file)} 出错: {e}")

df_all = pd.concat(df_list, ignore_index=True)
print(f"✅ 成功加载全量数据，共包含 {len(df_all)} 个有效网格点观测。")
# =================全局设定目标分辨率 =================
TARGET_RES = 0.1  # 可随时修改为 0.1, 0.25, 0.5 等

print(f"📐 正在将数据统一重采样至 {TARGET_RES}° 分辨率...")
# 计算新的粗网格中心坐标
df_all['lat_resampled'] = np.floor(df_all['grid_lat'] / TARGET_RES) * TARGET_RES + TARGET_RES / 2.0
df_all['lon_resampled'] = np.floor(df_all['grid_lon'] / TARGET_RES) * TARGET_RES + TARGET_RES / 2.0
# ==========================================
# 2. 绘图
# ==========================================

# ------------------------------------------
# 任务 A: 年均制图 (1x2 拼图: 浓度 + 观测误差)
# ------------------------------------------
if 'annual' in PLOT_TASKS:

    df_annual = df_all.groupby(['lat_resampled', 'lon_resampled']).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    
    map_matrix_enh = df_annual.pivot(index='lat_resampled', columns='lon_resampled', values='pred_xco2_enhanced')
    map_matrix_unc = df_annual.pivot(index='lat_resampled', columns='lon_resampled', values='pred_uncertainty_1sigma')
    
    lons = map_matrix_enh.columns.values
    lats = map_matrix_enh.index.values
    Z_enh = map_matrix_enh.values
    Z_unc = map_matrix_unc.values

    # 左图：预测浓度增强量
    format_map(axes[0], f'(a) Annual Mean $\Delta$XCO$_2$', show_left=True, show_bottom=True)
    # v1, v2 = np.percentile(df_annual['pred_xco2_enhanced'], 2), np.percentile(df_annual['pred_xco2_enhanced'], 98)
    v1 = 1.0
    v2 = 5.0
    sc1 = axes[0].pcolormesh(lons, lats, Z_enh, cmap=cmap, 
                             transform=ccrs.PlateCarree(), vmin=v1, vmax=v2, shading='auto')
    cb1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', pad=0.08, fraction=0.04)
    cb1.set_label('Predicted $\Delta$XCO$_2$ (ppm)', fontsize=13, fontweight='bold')

    # 右图：模型观测误差
    format_map(axes[1], f'(b) Representation Error', show_left=False, show_bottom=True)
    v3, v4 = np.percentile(df_annual['pred_uncertainty_1sigma'], 2), np.percentile(df_annual['pred_uncertainty_1sigma'], 98)
    
    sc2 = axes[1].pcolormesh(lons, lats, Z_unc, cmap='Spectral_r', 
                             transform=ccrs.PlateCarree(), vmin=v3, vmax=v4, shading='auto')
    cb2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', pad=0.08, fraction=0.04)
    cb2.set_label('Observation Error / Uncertainty (ppm)', fontsize=13, fontweight='bold')

    # 保存图片
    save_path = os.path.join(OUTPUT_DIR, f'{ACTIVE_VERSION}_Annual_Mean_Map_{TARGET_RES}deg.png')
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> ✅ 年均网格图已保存至: {save_path}")

# ------------------------------------------
# 任务 B: 季均制图 (2x2 拼图)
# ------------------------------------------
if 'seasonal' in PLOT_TASKS:
    df_season = df_all.groupby(['season', 'lat_resampled', 'lon_resampled']).mean(numeric_only=True).reset_index()
    
    # vmin, vmax = np.percentile(df_season['pred_xco2_enhanced'], 2), np.percentile(df_season['pred_xco2_enhanced'], 98)
    vmin = 0.0   
    vmax = 6.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    seasons = ['Spring (MAM)', 'Summer (JJA)', 'Autumn (SON)', 'Winter (DJF)']
    labels = ['(a)', '(b)', '(c)', '(d)']
    sc = None
    
    for i, season in enumerate(seasons):
        df_sub = df_season[df_season['season'] == season]
        
        # 核心逻辑：2x2 拼图的位置判断
        # 如果 i 是 0 或 2 (偶数)，则在最左列
        is_left = (i % 2 == 0)
        # 如果 i 是 2 或 3 (大于等于2)，则在最下行
        is_bottom = (i >= 2)
        
        format_map(axes[i], f"{labels[i]} {season}", show_left=is_left, show_bottom=is_bottom)
        
        if not df_sub.empty:
            map_matrix = df_sub.pivot(index='lat_resampled', columns='lon_resampled', values='pred_xco2_enhanced')
            sc = axes[i].pcolormesh(map_matrix.columns.values, map_matrix.index.values, map_matrix.values, 
                                    cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, shading='auto')
            
    fig.subplots_adjust(bottom=0.08, hspace=0.15, wspace=0.05)
    cbar_ax = fig.add_axes([0.2, 0.03, 0.6, 0.02])
    cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Predicted $\Delta$XCO$_2$ (ppm)', fontsize=13, fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, f'{ACTIVE_VERSION}_Seasonal_Mean_Map_{TARGET_RES}deg.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> ✅ 季均网格图已保存至: {save_path}")

# ------------------------------------------
# 任务 C: 月均制图 (3x4 拼图)
# ------------------------------------------
if 'monthly' in PLOT_TASKS:
    df_month = df_all.groupby(['month', 'lat_resampled', 'lon_resampled']).mean(numeric_only=True).reset_index()
    
    # vmin, vmax = np.percentile(df_month['pred_xco2_enhanced'], 2), np.percentile(df_month['pred_xco2_enhanced'], 98)
    vmin = 0.0   
    vmax = 6.0
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sc = None
    
    for i in range(12):
        month_val = i + 1
        df_sub = df_month[df_month['month'] == month_val]

        # 共 4 列，如果 i 是 0, 4, 8，则在最左列
        is_left = (i % 4 == 0)
        # 共 12 个图，如果 i 是 8, 9, 10, 11 (最后4个)，则在最下行
        is_bottom = (i >= 8)
        
        format_map(axes[i], f"{months[i]}", show_left=is_left, show_bottom=is_bottom)
        axes[i].tick_params(labelsize=8)
        
        if not df_sub.empty:
            map_matrix = df_sub.pivot(index='lat_resampled', columns='lon_resampled', values='pred_xco2_enhanced')
            sc = axes[i].pcolormesh(map_matrix.columns.values, map_matrix.index.values, map_matrix.values, 
                                    cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, shading='auto')
            
    fig.subplots_adjust(bottom=0.08, hspace=0.1, wspace=0.05)
    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.02])
    cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Predicted $\Delta$XCO$_2$ (ppm)', fontsize=14, fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, f'{ACTIVE_VERSION}_Monthly_Mean_Map_{TARGET_RES}deg.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> ✅ 月均网格图已保存至: {save_path}")

print("🎉 所有绘图任务已执行完毕！")