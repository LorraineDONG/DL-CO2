import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ==========================================
# 1. 全局学术级字体与样式配置 (完全继承 fig3.txt)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 11,               
    'axes.titlesize': 13,          
    'axes.titleweight': 'bold',    
    'axes.labelsize': 11,          
    'xtick.labelsize': 10,         
    'ytick.labelsize': 10,
    'legend.fontsize': 10,         
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

def get_season(month):
    """气象学标准季节划分 (继承自参考代码)"""
    if month in [3, 4, 5]: return 'Spring (MAM)'
    elif month in [6, 7, 8]: return 'Summer (JJA)'
    elif month in [9, 10, 11]: return 'Autumn (SON)'
    else: return 'Winter (DJF)'

def get_custom_cmap(txt_path=r'/home/whdong/WhGrYlRd.txt'):
    """加载自定义色带文件 (继承自 fig3.txt)"""
    if os.path.exists(txt_path):
        clrs = np.genfromtxt(txt_path, delimiter=' ')
        return mpl.colors.ListedColormap(clrs / 255.0)
    else:
        print(f"⚠️ 未找到自定义色带文件 {txt_path}，将采用内置 'jet' 色带替代。")
        return 'jet'

def load_and_resample_data(pkl_dir, target_res=0.1):
    """批量读取 PKL 文件并进行网格空间重采样，同时打上月度和季节标签"""
    print(f"📂 正在读取 {pkl_dir} 中的 .pkl 文件...")
    pkl_files = glob.glob(os.path.join(pkl_dir, "pred_0.1deg_*.pkl"))
    pkl_files.sort()

    if not pkl_files:
        raise FileNotFoundError(f"在路径 {pkl_dir} 下没有找到任何 .pkl 文件，请检查路径！")

    df_list = []
    for file in pkl_files:
        try:
            df_day = pd.read_pickle(file)
            if not np.issubdtype(df_day['date'].dtype, np.datetime64):
                df_day['date'] = pd.to_datetime(df_day['date'])
                
            # 动态提取时间维度标签 (合并自参考代码与fig3)
            df_day['month'] = df_day['date'].dt.month
            df_day['season'] = df_day['month'].apply(get_season)
            
            keep_cols = ['grid_lat', 'grid_lon', 'pred_xco2_enhanced', 'pred_uncertainty_1sigma', 'month', 'season']
            df_list.append(df_day[keep_cols])
        except Exception as e:
            print(f"  ⚠️ 读取 {os.path.basename(file)} 出错: {e}")

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"✅ 成功加载全量数据，共包含 {len(df_all)} 个原始观测记录。")
    
    print(f"📐 正在将数据统一空间重采样至 {target_res}° 分辨率...")
    df_all['lat_resampled'] = np.floor(df_all['grid_lat'] / target_res) * target_res + target_res / 2.0
    df_all['lon_resampled'] = np.floor(df_all['grid_lon'] / target_res) * target_res + target_res / 2.0
    
    return df_all

def format_academic_map(ax, title, show_left=True, show_bottom=True):
    """精细化定制地图底图要素 (严格继承自 fig3.txt 的地理对齐规范)"""
    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
    ax.text(0.02, 0.96, title, transform=ax.transAxes, 
            fontsize=11, fontweight='bold', color='black',
            va='top', ha='left', zorder=3,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2))
    
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#333333', zorder=2)

    # 世界国家边界
    world_shp_path = '/home/whdong/shapefile/world/world.shp'
    if os.path.exists(world_shp_path):
        world_borders = cfeature.ShapelyFeature(
            shpreader.Reader(world_shp_path).geometries(),
            ccrs.PlateCarree(), facecolor='none', edgecolor='#333333', linewidth=0.8, linestyle=':'
        )
        ax.add_feature(world_borders, zorder=2)

    # 中国省级边界
    china_prov_shp_path = '/home/whdong/shapefile/china/province.shp'
    if os.path.exists(china_prov_shp_path):
        china_prov_borders = cfeature.ShapelyFeature(
            shpreader.Reader(china_prov_shp_path, encoding='gbk').geometries(),
            ccrs.PlateCarree(), facecolor='none', edgecolor='#777777', linewidth=0.4, linestyle=':'
        )
        ax.add_feature(china_prov_borders, zorder=2)

    # 经纬度网格线刻度控制
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.8, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False   
    gl.right_labels = False 
    gl.left_labels = show_left
    gl.bottom_labels = show_bottom
    
    gl.xlocator = mticker.FixedLocator([75, 95, 115, 135])
    gl.ylocator = mticker.FixedLocator([20, 30, 40, 50])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 9, 'color': 'black'}
    gl.ylabel_style = {'size': 9, 'color': 'black'}

# ==========================================
# 4. 新增核心绘图模块：高保真季均制图 (2x2 结构)
# ==========================================
def plot_seasonal_means_2x2(df_all, cmap, output_dir, version, target_res=0.1):
    """分别绘制 2x2 的季均 XCO2en 增强量图与季均不确定性误差图"""
    print("🎨 正在构建高保真季均多面板图序列...")
    
    # 按照季节分组计算格点均值
    df_season = df_all.groupby(['season', 'lat_resampled', 'lon_resampled']).mean(numeric_only=True).reset_index()
    seasons = ['Spring (MAM)', 'Summer (JJA)', 'Autumn (SON)', 'Winter (DJF)']
    panels = ['(a)', '(b)', '(c)', '(d)']

    # 确定整体不确定性色带的上下限截断 (基于 2% 和 98% 分位数，对齐 fig3)
    v_unc_min = np.percentile(df_season['pred_uncertainty_1sigma'], 2)
    v_unc_max = np.percentile(df_season['pred_uncertainty_1sigma'], 98)

    # ------------------ 任务 1: 季均 XCO2 增强量 (2x2) ------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    sc_enh = None

    for i, season in enumerate(seasons):
        df_sub = df_season[df_season['season'] == season]
        is_left = (i % 2 == 0)
        is_bottom = (i >= 2)
        
        if not df_sub.empty:
            matrix = df_sub.pivot(index='lat_resampled', columns='lon_resampled', values='pred_xco2_enhanced')
            sc_enh = axes[i].pcolormesh(matrix.columns.values, matrix.index.values, matrix.values, 
                                        cmap=cmap, transform=ccrs.PlateCarree(), vmin=1.0, vmax=5.0, shading='auto', zorder=1)
        format_academic_map(axes[i], title=f"{panels[i]} {season}", show_left=is_left, show_bottom=is_bottom)

    fig.subplots_adjust(bottom=0.12, hspace=0.15, wspace=0.05)
    cbar_ax1 = fig.add_axes([0.25, 0.05, 0.5, 0.02]) # 底部精细化色带控制
    cb1 = fig.colorbar(sc_enh, cax=cbar_ax1, orientation='horizontal', extend='both')
    cb1.set_label('Seasonal Mean $\Delta$XCO$_2$ (ppm)', fontsize=12, fontweight='bold')
    cb1.ax.tick_params(size=0)
    
    path_enh = os.path.join(output_dir, f'FIG5-Seasonal_Mean_XCO2.png')
    plt.savefig(path_enh, bbox_inches='tight')
    plt.close()

    # ------------------ 任务 2: 季均不确定性误差 (2x2) ------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    sc_unc = None

    for i, season in enumerate(seasons):
        df_sub = df_season[df_season['season'] == season]
        is_left = (i % 2 == 0)
        is_bottom = (i >= 2)
        
        if not df_sub.empty:
            matrix = df_sub.pivot(index='lat_resampled', columns='lon_resampled', values='pred_uncertainty_1sigma')
            sc_unc = axes[i].pcolormesh(matrix.columns.values, matrix.index.values, matrix.values, 
                                        cmap='Spectral_r', transform=ccrs.PlateCarree(), vmin=v_unc_min, vmax=v_unc_max, shading='auto', zorder=1)
        format_academic_map(axes[i], title=f"{panels[i]} {season}", show_left=is_left, show_bottom=is_bottom)

    fig.subplots_adjust(bottom=0.12, hspace=0.15, wspace=0.05)
    cbar_ax2 = fig.add_axes([0.25, 0.05, 0.5, 0.02])
    cb2 = fig.colorbar(sc_unc, cax=cbar_ax2, orientation='horizontal', extend='both')
    cb2.set_label('Seasonal Representation Error [ppm]', fontsize=12, fontweight='bold')
    cb2.ax.tick_params(size=0)

    path_unc = os.path.join(output_dir, f'FIG5-Seasonal_Representation_Error.png')
    plt.savefig(path_unc, bbox_inches='tight')
    plt.close()
    print(f"✅ 季均双序列网格图成功导出至 {output_dir}")

# ==========================================
# 5. 新增核心绘图模块：高保真月均制图 (3x4 结构)
# ==========================================
def plot_monthly_means_3x4(df_all, cmap, output_dir, version, target_res=0.1):
    """分别绘制 3x4 的月均 XCO2en 增强量图与月均不确定性误差图"""
    print("🎨 正在构建高保真月均多面板图序列...")
    
    df_month = df_all.groupby(['month', 'lat_resampled', 'lon_resampled']).mean(numeric_only=True).reset_index()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    v_unc_min = np.percentile(df_month['pred_uncertainty_1sigma'], 2)
    v_unc_max = np.percentile(df_month['pred_uncertainty_1sigma'], 98)

    # ------------------ 任务 1: 月均 XCO2 增强量 (3x4) ------------------
    fig, axes = plt.subplots(3, 4, figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    sc_enh = None

    for i in range(12):
        month_val = i + 1
        df_sub = df_month[df_month['month'] == month_val]
        is_left = (i % 4 == 0)
        is_bottom = (i >= 8)
        
        if not df_sub.empty:
            matrix = df_sub.pivot(index='lat_resampled', columns='lon_resampled', values='pred_xco2_enhanced')
            sc_enh = axes[i].pcolormesh(matrix.columns.values, matrix.index.values, matrix.values, 
                                        cmap=cmap, transform=ccrs.PlateCarree(), vmin=1.0, vmax=5.0, shading='auto', zorder=1)
        format_academic_map(axes[i], title=months[i], show_left=is_left, show_bottom=is_bottom)

    fig.subplots_adjust(bottom=0.10, hspace=0.12, wspace=0.04)
    cbar_ax1 = fig.add_axes([0.3, 0.04, 0.4, 0.015])
    cb1 = fig.colorbar(sc_enh, cax=cbar_ax1, orientation='horizontal', extend='both')
    cb1.set_label('Monthly Mean $\Delta$XCO$_2$ (ppm)', fontsize=13, fontweight='bold')
    cb1.ax.tick_params(size=0)

    path_enh = os.path.join(output_dir, f'FIG5-Monthly_Mean_XCO2.png')
    plt.savefig(path_enh, bbox_inches='tight')
    plt.close()

    # ------------------ 任务 2: 月均不确定性误差 (3x4) ------------------
    fig, axes = plt.subplots(3, 4, figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    sc_unc = None

    for i in range(12):
        month_val = i + 1
        df_sub = df_month[df_month['month'] == month_val]
        is_left = (i % 4 == 0)
        is_bottom = (i >= 8)
        
        if not df_sub.empty:
            matrix = df_sub.pivot(index='lat_resampled', columns='lon_resampled', values='pred_uncertainty_1sigma')
            sc_unc = axes[i].pcolormesh(matrix.columns.values, matrix.index.values, matrix.values, 
                                        cmap='Spectral_r', transform=ccrs.PlateCarree(), vmin=v_unc_min, vmax=v_unc_max, shading='auto', zorder=1)
        format_academic_map(axes[i], title=f"{months[i]}", show_left=is_left, show_bottom=is_bottom)

    fig.subplots_adjust(bottom=0.10, hspace=0.12, wspace=0.04)
    cbar_ax2 = fig.add_axes([0.3, 0.04, 0.4, 0.015])
    cb2 = fig.colorbar(sc_unc, cax=cbar_ax2, orientation='horizontal', extend='both')
    cb2.set_label('Monthly Representation Error [ppm]', fontsize=13, fontweight='bold')
    cb2.ax.tick_params(size=0)

    path_unc = os.path.join(output_dir, f'FIG5-Monthly_Representation_Error.png')
    plt.savefig(path_unc, bbox_inches='tight')
    plt.close()
    print(f"✅ 月均双序列网格图成功导出至 {output_dir}")

def plot_city_monthly_pipeline(df_grid_output, input_dir, figure_save_path):
    """
    [新增全自动管线] 利用主流程内存中已有的输出网格数据，结合独立提取的输入层，
    直接无缝产出包含武汉市的高规格期刊时序对比图。
    """
    import re
    
    # 1. 定义目标城市与严谨的色彩美学体系
    TARGET_CITIES = {
        'Beijing': {'lat': 39.9, 'lon': 116.4, 'color': '#9B59B6'},   # 紫色
        'Shanghai': {'lat': 31.2, 'lon': 121.5, 'color': '#34495E'},  # 深蓝/灰
        'Wuhan': {'lat': 30.6, 'lon': 114.3, 'color': '#E67E22'},     # 学术暖橙
        'Guangzhou': {'lat': 23.1, 'lon': 113.3, 'color': '#A2B836'}  # 橄榄绿
    }
    
    # 2. 从输入目录提取 NO2 数据 (由于输入层未在主流程载入，需独立读取)
    print(f"📂 正在从 {input_dir} 独立提取输入层 no2_trop 的城市序列...")
    input_files = glob.glob(os.path.join(input_dir, "post_data_*.pkl"))
    input_files.sort()
    
    if not input_files:
        print(f"⚠️ 未在 {input_dir} 下找到输入特征文件，子图(a)将跳过。")
        df_input = pd.DataFrame()
    else:
        df_first = pd.read_pickle(input_files[0])
        city_exact_grids = {}
        for city, coords in TARGET_CITIES.items():
            dist = (df_first['grid_lat'] - coords['lat'])**2 + (df_first['grid_lon'] - coords['lon'])**2
            nearest_idx = dist.idxmin()
            city_exact_grids[city] = {
                'lat': df_first.loc[nearest_idx, 'grid_lat'],
                'lon': df_first.loc[nearest_idx, 'grid_lon']
            }
        
        input_records = []
        for f in input_files:
            match = re.search(r'(\d{8})', os.path.basename(f))
            if not match: continue
            month = int(match.group(1)[4:6])
            try:
                df = pd.read_pickle(f)
                for city, grid in city_exact_grids.items():
                    subset = df[(df['grid_lat'] == grid['lat']) & (df['grid_lon'] == grid['lon'])]
                    if not subset.empty and pd.notna(subset['no2_trop'].values[0]):
                        input_records.append({'City': city, 'Month': month, 'no2_trop': subset['no2_trop'].values[0]})
            except: pass
        df_input = pd.DataFrame(input_records).groupby(['City', 'Month'])['no2_trop'].agg(['mean', 'std']).reset_index()

    # 3. 【极速优化】直接从内存的 df_grid_output 中过滤输出层，免去重复读取磁盘
    print("⚡ 正在从内存数据流中极速检索城市的 XCO2 预测序列...")
    output_records = []
    for city, grid in city_exact_grids.items():
        # 直接利用前面算好的精确格点对现成的网格大 DataFrame 进行内存切片
        subset = df_grid_output[(df_grid_output['grid_lat'] == grid['lat']) & (df_grid_output['grid_lon'] == grid['lon'])]
        if not subset.empty:
            for _, row in subset.iterrows():
                output_records.append({'City': city, 'Month': row['month'], 'pred_xco2_enhanced': row['pred_xco2_enhanced']})
    df_output = pd.DataFrame(output_records).groupby(['City', 'Month'])['pred_xco2_enhanced'].agg(['mean', 'std']).reset_index()

    # 4. 开始绘制高规格双面板折线图
    print("🗺️ 正在生成时间序列曲线拼图...")
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    months_num = np.arange(1, 13)
    months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def plot_panel(ax, df_data, target_name, title):
        if df_data.empty: return
        for city in TARGET_CITIES.keys():
            color = TARGET_CITIES[city]['color']
            city_data = df_data[df_data['City'] == city].sort_values('Month')
            if city_data.empty: continue
            
            x = city_data['Month'].values
            y_mean = city_data['mean'].values
            y_std = city_data['std'].values
            
            ax.scatter(x, y_mean, color=color, s=40, label=city, zorder=3)
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.12, zorder=1)
            try:
                poly_coeffs = np.polyfit(x, y_mean, 3)
                x_smooth = np.linspace(1, 12, 100)
                ax.plot(x_smooth, np.polyval(poly_coeffs, x_smooth), color=color, linewidth=2.5, zorder=2)
            except:
                ax.plot(x, y_mean, color=color, linewidth=2.5, zorder=2)
                
        ax.set_title(title, fontsize=15, fontweight='bold', loc='left', pad=12)
        ax.set_ylabel(target_name, fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.grid(True, linestyle=':', alpha=0.3, color='gray')

    plot_panel(axes[0], df_input, 'Tropospheric NO$_2$ Column', '(a) Monthly NO$_2$ Series')
    plot_panel(axes[1], df_output, 'Predicted $\Delta$XCO$_2$ [ppm]', '(b) Predicted XCO$_2$ Enhancement')
    
    axes[1].set_xticks(months_num)
    axes[1].set_xticklabels(months_labels, fontsize=11)
    
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=4, fontsize=11, frameon=False)
    axes[0].legend(loc='upper center', ncol=4, fontsize=13, frameon=True, facecolor='white', framealpha=0.85, edgecolor='none')
    
    os.makedirs(os.path.dirname(figure_save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(figure_save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 [自动化集成成功] 城市月时序图已顺利落盘保存至:\n{figure_save_path}")
    plt.close()

# ==========================================
# 6. 生产流控制台核心主执行引擎
# ==========================================
def main():
    ACTIVE_VERSION = 'A01'
    PKL_DIR = f'/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}/pkl_files'
    OUTPUT_DIR = f'/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}/figures'
    TARGET_RES = 0.1  # 空间重采样网格分辨率
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 载入自定义学术色带与原始序列数据库
    custom_cmap = get_custom_cmap()
    df_all = load_and_resample_data(pkl_dir=PKL_DIR, target_res=TARGET_RES)
    
    # 🚀 【核心任务控制区】可以根据需要随时放开、关闭特定的出图频率任务
    # 任务 A: 运行季均拼图生产线 (输出 2张高分拼图)
    # plot_seasonal_means_2x2(df_all=df_all, cmap=custom_cmap, output_dir=OUTPUT_DIR, 
    #                         version=ACTIVE_VERSION, target_res=TARGET_RES)
    
    # 任务 B: 运行月均拼图生产线 (输出 2张高分拼图)
    # plot_monthly_means_3x4(df_all=df_all, cmap=custom_cmap, output_dir=OUTPUT_DIR, 
    #                        version=ACTIVE_VERSION, target_res=TARGET_RES)

   
    # 任务 C: 运行城市月均时间序列折线图生产线
    # 1. 补充定义输入特征文件夹路径 
    INPUT_FEATURES_DIR = '/home/whdong/dl/ML-prediction_input_data'
    CITY_SAVE_PATH = os.path.join(OUTPUT_DIR, 'FIG5-City_Monthly_TimeSeries.png')

    # 2. 直接无缝调用，将 df_all 传入作为数据源
    plot_city_monthly_pipeline(df_grid_output=df_all, input_dir=INPUT_FEATURES_DIR, figure_save_path=CITY_SAVE_PATH)

if __name__ == "__main__":
    main()