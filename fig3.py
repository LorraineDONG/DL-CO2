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
# 1. 全局学术级字体与样式配置
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
    
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',       
    'savefig.pad_inches': 0.05
})

# ==========================================
# 2. 核心功能函数模块化封装
# ==========================================

def get_custom_cmap(txt_path=r'/home/whdong/WhGrYlRd.txt'):
    """加载自定义色带文件"""
    if os.path.exists(txt_path):
        clrs = np.genfromtxt(txt_path, delimiter=' ')
        return mpl.colors.ListedColormap(clrs / 255.0)
    else:
        print(f"⚠️ 未找到自定义色带文件 {txt_path}，将采用内置 'jet' 色带替代。")
        return 'jet'


def load_and_resample_data(pkl_dir, target_res=0.1):
    """批量读取指定目录下的 PKL 文件并进行网格空间重采样"""
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
                
            keep_cols = ['grid_lat', 'grid_lon', 'pred_xco2_enhanced', 'pred_uncertainty_1sigma']
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
    """
    精细化定制地图底图要素（注意：此函数需在 pcolormesh 绘图后调用，以防止范围被自动覆盖）
    """
    # 1. 强行重置并严格锁定地理显示范围：经度70-140，纬度15-55
    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
    ax.set_title(title, pad=12)
    
    # 2. 基础海岸线图层（zorder=2 确保压在数据网格之上）
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#333333', zorder=2)

    # 3. 世界国家边界 (国界线)
    world_shp_path = '/home/whdong/shapefile/world/world.shp'
    if os.path.exists(world_shp_path):
        world_borders = cfeature.ShapelyFeature(
            shpreader.Reader(world_shp_path).geometries(),
            ccrs.PlateCarree(),     
            facecolor='none',       
            edgecolor='#333333',    
            linewidth=0.8,          
            linestyle=':'           
        )
        ax.add_feature(world_borders, zorder=2)

    # 4. 中国省级边界 (省界线)
    china_prov_shp_path = '/home/whdong/shapefile/china/province.shp'
    if os.path.exists(china_prov_shp_path):
        china_prov_borders = cfeature.ShapelyFeature(
            shpreader.Reader(china_prov_shp_path, encoding='gbk').geometries(),
            ccrs.PlateCarree(),
            facecolor='none',       
            edgecolor='#777777',    
            linewidth=0.4,          
            linestyle=':'
        )
        ax.add_feature(china_prov_borders, zorder=2)

    # 5. 经纬度网格线与显式刻度控制
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.8, color='gray', alpha=0.4, linestyle='--')
    
    gl.top_labels = False   
    gl.right_labels = False 
    gl.left_labels = show_left
    gl.bottom_labels = show_bottom
    
    # 显式设定经纬度刻度线的显示数值，防止因范围改变导致刻度杂乱
    gl.xlocator = mticker.FixedLocator([75, 85, 95, 105, 115, 125, 135, 140])
    gl.ylocator = mticker.FixedLocator([10, 15, 20, 25, 30, 35, 40, 45, 50])
    
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}


def plot_annual_mean_1x2(df_all, cmap, output_dir, version, target_res=0.1):
    """
    绘制年均全景分析图：浓度增强量 (左图) + 观测代表性误差 (右图)
    """
    print("🎨 正在构建高保真年均 1x2 拼图...")
    
    df_annual = df_all.groupby(['lat_resampled', 'lon_resampled']).mean(numeric_only=True).reset_index()
    
    map_matrix_enh = df_annual.pivot(index='lat_resampled', columns='lon_resampled', values='pred_xco2_enhanced')
    map_matrix_unc = df_annual.pivot(index='lat_resampled', columns='lon_resampled', values='pred_uncertainty_1sigma')
    
    lons = map_matrix_enh.columns.values
    lats = map_matrix_enh.index.values
    Z_enh = map_matrix_enh.values
    Z_unc = map_matrix_unc.values

    # 定义标准学术画幅比例
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # ------------------ 左图：预测浓度增强量 ------------------
    v1, v2 = 1.0, 5.0  
    # 先绘制网格数据层 (zorder=1)
    sc1 = axes[0].pcolormesh(lons, lats, Z_enh, cmap=cmap, transform=ccrs.PlateCarree(), 
                             vmin=v1, vmax=v2, shading='auto', zorder=1)
    
    # 【核心修正】在 pcolormesh 执行后调用格式化函数，确保指定的 70-140, 15-55 经纬度完美生效
    format_academic_map(axes[0], title=r'(a) Annual Mean $\Delta$XCO$_2$', show_left=True, show_bottom=True)
    
    cb1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', pad=0.09, 
                       shrink=0.95, aspect=40, extend='both')
    cb1.set_label('Predicted $\Delta$XCO$_2$ (ppm)', fontsize=12, fontweight='bold')
    cb1.ax.tick_params(size=0)  

    # ------------------ 右图：模型观测/不确定性误差 ------------------
    v3 = np.percentile(df_annual['pred_uncertainty_1sigma'], 2)
    v4 = np.percentile(df_annual['pred_uncertainty_1sigma'], 98)
    
    # 先绘制数据层
    sc2 = axes[1].pcolormesh(lons, lats, Z_unc, cmap='Spectral_r', transform=ccrs.PlateCarree(), 
                             vmin=v3, vmax=v4, shading='auto', zorder=1)
    
    # 【核心修正】在右侧子图绘制完数据后，调用格式化函数，并将 show_left 设为 True 打开左轴刻度显示
    format_academic_map(axes[1], title=r'(b) Representation Error', show_left=False, show_bottom=True)
    
    # 【修改提示】同步将右侧颜色条高度变薄 (fraction=0.02)
    cb2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', pad=0.09, 
                       shrink=0.95, aspect=40, extend='both')
    cb2.set_label('Observation Error / Uncertainty (ppm)', fontsize=12, fontweight='bold')
    cb2.ax.tick_params(size=0)

    # ------------------ 拼图整饰与落盘 ------------------
    plt.subplots_adjust(wspace=0.05)  
    
    save_filename = f'FIG3-Annual_Mean_Map.png'
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 高保真学术年均网格图（范围：70-140E, 15-55N）已成功导出至: {save_path}")


# ==========================================
# 3. 生产流主执行引擎
# ==========================================
def main():
    ACTIVE_VERSION = 'A01'
    PKL_DIR = f'/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}'
    OUTPUT_DIR = f'/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}/figures'
    TARGET_RES = 0.1  
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    custom_cmap = get_custom_cmap()
    df_all = load_and_resample_data(pkl_dir=PKL_DIR, target_res=TARGET_RES)
    plot_annual_mean_1x2(df_all=df_all, cmap=custom_cmap, output_dir=OUTPUT_DIR, 
                         version=ACTIVE_VERSION, target_res=TARGET_RES)

if __name__ == "__main__":
    main()