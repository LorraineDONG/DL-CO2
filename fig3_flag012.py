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


def load_and_resample_data(pkl_dir, target_res=0.1, load_flag=True):
    """批量读取指定目录下的 PKL 文件并进行网格空间重采样"""
    print(f"📂 正在读取 {pkl_dir} 中的 .pkl 文件...")
    pkl_files = glob.glob(os.path.join(pkl_dir, "pred_0.1deg_*.pkl"))
    pkl_files.sort()
    print(f"  搜索路径: {os.path.join(pkl_dir, 'pred_0.1deg_*.pkl')}")
    print(f"  找到文件数: {len(pkl_files)}")

    if not pkl_files:
        raise FileNotFoundError(f"在路径 {pkl_dir} 下没有找到任何 .pkl 文件，请检查路径！")

    df_list = []
    for file in pkl_files:
        try:
            df_day = pd.read_pickle(file)
            if not np.issubdtype(df_day['date'].dtype, np.datetime64):
                df_day['date'] = pd.to_datetime(df_day['date'])
                
            keep_cols = ['grid_lat', 'grid_lon', 'pred_xco2_enhanced', 'pred_uncertainty_1sigma']
            try:
                qc_cols = keep_cols + ['quality_label', 'ood_score', 'spatial_domain']
                df_day[qc_cols]
                keep_cols = qc_cols
            except KeyError:
                pass
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



def print_quality_statistics(df_all, output_dir):
    """输出年度质量标签统计"""
    if 'quality_label' not in df_all.columns:
        print("  (无 quality_label 列，跳过)")
        return
    total = len(df_all)
    counts = df_all['quality_label'].value_counts()
    print("\n========== 年度质量标签统计 ==========")
    for lbl in ['A','B','C','D']:
        c = counts.get(lbl, 0)
        print(f"    {lbl}: {c:>10,} ({c/total*100:>5.1f}%)")

    df = df_all.copy()
    df['lat_r'] = np.floor(df['grid_lat']/0.1)*0.1+0.05
    df['lon_r'] = np.floor(df['grid_lon']/0.1)*0.1+0.05
    gt = df.groupby(['lat_r','lon_r']).size().reset_index(name='total_days')
    gd = df[df['quality_label']=='D'].groupby(['lat_r','lon_r']).size().reset_index(name='d_days')
    gs = gt.merge(gd, on=['lat_r','lon_r'], how='left').fillna(0)
    gs['d_ratio'] = gs['d_days']/gs['total_days']
    nc = len(gs)
    print(f"\n网格级 D 级 ({nc} 网格):")
    print(f"  至少 1 天 D: {(gs['d_days']>0).sum()}  ({(gs['d_days']>0).sum()/nc*100:.1f}%)")
    print(f"  多数天 D:    {(gs['d_ratio']>0.5).sum()}  ({(gs['d_ratio']>0.5).sum()/nc*100:.1f}%)")
    print(f"  全年都是 D:  {(gs['d_days']==gs['total_days']).sum()}  ({(gs['d_days']==gs['total_days']).sum()/nc*100:.1f}%)")
    print("="*40)

    csv_path = os.path.join(output_dir, 'Quality_Annual_Stats.csv')
    pd.DataFrame([{'Quality':'A','Count':int(counts.get('A',0)),'Pct':round(counts.get('A',0)/total*100,1)},
                  {'Quality':'B','Count':int(counts.get('B',0)),'Pct':round(counts.get('B',0)/total*100,1)},
                  {'Quality':'C','Count':int(counts.get('C',0)),'Pct':round(counts.get('C',0)/total*100,1)},
                  {'Quality':'D','Count':int(counts.get('D',0)),'Pct':round(counts.get('D',0)/total*100,1)},
                  {'Quality':'D_any_day','Count':int((gs['d_days']>0).sum()),'Pct':round((gs['d_days']>0).sum()/nc*100,1)}]
    ).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"统计 CSV: {csv_path}")


def plot_annual_mean_1x2(df_all, cmap, output_dir, version, target_res=0.1, quality_mode='none'):
    """
    绘制年均全景分析图：浓度增强量 (左图) + 观测代表性误差 (右图)
    """
    print("🎨 正在构建高保真年均 1x2 拼图...")
    
    df_annual = df_all.groupby(['lat_resampled', 'lon_resampled']).mean(numeric_only=True).reset_index()

    flag_suffix = ''
    has_flag = False
    if quality_mode in ['drop_d', 'overlay_d'] and 'quality_label' in df_all.columns:
        has_flag = True
        if quality_mode == 'drop_d':
            n_total = len(df_all)
            df_clean = df_all[df_all['quality_label'] != 'D'].copy()
            n_dropped = n_total - len(df_clean)
            print(f'   移除 {n_dropped} 个 D 级点, 剩余 {len(df_clean)} 点。')
            df_annual = df_clean.groupby(['lat_resampled','lon_resampled']).mean(numeric_only=True).reset_index()
        flag_suffix = '_Flag'
    
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
    lon_res = lons[1] - lons[0] if len(lons) > 1 else 0.1
    lat_res = lats[1] - lats[0] if len(lats) > 1 else 0.1
    lon_edges = np.concatenate([lons - lon_res/2, [lons[-1] + lon_res/2]])
    lat_edges = np.concatenate([lats - lat_res/2, [lats[-1] + lat_res/2]])

    Z_enh_masked = np.ma.array(Z_enh, mask=np.isnan(Z_enh))
    sc1 = axes[0].pcolormesh(lon_edges, lat_edges, Z_enh_masked, cmap=cmap,
                            transform=ccrs.PlateCarree(), vmin=v1, vmax=v2,
                            shading='flat', zorder=1)
        
    # 【核心修正】在 pcolormesh 执行后调用格式化函数，确保指定的 70-140, 15-55 经纬度完美生效
    format_academic_map(axes[0], title=r'(a) Annual Mean $\Delta$XCO$_2$', show_left=True, show_bottom=True)
    
    cb1 = fig.colorbar(sc1, ax=axes[0], orientation='horizontal', pad=0.09, 
                       shrink=0.95, aspect=40, extend='both')
    cb1.set_label('Predicted $\Delta$XCO$_2$ [ppm]', fontsize=12, fontweight='bold')
    cb1.ax.tick_params(size=0)  

    # ------------------ 右图：模型观测/不确定性误差 ------------------
    v3 = np.percentile(df_annual['pred_uncertainty_1sigma'], 2)
    v4 = np.percentile(df_annual['pred_uncertainty_1sigma'], 99)
    
    # 先绘制数据层
    Z_unc_masked = np.ma.array(Z_unc, mask=np.isnan(Z_unc))
    sc2 = axes[1].pcolormesh(lon_edges, lat_edges, Z_unc_masked, cmap='Spectral_r',
                            transform=ccrs.PlateCarree(), vmin=v3, vmax=v4,
                            shading='flat', zorder=1)
    
    # 【核心修正】在右侧子图绘制完数据后，调用格式化函数，并将 show_left 设为 True 打开左轴刻度显示
    format_academic_map(axes[1], title=r'(b) Representation Error', show_left=False, show_bottom=True)
    
    # 【修改提示】同步将右侧颜色条高度变薄 (fraction=0.02)
    cb2 = fig.colorbar(sc2, ax=axes[1], orientation='horizontal', pad=0.09, 
                       shrink=0.95, aspect=40, extend='both')
    cb2.set_label('Observation Error / Uncertainty [ppm]', fontsize=12, fontweight='bold')
    cb2.ax.tick_params(size=0)

    # ------------------ 拼图整饰与落盘 ------------------
    if quality_mode in ['drop_d','overlay_d']:
        print_quality_statistics(df_all, output_dir)

    plt.subplots_adjust(wspace=0.05)  
    
    save_filename = f'FIG3-Annual_Mean_Map{flag_suffix}.png'
    save_path = os.path.join(output_dir, save_filename)
    if quality_mode == 'overlay_d' and has_flag:
        df_d = df_all[df_all['quality_label'] == 'D']
        if len(df_d) > 0:
            d_grid = df_d.groupby(['lat_resampled','lon_resampled']).size().reset_index(name='cnt')
            for ax in axes:
                ax.scatter(d_grid['lon_resampled'], d_grid['lat_resampled'],
                          marker='x', color='white', s=2, alpha=0.5,
                          transform=ccrs.PlateCarree(), zorder=5, linewidth=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 高保真学术年均网格图（范围：70-140E, 15-55N）已成功导出至: {save_path}")


# ==========================================
# 3. 生产流主执行引擎
# ==========================================
def main():
    ACTIVE_VERSION = 'A01'
    PKL_DIR = f'/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}/pkl_files_flag'
    OUTPUT_DIR = f'/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}/figures'
    TARGET_RES = 0.1  
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    custom_cmap = get_custom_cmap()
    df_all = load_and_resample_data(pkl_dir=PKL_DIR, target_res=TARGET_RES)
    plot_annual_mean_1x2(df_all=df_all, cmap=custom_cmap, output_dir=OUTPUT_DIR, 
                         version=ACTIVE_VERSION, target_res=TARGET_RES, quality_mode = 'drop_d')

if __name__ == "__main__":
    main()

