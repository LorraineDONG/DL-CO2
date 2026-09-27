# -*- coding: utf-8 -*-
"""fig_before_after_qc.py — 去除 D 级前后年均对比 (1x2)"""
import os, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs, cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

PKL_DIR = '/home/whdong/dl/ML-prediction-output_result/A01/pkl_files_flag'
OUTPUT_DIR = '/home/whdong/dl/ML-prediction-output_result/A01/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,
    'xtick.labelsize':10,'ytick.labelsize':10,
    'axes.linewidth':1.2,'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.size':5,'xtick.major.width':1.2,
    'xtick.top':True,'ytick.right':True,
    'axes.grid':True,'grid.linestyle':'--','grid.linewidth':0.6,'grid.alpha':0.35,
    'grid.color':'#B0B0B0','figure.dpi':300,'savefig.dpi':300,
    'savefig.bbox':'tight','savefig.pad_inches':0.05})

def _safe_pc(ax, matrix, **kw):
    lons = matrix.columns.values; lats = matrix.index.values
    Z = np.ma.array(matrix.values, mask=np.isnan(matrix.values))
    kw.pop('shading', None)
    return ax.pcolormesh(lons, lats, Z, shading='nearest', **kw)

def add_map_features(ax, show_left=True, show_bottom=True):
    """完全对齐 fig5.py 的省界和国界画法"""
    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#333333', zorder=2)
    wp = '/home/whdong/shapefile/world/world.shp'
    if os.path.exists(wp):
        ax.add_feature(cfeature.ShapelyFeature(shpreader.Reader(wp).geometries(),
            ccrs.PlateCarree(), facecolor='none', edgecolor='#333333', linewidth=0.8, linestyle=':'), zorder=2)
    cp = '/home/whdong/shapefile/china/province.shp'
    if os.path.exists(cp):
        ax.add_feature(cfeature.ShapelyFeature(shpreader.Reader(cp, encoding='gbk').geometries(),
            ccrs.PlateCarree(), facecolor='none', edgecolor='#777777', linewidth=0.4, linestyle=':'), zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels=False; gl.right_labels=False
    gl.xlocator=mticker.FixedLocator([75,95,115,135])
    gl.ylocator=mticker.FixedLocator([20,30,40,50])
    gl.xformatter=LONGITUDE_FORMATTER; gl.yformatter=LATITUDE_FORMATTER
    gl.xlabel_style={'size':9,'color':'black'}; gl.ylabel_style={'size':9,'color':'black'}
    gl.left_labels = show_left
    gl.bottom_labels = show_bottom

def load_and_grid(pkl_dir, target_res=0.1):
    files = sorted(glob.glob(os.path.join(pkl_dir, "pred_0.1deg_*.pkl")))
    if not files: raise FileNotFoundError(pkl_dir)
    dfs = []
    for f in files:
        try:
            d = pd.read_pickle(f)
            if 'quality_label' not in d.columns: d['quality_label'] = 'A'
            dfs.append(d[['grid_lat','grid_lon','pred_xco2_enhanced','pred_uncertainty_1sigma','quality_label']])
        except: pass
    df = pd.concat(dfs, ignore_index=True)
    print(f"加载 {len(df):,} 条")
    df['lat_r'] = np.floor(df['grid_lat']/target_res)*target_res+target_res/2
    df['lon_r'] = np.floor(df['grid_lon']/target_res)*target_res+target_res/2
    return df

def main():
    df = load_and_grid(PKL_DIR)
    g_all = df.groupby(['lat_r','lon_r'])[['pred_xco2_enhanced','pred_uncertainty_1sigma']].mean().reset_index()
    g_qc = df[df['quality_label']!='D'].groupby(['lat_r','lon_r'])[['pred_xco2_enhanced','pred_uncertainty_1sigma']].mean().reset_index()
    print(f"  ALL: {len(g_all)} 网格, QC: {len(g_qc)} 网格")

    pk = dict(index='lat_r', columns='lon_r')
    diff_x = g_all.pivot(values='pred_xco2_enhanced', **pk) - g_qc.pivot(values='pred_xco2_enhanced', **pk)
    diff_s = g_all.pivot(values='pred_uncertainty_1sigma', **pk) - g_qc.pivot(values='pred_uncertainty_1sigma', **pk)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': ccrs.PlateCarree()})

    vx = np.nanpercentile(np.abs(diff_x.values), 98)
    pcm1 = _safe_pc(axes[0], diff_x, cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin=-vx, vmax=vx)
    add_map_features(axes[0])
    axes[0].set_title('(c) XCO2 Enhancement Diff (ALL - QC)', pad=12)
    # axes[0].set_title('(a) XCO2 Enhancement Diff (ALL - QC)', pad=12)
    cb1 = plt.colorbar(pcm1, ax=axes[0], orientation='horizontal', pad=0.06, shrink=0.85, aspect=35, extend='both')
    cb1.set_label('XCO2 Enhancement Diff [ppm]', fontsize=11); cb1.ax.tick_params(size=0)

    vs = np.nanpercentile(np.abs(diff_s.values), 98)
    pcm2 = _safe_pc(axes[1], diff_s, cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin=-0.2, vmax=0.2)
    add_map_features(axes[1], show_left=False)
    axes[1].set_title('(d) Sigma Diff (ALL - QC)', pad=12)
    # axes[1].set_title('(b) Sigma Diff (ALL - QC)', pad=12)
    cb2 = plt.colorbar(pcm2, ax=axes[1], orientation='horizontal', pad=0.06, shrink=0.85, aspect=35, extend='both')
    cb2.set_label('Sigma Diff [ppm]', fontsize=11); cb2.ax.tick_params(size=0)

    plt.subplots_adjust(wspace=0.04)
    sp = os.path.join(OUTPUT_DIR, 'FIG16-Before_After_QC_Difference.png')
    plt.savefig(sp, dpi=300, bbox_inches='tight'); plt.close()
    print(f"保存至: {sp}")
    print(f"XCO2 diff: 均值={np.nanmean(diff_x):.4f}, P98={vx:.4f}")
    print(f"Sigma diff: 均值={np.nanmean(diff_s):.4f}, P98={vs:.4f}")

if __name__ == "__main__":
    main()
