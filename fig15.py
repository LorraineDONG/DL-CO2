# -*- coding: utf-8 -*-
"""
fig_qc_timespace.py — 质量标签时空分布

输出 1x2 面板:
  (a) 各月 A/B/C/D 占比堆叠柱状图
  (b) D 级频率空间地图（每个网格全年中 D 级天数占比）
"""

import os, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import cartopy.crs as ccrs, cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import shape

PKL_DIR = '/home/whdong/dl/ML-prediction-output_result/A01/pkl_files_flag'
OUTPUT_DIR = '/home/whdong/dl/ML-prediction-output_result/A01/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'axes.titleweight':'bold',
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,
    'axes.linewidth':1.5,'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.size':5,'ytick.major.size':5,'xtick.top':True,'ytick.right':True,
    'axes.grid':True,'grid.linestyle':'--','grid.linewidth':0.6,'grid.alpha':0.35,
    'grid.color':'#B0B0B0','figure.dpi':300,'savefig.dpi':300,
    'savefig.bbox':'tight','savefig.pad_inches':0.05
})

def load_data():
    print(f"读取 {PKL_DIR} ...")
    fl = sorted(glob.glob(os.path.join(PKL_DIR, "pred_0.1deg_*.pkl")))
    if not fl: raise FileNotFoundError(f"未找到文件: {PKL_DIR}")
    dl = []
    for f in fl:
        try:
            d = pd.read_pickle(f)
            if np.issubdtype(d['date'].dtype, np.datetime64):
                d['month'] = d['date'].dt.month
            dl.append(d[['grid_lat','grid_lon','month','pred_xco2_enhanced','quality_label']])
        except Exception as e:
            print(f"  跳过 {os.path.basename(f)}: {e}")
    df = pd.concat(dl, ignore_index=True)
    print(f"加载 {len(df):,} 条记录")
    df['lat_r'] = np.floor(df['grid_lat']/0.1)*0.1+0.05
    df['lon_r'] = np.floor(df['grid_lon']/0.1)*0.1+0.05
    return df

def plot_monthly_bars(ax, df):
    ct = df.groupby(['month','quality_label']).size().unstack(fill_value=0)
    cp = ct.div(ct.sum(1), axis=0)*100
    cm = {'A':'#2ecc71','B':'#3498db','C':'#f39c12','D':'#e74c3c'}
    bt = np.zeros(12)
    for lb in ['A','B','C','D']:
        if lb in cp.columns:
            v = cp[lb].reindex(range(1,13), fill_value=0).values
            ax.bar(range(1,13), v, bottom=bt, color=cm[lb], edgecolor='white', linewidth=0.5,
                   label=f'{lb} ({ct[lb].sum():,})', width=0.7)
            bt += v
    ax.set_xlabel('Month'); ax.set_ylabel('Proportion (%)')
    ax.set_xticks(range(1,13)); ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylim(0,100); ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.text(0.03,0.96,'(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.legend(loc='lower center', bbox_to_anchor=(2.1, -0.02),
          fontsize=10, ncol=4, framealpha=0.85, edgecolor='silver')

def add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333', zorder=2)
    # 世界国界
    wp = '/home/whdong/shapefile/world/world.shp'
    if os.path.exists(wp):
        wb = cfeature.ShapelyFeature(
            shpreader.Reader(wp).geometries(), ccrs.PlateCarree(),
            facecolor='none', edgecolor='#333333', linewidth=0.6, linestyle=':')
        ax.add_feature(wb, zorder=2)
    # 中国省界
    cp = '/home/whdong/shapefile/china/province.shp'
    if os.path.exists(cp):
        import shapefile
        sf = shapefile.Reader(cp, encoding='gbk')
        pb = cfeature.ShapelyFeature(
            [shape(s.__geo_interface__) for s in sf.shapes()], ccrs.PlateCarree(),
            facecolor='none', edgecolor='#777777', linewidth=0.3, linestyle=':')
        ax.add_feature(pb, zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels=False; gl.right_labels=False
    gl.xformatter=LONGITUDE_FORMATTER; gl.yformatter=LATITUDE_FORMATTER
    gl.xlabel_style={'size':9}; gl.ylabel_style={'size':9}

def plot_d_frequency_map(ax, df):
    gt = df.groupby(['lat_r','lon_r']).size().reset_index(name='total')
    gd = df[df['quality_label']=='D'].groupby(['lat_r','lon_r']).size().reset_index(name='d')
    gs = gt.merge(gd, how='left', on=['lat_r','lon_r']).fillna(0)
    gs['pct'] = gs['d']/gs['total']*100
    ax.set_extent([70,140,15,55], crs=ccrs.PlateCarree())
    add_map_features(ax)
    Z = gs.pivot(index='lat_r', columns='lon_r', values='pct')
    Zm = np.ma.array(Z.values, mask=np.isnan(Z.values))
    pcm = ax.pcolormesh(Z.columns.values, Z.index.values, Zm, cmap='YlOrRd',
                        transform=ccrs.PlateCarree(), vmin=0, vmax=50, shading='nearest')
    ax.text(0.03,0.96,'(b)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    cb = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.06, shrink=0.75, aspect=35, extend='max')
    cb.set_label('D-grade Frequency (%)', fontsize=10); cb.ax.tick_params(size=0)

def main():
    df = load_data()
    fig = plt.figure(figsize=(16,7))
    gs = fig.add_gridspec(1,2, width_ratios=[0.9,1.6], wspace=0.18)
    ax_a = fig.add_subplot(gs[0]); ax_b = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
    plot_monthly_bars(ax_a, df); plot_d_frequency_map(ax_b, df)
    sp = os.path.join(OUTPUT_DIR, 'FIG15-QC_Temporal_Spatial.png')
    plt.savefig(sp, dpi=300, bbox_inches='tight'); plt.close()
    print(f"保存至: {sp}")
    total = len(df); print(f"\n总记录数: {total:,}")
    for lbl in ['A','B','C','D']:
        c = (df['quality_label']==lbl).sum()
        print(f"  {lbl}: {c:>10,} ({c/total*100:.1f}%)")

if __name__ == "__main__":
    main()
