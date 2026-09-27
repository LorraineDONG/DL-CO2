# -*- coding: utf-8 -*-
"""fig10.py — 特征空间密度诊断 (v3)"""
import os, json, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
from scipy.optimize import brentq
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter, FuncFormatter
import cartopy.crs as ccrs, cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import shape

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'axes.titleweight':'bold',
    'axes.labelsize':13,'axes.labelweight':'normal',
    'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':12,
    'axes.linewidth':1.5,'lines.linewidth':1.8,'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.size':6,'ytick.major.size':6,'xtick.major.width':1.2,
    'xtick.top':True,'ytick.right':True,
    'axes.grid':True,'grid.linestyle':'--','grid.linewidth':0.8,'grid.alpha':0.4,
    'grid.color':'#B0B0B0','figure.dpi':300,'savefig.dpi':300,
    'savefig.bbox':'tight','savefig.pad_inches':0.05})

DATA_FILE='/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
FEATURES_JSON='/home/whdong/dl/best_params/selected_features-A01.json'
SCALER_PATH='/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
MODEL_BASE_PATH='/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01'
OUTPUT_DIR='/home/whdong/dl/figures/scatter_density_A01'
TARGET='xco2_enhanced';SEEDS=[42,100,2023,888,999,1234,5678,777,666,1024]
WORLD_SHP='/home/whdong/shapefile/world/world.shp'
CHINA_PROV_SHP='/home/whdong/shapefile/china/province.shp'
OS=0.98
os.makedirs(OUTPUT_DIR,exist_ok=True)

def calc_cov(yt,yp,sr,k,ep):
    z=norm.ppf(0.5+ep/2);cs=sr*k;l=yp-z*cs;u=yp+z*cs;return np.mean((yt>=l)&(yt<=u))
def find_k(yt,yp,sr,tp=0.95,kr=(0.5,35)):
    def o(k):cs=sr*k;l=yp-1.96*cs;u=yp+1.96*cs;return np.mean((yt>=l)&(yt<=u))-tp
    try:return brentq(o,kr[0],kr[1])
    except:return np.nan
def rc_curve(yt,yp,sr,kf):
    ep=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
    cv=np.array([calc_cov(yt,yp,sr,kf,p) for p in ep]);return ep,cv
def ce_func(probs,coverages):return np.mean(np.abs(coverages-probs))*100

def add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE,linewidth=1.0,edgecolor='#333333')
    if os.path.exists(WORLD_SHP):
        try:ax.add_feature(cfeature.ShapelyFeature(shpreader.Reader(WORLD_SHP).geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='#333333',linewidth=0.8,linestyle=':'))
        except:ax.add_feature(cfeature.BORDERS,linewidth=0.8,linestyle=':')
    if os.path.exists(CHINA_PROV_SHP):
        try:import shapefile as ps;sf=ps.Reader(CHINA_PROV_SHP,encoding='gbk');ax.add_feature(cfeature.ShapelyFeature([shape(s.__geo_interface__) for s in sf.shapes()],ccrs.PlateCarree(),facecolor='none',edgecolor='#777777',linewidth=0.4,linestyle=':'))
        except:pass
    gl=ax.gridlines(draw_labels=True,linewidth=0.8,color='gray',alpha=0.4,linestyle='--')
    gl.top_labels=False;gl.right_labels=False;gl.xformatter=LONGITUDE_FORMATTER;gl.yformatter=LATITUDE_FORMATTER
    gl.xlabel_style={'size':13};gl.ylabel_style={'size':13}

def draw_ood_scatter_plot(ax,ood_scores,abs_errors):
    ms=50000;n=len(ood_scores);idx=np.random.choice(n,ms,replace=False)if n>ms else np.arange(n)
    x,y=ood_scores[idx],abs_errors[idx];xy=np.vstack([x,y]);z=gaussian_kde(xy)(xy);si=z.argsort()
    ax.scatter(x[si],y[si],c=z[si],s=5,cmap='viridis',alpha=0.9,edgecolor='none')
    m,b=np.polyfit(x,y,1);xl=np.array([x.min(),x.max()]);ax.plot(xl,m*xl+b,color='firebrick',linestyle='-',linewidth=2,label='Trend (slope={:.2f})'.format(m))
    r=np.corrcoef(x,y)[0,1];ax.set_xlabel('OOD Score (z-score)');ax.set_ylabel('Abs Error [ppm]')
    props=dict(boxstyle='round,pad=0.4',facecolor='white',alpha=0.85,edgecolor='silver')
    ax.text(0.05,0.95,'r={:.3f}\nN={:,}'.format(r,len(x)),transform=ax.transAxes,fontsize=10,va='top',bbox=props);ax.legend(loc='lower right')
    return

def draw_binned_ood_mae(ax,ood_scores,abs_errors,nb=15):
    df=pd.DataFrame({'ood':ood_scores,'err':abs_errors});mo=df['ood'].quantile(OS)
    bs=np.linspace(df['ood'].min(),mo,nb+1);df['bin']=pd.cut(df['ood'],bins=bs)
    st=df.groupby('bin',observed=False).agg(ood_mean=('ood','mean'),err_mean=('err','mean'),err_sem=('err',lambda x:np.std(x,ddof=1)/np.sqrt(len(x))if len(x)>1 else 0),count=('err','count')).dropna()
    st=st[st['count']>30]
    ax.errorbar(st['ood_mean'],st['err_mean'],yerr=st['err_sem'],fmt='o',color='teal',ecolor='gray',elinewidth=1.5,capsize=4,markersize=7,markeredgecolor='white',markeredgewidth=1,label='Binned Mean ± SEM')
    if len(st)>1:
        m,b=np.polyfit(st['ood_mean'],st['err_mean'],1);ax.plot(st['ood_mean'],m*st['ood_mean']+b,color='firebrick',linestyle='-',linewidth=2,label='Trend (slope={:.2f})'.format(m))
        r=np.corrcoef(st['ood_mean'],st['err_mean'])[0,1]
    else:r=0
    ax.set_xlabel('OOD Score (z-score)');ax.set_ylabel('Mean Absolute Error [ppm]')
    ax.text(0.04,0.95,'(a)',transform=ax.transAxes,fontsize=16,fontweight='bold',va='top',ha='left')
    props=dict(boxstyle='round,pad=0.4',facecolor='white',alpha=0.85,edgecolor='silver')
    ax.text(0.05,0.86,'Binned r={:.3f}'.format(r),transform=ax.transAxes,fontsize=12,verticalalignment='top',bbox=props)
    ax.legend(loc='lower right')

def draw_ood_reliability(ax,ood_scores,y_true,y_pred,std_raw,global_k):
    p33,p67=np.percentile(ood_scores,[33,67])
    strata=[('Low OOD',ood_scores<p33,'#2ecc71'),('Mid OOD',(ood_scores>=p33)&(ood_scores<p67),'#f39c12'),('High OOD',ood_scores>=p67,'#e74c3c')]
    ep=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
    ax.plot([0,1],[0,1],color='black',linestyle='--',linewidth=1.5,label='Perfect (1:1)')
    for label,mask,color in strata:
        yt,yp,sr=y_true[mask],y_pred[mask],std_raw[mask]
        if np.sum(mask)<100:continue
        ko=find_k(yt,yp,sr);ko=ko if not np.isnan(ko) else global_k
        ep2,cc=rc_curve(yt,yp,sr,ko);cce=ce_func(ep2,cc)
        ax.plot(ep,cc,marker='o',markersize=4,linestyle='-',color=color,linewidth=1.8,label=label+' (k={:.1f}, CE={:.1f}%)'.format(ko,cce))
    ax.set_xlim([0,1]);ax.set_ylim([0,1])
    ax.set_xlabel('Expected Confidence Level');ax.set_ylabel('Observed Coverage (PICP)')
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:''if np.isclose(x,0)else '{:.0f}%'.format(x*100)))
    ax.text(0.04,0.95,'(b)',transform=ax.transAxes,fontsize=16,fontweight='bold',va='top',ha='left')
    ax.legend(loc='lower right',fontsize=12,frameon=True,edgecolor='silver')

def draw_ood_spatial_map(ax,df_test,ood_scores):
    ax.set_extent([70,140,15,55],crs=ccrs.PlateCarree());add_map_features(ax)
    gd=pd.DataFrame({'lon':df_test['grid_lon'].values,'lat':df_test['grid_lat'].values,'ood':ood_scores})
    gm=gd.groupby(['lon','lat'])['ood'].mean().reset_index()
    pm=np.percentile(ood_scores,OS*100)
    sc=ax.scatter(gm['lon'],gm['lat'],c=gm['ood'],cmap='YlOrRd',vmin=-1,vmax=1.8,s=8,transform=ccrs.PlateCarree(),zorder=3)
    ax.text(0.04,0.95,'(c)',transform=ax.transAxes,fontsize=16,fontweight='bold',va='top',ha='left')
    cb=plt.colorbar(sc,ax=ax,orientation='horizontal',shrink=0.7,aspect=35,pad=0.06,extend='both')
    cb.set_label('OOD Score (z-score)',fontsize=13);cb.ax.tick_params(size=0)

def create_figure(ood_scores,abs_errors,y_test,ensemble_pred,ensemble_std_raw,global_k,df_test,save_path):
    fig=plt.figure(figsize=(12,12))
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.3],hspace=0.2,wspace=0.22)
    ax_a=fig.add_subplot(gs[0,0]);ax_b=fig.add_subplot(gs[0,1])
    ax_c=fig.add_subplot(gs[1,:],projection=ccrs.PlateCarree())
    draw_binned_ood_mae(ax_a,ood_scores,abs_errors)
    draw_ood_reliability(ax_b,ood_scores,y_test,ensemble_pred,ensemble_std_raw,global_k)
    draw_ood_spatial_map(ax_c,df_test,ood_scores)
    plt.savefig(save_path,dpi=300,bbox_inches='tight');plt.close()
    print('Saved: '+save_path)

def main():
    print('='*55+'\n  Fig 10: Feature Space Density\n'+'='*55)
    print('1/5: load data...')
    df=pd.read_pickle(DATA_FILE);dc=df.dropna().copy()
    dc['no2_trop_log']=np.log(dc['no2_trop']);dc['date']=pd.to_datetime(dc['date']);dc['month']=dc['date'].dt.month;dc['doy']=dc['date'].dt.dayofyear
    dc['month_sin']=np.sin(2*np.pi*dc['month']/12.0);dc['month_cos']=np.cos(2*np.pi*dc['month']/12.0)
    dc['doy_sin']=np.sin(2*np.pi*dc['doy']/365.25);dc['doy_cos']=np.cos(2*np.pi*dc['doy']/365.25)
    dc['ndvi_t2m_cross']=dc['ndvi']*dc['era5_t2m'];dc['ssrd_t2m_cross']=dc['era5_ssrd']*dc['era5_t2m']
    dc['ntl_nox_cross']=dc['ntl']*dc['meic_nox'];dc['era5_wind_speed']=np.sqrt(dc['era5_u100']**2+dc['era5_v100']**2)
    with open(FEATURES_JSON)as f:sf=json.load(f)
    print('2/5: split...');dp,dt=train_test_split(dc,test_size=0.2,random_state=42)
    Xpr,Xtr=dp[sf].values,dt[sf].values;yt=dt[TARGET].values
    print('3/5: scale...');es=joblib.load(SCALER_PATH);Xps=es.transform(Xpr);Xts=es.transform(Xtr)
    print('4/5: inference...')
    tp=[];[tp.append(joblib.load('{b}_seed{s}.pkl'.format(b=MODEL_BASE_PATH,s=seed)).predict(Xts))for seed in SEEDS]
    tp=np.array(tp);ep=np.mean(tp,0);esr=np.std(tp,0);ae=np.abs(ep-yt)
    gk=find_k(yt,ep,esr);gk=gk if not np.isnan(gk)else 11.99;print('  global k={:.3f}'.format(gk))
    print('5/5: OOD...')
    nn=NearestNeighbors(n_neighbors=100,metric='euclidean',n_jobs=-1);nn.fit(Xps)
    d,_=nn.kneighbors(Xts);o=(d.mean(1)-d.mean())/d.std();print('  OOD: mean={:.2f}, max={:.2f}'.format(o.mean(),o.max()))
    print('\n'+'='*68)
    print('  OOD Score Distribution by Prediction Magnitude')
    print('='*68)
    hdr='  {0:<10} {1:>7} {2:>9} {3:>10} {4:>10} {5:>10}'.format('Group','N','OOD_mean','Low OOD(%)','Mid OOD(%)','High OOD(%)')
    print(hdr);print('  '+'-'*68)
    rows=[];p33r,p67r=np.percentile(o,[33,67])
    for lb,mk in [('<2 ppm',ep<2.0),('2-5 ppm',(ep>=2.0)&(ep<=5.0)),('>5 ppm',ep>5.0)]:
        sb=o[mk];n=int(np.sum(mk))
        if n==0:continue
        m=np.mean(sb)
        lo=np.mean(sb<p33r)*100
        mi=np.mean((sb>=p33r)&(sb<p67r))*100
        hi=np.mean(sb>=p67r)*100
        rows.append({'Group':lb,'N':n,'OOD_mean':round(m,3),'Low_pct':round(lo,1),'Mid_pct':round(mi,1),'High_pct':round(hi,1)})
        print('  {0:<10} {1:>7} {2:>9.3f} {3:>9.1f}% {4:>9.1f}% {5:>9.1f}%'.format(lb,n,m,lo,mi,hi))
    print('='*68)
    csvp=os.path.join(OUTPUT_DIR,'OOD_vs_Magnitude.csv')
    pd.DataFrame(rows).to_csv(csvp,index=False,encoding='utf-8-sig');print('  Saved: '+csvp)
    sp=os.path.join(OUTPUT_DIR,'FIG10-Feature_Space_Density.png')
    create_figure(o,ae,yt,ep,esr,gk,dt,sp);print('Fig 10 done.')
if __name__=='__main__':main()

