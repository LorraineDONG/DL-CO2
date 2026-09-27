# -*- coding: utf-8 -*-
"""fig13_new.py — SHAP Variance Decomposition (OOD-stratified)
输出1: 低OOD vs 高OOD |SHAP| & SHAP Std 对比
输出2: 主导不确定性来源空间分布 (Top 8)
输出3: 低OOD vs 高OOD SHAP方差剖面 (归一化,双色叠加)
"""
import os,json,joblib
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import shap
import cartopy.crs as ccrs,cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER,LATITUDE_FORMATTER
from shapely.geometry import shape

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'axes.labelsize':11,
    'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,
    'axes.linewidth':1.2,'lines.linewidth':1.8,
    'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.size':5,'ytick.major.size':5,'xtick.top':True,'ytick.right':True,
    'axes.grid':True,'grid.linestyle':'--','grid.linewidth':0.6,
    'grid.alpha':0.4,'grid.color':'#B0B0B0',
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.05})

DATA_FILE='/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
FEATURES_JSON='/home/whdong/dl/best_params/selected_features-A01.json'
SCALER_PATH='/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
MODEL_BASE_PATH='/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01'
OUTPUT_DIR='/home/whdong/dl/figures/scatter_density_A01'
WORLD_SHP='/home/whdong/shapefile/world/world.shp';CHINA_PROV_SHP='/home/whdong/shapefile/china/province.shp'
TARGET='xco2_enhanced';SEEDS=[42,100,2023,888,999,1234,5678,777,666,1024]
os.makedirs(OUTPUT_DIR,exist_ok=True)

def fdn(n):
    m={'no2_trop_mean':'no2_trop_mean','no2_trop_log':'no2_trop_log','era5_blh':'blh',
       'era5_blh_lag1':'blh_lag1','era5_blh_lead1':'blh_lead1','era5_d2m':'d2m',
       'era5_d2m_lag1':'d2m_lag1','era5_d2m_lead1':'d2m_lead1',
       'era5_sp':'sp','era5_sp_lag1':'sp_lag1','era5_sp_lead1':'sp_lead1',
       'era5_ssrd':'ssrd','era5_ssrd_lag1':'ssrd_lag1','era5_ssrd_lead1':'ssrd_lead1',
       'era5_t2m':'t2m','era5_t2m_lag1':'t2m_lag1','era5_t2m_lead1':'t2m_lead1',
       'era5_tcwv':'tcwv','era5_tcwv_lag1':'tcwv_lag1','era5_tcwv_lead1':'tcwv_lead1',
       'era5_u100':'u100','era5_u100_lag1':'u100_lag1','era5_u100_lead1':'u100_lead1',
       'era5_u10':'u10','era5_u10_lag1':'u10_lag1','era5_u10_lead1':'u10_lead1',
       'era5_v100':'v100','era5_v100_lag1':'v100_lag1','era5_v100_lead1':'v100_lead1',
       'era5_v10':'v10','era5_v10_lag1':'v10_lag1','era5_v10_lead1':'v10_lead1',
       'meic_nox':'meic_nox','ndvi':'ndvi','ntl':'ntl','dem_mean':'dem_mean',
       'ndvi_t2m_cross':'ndvi_t2m_cross','ssrd_t2m_cross':'ssrd_t2m_cross','ntl_nox_cross':'ntl_nox_cros',
       'doy_sin':'doy_sin','doy_cos':'doy_cos','month_sin':'month_sin','month_cos':'month_cos',
       'grid_lon':'grid_lon','grid_lat':'grid_lat'}
    return m.get(n,n)
def funits(n):
    if 'no2' in n:return '[10^16 mol/cm2]'
    if 't2m' in n or 'd2m' in n:return '[K]'
    if 'blh' in n:return '[m]'
    if 'sp' in n:return '[hPa]'
    if 'ssrd' in n:return '[MJ/m2]'
    if 'wind' in n or 'u10' in n or 'u100' in n or 'v10' in n or 'v100' in n:return '[m/s]'
    if 'tcwv' in n:return '[kg/m2]'
    if 'dem' in n:return '[m]'
    if 'lon' in n or 'lat' in n:return '[deg]'
    if 'meic' in n:return '[mol/m2/s]'
    if 'ntl' in n and 'cross' not in n:return '[W/m2/sr]'
    return ''
def add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE,linewidth=0.8,edgecolor='#333333')
    if os.path.exists(WORLD_SHP):
        try:ax.add_feature(cfeature.ShapelyFeature(shpreader.Reader(WORLD_SHP).geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='#333333',linewidth=0.6,linestyle=':'))
        except:ax.add_feature(cfeature.BORDERS,linewidth=0.6,linestyle=':')
    if os.path.exists(CHINA_PROV_SHP):
        try:
            import shapefile as ps
            sf=ps.Reader(CHINA_PROV_SHP,encoding='gbk')
            ax.add_feature(cfeature.ShapelyFeature([shape(s.__geo_interface__) for s in sf.shapes()],ccrs.PlateCarree(),facecolor='none',edgecolor='#777777',linewidth=0.3,linestyle=':'))
        except:pass
    gl=ax.gridlines(draw_labels=True,linewidth=0.6,color='gray',alpha=0.4,linestyle='--')
    gl.top_labels=False;gl.right_labels=False
    gl.xformatter=LONGITUDE_FORMATTER;gl.yformatter=LATITUDE_FORMATTER
    gl.xlabel_style={'size':9};gl.ylabel_style={'size':9}

# ====== 输出1: 低OOD vs 高OOD (每个面板含 |SHAP| 蓝 + SHAP Std 红) ======
def save_fig1(sa,sfull,fn,sp,lm,hm):
    """sa: (10,N,20) shap_values, sfull: (N,20) shap_std"""
    eps=1e-10
    # 低OOD
    imp_l=np.mean(np.abs(sa[:,lm,:]),axis=(0,1));dis_l=np.mean(sfull[lm],0)
    # 高OOD
    imp_h=np.mean(np.abs(sa[:,hm,:]),axis=(0,1));dis_h=np.mean(sfull[hm],0)
    # 分别归一化到 0-1
    im_l=(imp_l-imp_l.min())/(imp_l.max()-imp_l.min()+eps)
    di_l=(dis_l-dis_l.min())/(dis_l.max()-dis_l.min()+eps)
    im_h=(imp_h-imp_h.min())/(imp_h.max()-imp_h.min()+eps)
    di_h=(dis_h-dis_h.min())/(dis_h.max()-dis_h.min()+eps)
    # 左图: 低OOD, 按 imp_l 排序
    si=np.argsort(imp_l);lb=[fdn(fn[i]) for i in si]
    yp=np.arange(len(fn));bh=0.35
    fig,(a1,a2)=plt.subplots(1,2,figsize=(17,6)) #,sharey=True)
    a1.barh(yp-bh/2,im_l[si],bh,color='#4C72B0',alpha=0.85,edgecolor='white',linewidth=0.5,label='|SHAP|')
    a1.barh(yp+bh/2,di_l[si],bh,color='#C44E52',alpha=0.85,edgecolor='white',linewidth=0.5,label='SHAP Std')
    for i,yi in enumerate(yp):
        a1.text(di_l[si][i]+0.02,yi,f'{dis_l[si][i]:.4f}',
                fontsize=11,va='center',color='#C44E52',fontweight='bold')
    a1.set_yticks(yp);a1.set_yticklabels(lb,fontsize=11)
    a1.set_xlabel('Normalized Contribution (0-1)');a1.set_title('Low OOD Group',fontsize=12,fontweight='bold')
    a1.set_xlim(0,1.15);a1.legend(loc='lower right',fontsize=11)
    a1.text(0.03,0.05,'(a)',transform=a1.transAxes,fontsize=14,fontweight='bold',va='top')
    # 右图: 高OOD, 按 imp_h 排序
    si=np.argsort(imp_h);lb=[fdn(fn[i]) for i in si]
    a2.barh(yp-bh/2,im_h[si],bh,color='#4C72B0',alpha=0.85,edgecolor='white',linewidth=0.5,label='|SHAP|')
    a2.barh(yp+bh/2,di_h[si],bh,color='#C44E52',alpha=0.85,edgecolor='white',linewidth=0.5,label='SHAP Std')
    for i,yi in enumerate(yp):
        a2.text(di_h[si][i]+0.02,yi,f'{dis_h[si][i]:.4f}',
                fontsize=11,va='center',color='#C44E52',fontweight='bold')
    a2.set_yticks(yp);a2.set_yticklabels(lb,fontsize=11)
    a2.set_xlabel('Normalized Contribution (0-1)');a2.set_title('High OOD Group',fontsize=12,fontweight='bold')
    a2.set_xlim(0,1.15)#;a2.legend(loc='lower right',fontsize=9)
    a2.text(0.03,0.05,'(b)',transform=a2.transAxes,fontsize=14,fontweight='bold',va='top')
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(sp,dpi=300,bbox_inches='tight');plt.close()
    print(f"输出1: {sp}")
     # ==== 打印原始数据到屏幕 ====
    print("\n===== Low OOD Group =====")
    print(f"{'Feature':>22}  {'|SHAP|':>10}  {'SHAP Std':>10}")
    print("-" * 46)
    si_l = np.argsort(imp_l)[::-1]  # 按 |SHAP| 从大到小
    for i in si_l:
        print(f"{fdn(fn[i]):>22}  {imp_l[i]:>10.4f}  {dis_l[i]:>10.4f}")
    
    print("\n===== High OOD Group =====")
    print(f"{'Feature':>22}  {'|SHAP|':>10}  {'SHAP Std':>10}")
    print("-" * 46)
    si_h = np.argsort(imp_h)[::-1]  # 按 |SHAP| 从大到小
    for i in si_h:
        print(f"{fdn(fn[i]):>22}  {imp_h[i]:>10.4f}  {dis_h[i]:>10.4f}")
    print("=" * 46)

# ====== 输出2: 主导不确定性来源 (Top 8) ======
def save_fig2(lons,lats,dom_idx,fn,sp,top_n=8):
    from collections import Counter
    tf=[i for i,_ in Counter(dom_idx).most_common(top_n)]
    nc=len(tf)+1;cm=plt.cm.tab20;cc=[cm(i/max(nc,1)) for i in range(nc)]
    ca=np.full(len(dom_idx),len(tf),dtype=int);cl=[]
    for i,fi in enumerate(tf):ca[dom_idx==fi]=i;cl.append(fdn(fn[fi]))
    cl.append('Other')
    g=pd.DataFrame({'lon':lons,'lat':lats,'cat':ca})
    gm=g.groupby(['lon','lat'])['cat'].agg(lambda x:x.mode().iloc[0] if len(x.mode())>0 else len(tf)).reset_index()
    fig=plt.figure(figsize=(9,8));ax=plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([70,140,15,55],crs=ccrs.PlateCarree());add_map_features(ax)
    for ci in range(nc):
        s=gm[gm['cat']==ci]
        if len(s)>0:ax.scatter(s['lon'],s['lat'],color=cc[ci],s=14,label=cl[ci],transform=ccrs.PlateCarree(),zorder=3,edgecolor='none')
    ax.legend(loc='lower left',fontsize=10,markerscale=0.8,ncol=2,framealpha=0.85,edgecolor='silver')
    plt.tight_layout();plt.savefig(sp,dpi=300,bbox_inches='tight');plt.close()
    print(f"输出2: {sp}")

# ====== 输出3: 低OOD vs 高OOD SHAP方差剖面 ======
def save_fig3(Xr,sl,sh,fn,sp,lm,hm):
    mc=np.mean(np.vstack([sl,sh]),0);t8=np.argsort(mc)[-8:][::-1]
    fig,axes=plt.subplots(4,2,figsize=(10,14));af=axes.flatten()
    pl=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    for i,(ax,idx) in enumerate(zip(af,t8)):
        fvl=Xr[lm,idx];fvh=Xr[hm,idx];svl=sl[:,idx];svh=sh[:,idx]
        fa=np.concatenate([fvl,fvh])
        plo,phi=np.percentile(fa,2),np.percentile(fa,98)
        bins=np.linspace(plo,phi,12);bc=(bins[:-1]+bins[1:])/2
        def bm(fv,sv):
            r=[]
            for j in range(len(bins)-1):
                m=(fv>=bins[j])&(fv<bins[j+1]);n=np.sum(m)
                r.append(np.mean(sv[m]) if n>=10 else np.nan)
            return np.array(r)
        bml=bm(fvl,svl);bmh=bm(fvh,svh)
        ab=np.array([v for v in list(bml)+list(bmh) if not np.isnan(v)])
        if len(ab)>1 and ab.max()>ab.min():
            vmn,vmx=ab.min(),ab.max()
            bml=(bml-vmn)/(vmx-vmn+1e-10);bmh=(bmh-vmn)/(vmx-vmn+1e-10)
        else:bml=np.full_like(bml,np.nan);bmh=np.full_like(bmh,np.nan)
        vl=~np.isnan(bml);vh=~np.isnan(bmh)
        ax.plot(bc[vl],bml[vl],'o-',color='#3498db',linewidth=1.5,markersize=4,label='Low OOD' if i==0 else '')
        ax.plot(bc[vh],bmh[vh],'o-',color='#e74c3c',linewidth=1.5,markersize=4,label='High OOD' if i==0 else '')
        fd=fdn(fn[idx]);u=funits(fn[idx])
        ax.set_xlabel(f'{fd} {u}',fontsize=10);ax.set_ylabel('Norm. SHAP Std',fontsize=10)
        ax.tick_params(labelsize=8);ax.set_ylim(0,1.05)
        ax.text(0.03,0.95,pl[i],transform=ax.transAxes,fontsize=12,fontweight='bold',va='top')
    fig.legend(['Low OOD','High OOD'],loc='lower center',ncol=2,fontsize=10,framealpha=0.85,edgecolor='silver')
    plt.subplots_adjust(hspace=0.35,wspace=0.30,bottom=0.06)
    plt.savefig(sp,dpi=300,bbox_inches='tight');plt.close()
    print(f"输出3: {sp}")

# ====== 主执行流 ======
def main():
    print("="*55+"\n  fig13_new: OOD-Stratified SHAP Decomposition\n"+"="*55)
    print("1/5: 加载数据...")
    df=pd.read_pickle(DATA_FILE);dc=df.dropna().copy()
    dc['no2_trop_log']=np.log(dc['no2_trop'])
    dc['date']=pd.to_datetime(dc['date']);dc['month']=dc['date'].dt.month;dc['doy']=dc['date'].dt.dayofyear
    dc['month_sin']=np.sin(2*np.pi*dc['month']/12.0);dc['month_cos']=np.cos(2*np.pi*dc['month']/12.0)
    dc['doy_sin']=np.sin(2*np.pi*dc['doy']/365.25);dc['doy_cos']=np.cos(2*np.pi*dc['doy']/365.25)
    dc['ndvi_t2m_cross']=dc['ndvi']*dc['era5_t2m'];dc['ssrd_t2m_cross']=dc['era5_ssrd']*dc['era5_t2m']
    dc['ntl_nox_cross']=dc['ntl']*dc['meic_nox']
    with open(FEATURES_JSON,'r')as f:sf=json.load(f)

    print("2/5: OOD计算...")
    dp,dt=train_test_split(dc,test_size=0.2,random_state=42)
    Xpr=dp[sf].values;Xtr=dt[sf].values;yt=dt[TARGET].values
    es=joblib.load(SCALER_PATH);Xps=es.transform(Xpr);Xts=es.transform(Xtr)
    nn=NearestNeighbors(n_neighbors=100,metric='euclidean',n_jobs=-1);nn.fit(Xps)
    d,_=nn.kneighbors(Xts);oo=(d.mean(1)-d.mean())/d.std()
    np.random.seed(42);ns=min(5000,len(Xts));si=np.random.choice(len(Xts),ns,replace=False)
    Xsr=Xtr[si];Xss=Xts[si];tl=dt['grid_lon'].values[si];ta=dt['grid_lat'].values[si];so=oo[si]
    p33,p67=np.percentile(so,33),np.percentile(so,67);lm=so<p33;hm=so>=p67
    print(f"   SHAP样本:{ns},低OOD:{np.sum(lm)},高OOD:{np.sum(hm)}")

    print("3/5: SHAP计算...")
    sa=[]
    for s in SEEDS:
        m=joblib.load(f"{MODEL_BASE_PATH}_seed{s}.pkl")
        sa.append(shap.TreeExplainer(m).shap_values(Xss))
    sa=np.array(sa);sfull=np.std(sa,0);slo=sfull[lm];shi=sfull[hm];di=np.argmax(sfull,1)

    print("4/5: 渲染...")
    p1=os.path.join(OUTPUT_DIR,'FIG13_new-Importance_vs_Disagreement.png')
    p2=os.path.join(OUTPUT_DIR,'FIG13_new-Dominant_Uncertainty_Source.png')
    p3=os.path.join(OUTPUT_DIR,'FIG13_new-Feature_SHAP_Profiles.png')
    save_fig1(sa,sfull,sf,p1,lm,hm)
    save_fig2(tl,ta,di,sf,p2)
    save_fig3(Xsr,slo,shi,sf,p3,lm,hm)
    print("完成！")
if __name__=="__main__":main()
