import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap

# ==========================================
# 1. 全局设置学术期刊级样式规范
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
    'axes.linewidth': 1.2,         
    'xtick.direction': 'in',       
    'ytick.direction': 'in',       
    'xtick.top': True,             
    'ytick.right': True,           
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.4,
    'grid.color': '#B0B0B0',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',       
    'savefig.pad_inches': 0.05
})

# ==========================================
# 2. 核心高鲁棒性 1x3 拼图渲染引擎
# ==========================================
def draw_shap_interaction_panel(X_raw, shap_values, feature_names, plot_configs, save_path):
    """
    通用型 1x3 物理交互散点图绘制函数 (鲁棒性极强，输入变量对即可直接出图)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    panels = ['(a)', '(b)', '(c)']
    
    for i, cfg in enumerate(plot_configs):
        ax = axes[i]
        feat_x = cfg['x_feature']
        feat_z = cfg['z_feature']
        
        # 稳健性检查：防止特征拼写错误导致崩溃
        if feat_x not in feature_names or feat_z not in feature_names:
            print(f"⚠️ 错误: 特征库中未找到 {feat_x} 或 {feat_z}，该子图跳过。")
            continue
            
        idx_x = feature_names.index(feat_x)
        idx_z = feature_names.index(feat_z)
        
        # 提取点对点物理对齐的数据流
        x_data = X_raw[:, idx_x]
        y_data = shap_values[:, idx_x] # 提取对应的 SHAP 贡献值
        z_data = X_raw[:, idx_z]

        if feat_x == 'no2_trop_mean':
            x_data = x_data / 1e16

        if feat_z == 'no2_trop_mean':
            z_data = z_data / 1e16

        if feat_x == 'era5_ssrd_lead1':
            x_data = x_data / 1e6

        if feat_x == 'era5_sp':
            x_data = x_data / 100

        # 绘制物理底色虚线 Y=0
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)
        
        # 绘制学术散点层 (采用高对比度 Spectral_r 色带)
        sc = ax.scatter(x_data, y_data, c=z_data, cmap='Spectral_r', 
                        s=15, alpha=0.8, edgecolors='none', zorder=2)
        
        # 细致整饰坐标轴标签与子图文字
        ax.set_xlabel(cfg.get('x_label', feat_x), fontweight='bold')
        ax.set_ylabel(f'SHAP Value for {cfg.get("x_short_name", feat_x)}', fontweight='bold')
        
        # 独立悬挂子图对应的学术 Colorbar，避免挤压主图尺寸
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label(cfg.get('z_label', feat_z), fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        # 左上角大字硬锁面板标签 (a), (b), (c)
        ax.text(0.03, 0.95, panels[i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)
        ax.set_title(cfg.get('title', ''), fontsize=11, pad=10, loc='center')

    plt.subplots_adjust(wspace=0.35) # 预留充裕的水平间距给 Colorbar 文本
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"🎉 成功导出高保真 1x3 交互拼图至: {save_path}")


# ==========================================
# 3. 生产流主执行引擎
# ==========================================
def main():
    # 严格对齐 fig1.py 的环境路径
    DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    FEATURES_JSON = '/home/whdong/dl/best_params/selected_features-A01.json'
    SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl'
    MODEL_BASE_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01' 
    OUTPUT_DIR = '/home/whdong/dl/figures/scatter_density_A01' 
    
    TARGET = 'xco2_enhanced'
    SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]

    print("\n📂 1/4: 加载原始数据与特征工程同步复原...")
    df = pd.read_pickle(DATA_FILE)
    df_clean = df.dropna().copy()
    
    # 特征工程全量重构
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

    # 数据集切分与固定下采样 (严格保持与你的训练空间流一致)
    df_pool, _ = train_test_split(df_clean, test_size=0.2, random_state=42)
    X_pool_raw = df_pool[selected_features].values
    
    eval_scaler = joblib.load(SCALER_PATH)
    X_pool_scaled = eval_scaler.transform(X_pool_raw)

    np.random.seed(42)
    shap_sample_size = min(5000, len(X_pool_scaled))
    shap_idx = np.random.choice(len(X_pool_scaled), shap_sample_size, replace=False)
    
    # 核心：scaled 用于机器学习推理，raw 用于绘图物理展示
    X_shap_sample_scaled = X_pool_scaled[shap_idx]
    X_shap_sample_raw = X_pool_raw[shap_idx]

    print("🧠 2/4: 并行遍历 10 模型提取 SHAP 值并执行集成均值化...")
    shap_values_all = []
    for seed in SEEDS:
        model_path = f"{MODEL_BASE_PATH}_seed{seed}.pkl"
        model = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)
        shap_values_all.append(explainer.shap_values(X_shap_sample_scaled))
    ensemble_shap_values = np.mean(shap_values_all, axis=0)

    # =========================================================
    # 🎨 3/4: 配置并渲染第一组拼图 (主位物理驱动派)
    # =========================================================
    print("\n🎬 3/4: 开始渲染第一组 SHAP 物理交互拼图...")
    configs_set1 = [
        {
            'x_feature': 'no2_trop_mean', 'z_feature': 'era5_t2m_lead1', 'x_short_name': r'NO$_2$ VCD',
            'x_label': r'NO$_2$ VCD ($10^{16}$ molecules/cm$^2$)', 'z_label': r'2m Temp Lead 1 ($K$)',
            'title': r'Emission Proxy $\times$ Thermal Feedback'
        },
        {
            'x_feature': 'doy_sin', 'z_feature': 'no2_trop_mean', 'x_short_name': 'DOY (sin)',
            'x_label': 'Day of Year (sin)', 'z_label': r'NO$_2$ VCD ($10^{16}$ molecules/cm$^2$)',
            'title': r'Temporal Cycle $\times$ Emission Intensity'
        },
        {
            'x_feature': 'grid_lat', 'z_feature': 'era5_t2m_lead1', 'x_short_name': 'Latitude',
            'x_label': 'Latitude (°N)', 'z_label': r'2m Temp Lead 1 ($K$)',
            'title': r'Latitudinal Space $\times$ Heating Boundary'
        }
    ]
    save_path_set1 = os.path.join(OUTPUT_DIR, 'FIG8-SHAP_Interactions_Set1.png')
    draw_shap_interaction_panel(X_shap_sample_raw, ensemble_shap_values, selected_features, configs_set1, save_path_set1)

    # =========================================================
    # 🎨 4/4: 配置并渲染第二组拼图 (黑马机制挖掘派)
    # =========================================================
    print("🎬 4/4: 开始渲染第二组 SHAP 物理交互拼图...")
    configs_set2 = [
        {
            'x_feature': 'era5_sp', 'z_feature': 'no2_trop_mean', 'x_short_name': 'Surface Pressure',
            'x_label': 'Surface Pressure ($hPa$)', 'z_label': r'NO$_2$ VCD ($10^{16}$ molecules/cm$^2$)',
            'title': r'Surface Pressure $\times$ Emission Intensity'
        },
        {
            'x_feature': 'era5_d2m_lag1', 'z_feature': 'doy_sin', 'x_short_name': 'Dewpoint Temp',
            'x_label': '2m Dewpoint Temp Lag 1 ($K$)', 'z_label': 'Day of Year (sin)',
            'title': r'Dewpoint Temp $\times$ Seasonal Cycle'
        },
        {
            'x_feature': 'era5_ssrd_lead1', 'z_feature': 'grid_lat', 'x_short_name': 'Solar Radiation',
            'x_label': r'Surface Solar Radiation Lead 1 ($10^{6}$ $J/m^2$)', 'z_label': 'Latitude (°N)',
            'title': r'Solar Radiation $\times$ Latitudinal Gradient'
        }
    ]
    save_path_set2 = os.path.join(OUTPUT_DIR, 'FIG8-SHAP_Interactions_Set2.png')
    draw_shap_interaction_panel(X_shap_sample_raw, ensemble_shap_values, selected_features, configs_set2, save_path_set2)


if __name__ == "__main__":
    main()