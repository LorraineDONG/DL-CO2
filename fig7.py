import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

# ==========================================
# 1. 全局设置字体与学术级样式
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 12,               
    'axes.titlesize': 15,          
    'axes.titleweight': 'bold',    
    'axes.labelsize': 13,          
    'axes.labelweight': 'bold',    
    'xtick.labelsize': 12,         
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',       
    'savefig.pad_inches': 0.05
})

# ==========================================
# 2. 核心绘图函数：带右上角空白网格与底部阵列图例
# ==========================================
def plot_correlation_matrix(df, features, feature_labels, output_path):
    print(f"📊 正在计算 {len(features)} 个特征的 Pearson 相关系数矩阵...")
    
    data = df[features]
    corr_matrix = data.corr(method='pearson')
    n_features = len(features)

    # 建立画幅 (增加垂直高度，给底部的图例留出充足的物理空间)
    fig, ax = plt.subplots(figsize=(10, 11.5), dpi=300)

    # ---------------------------------------------------------
    # 【核心技巧 1】绘制底部的“空白网格” (维持正方形视觉重量)
    # ---------------------------------------------------------
    # 创建一个全零的 DataFrame 作为底图
    base_data = pd.DataFrame(np.zeros_like(corr_matrix), columns=corr_matrix.columns, index=corr_matrix.index)
    # 使用极其柔和的浅灰色（#F8F9FB）填充右上角空白格，并加上白色边框线
    sns.heatmap(base_data, cmap=mcolors.ListedColormap(['#F8F9FB']), 
                cbar=False, annot=False, linewidths=0.8, linecolor='white', ax=ax)

    # ---------------------------------------------------------
    # 【核心技巧 2】叠加真正的下三角数据
    # ---------------------------------------------------------
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', 
                vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.8, linecolor='white',
                cbar_kws={"shrink": 0.82, "pad": 0.04, "aspect": 30},
                annot=True, fmt=".2f", annot_kws={"size": 9, "weight": "bold"}, 
                ax=ax) # ax=ax 保证它精确盖在第一层网格上

    # 坐标轴极简化：只显示编号 F1, F2...
    numbered_labels = [f"F{i+1}" for i in range(n_features)]
    ax.set_xticklabels(numbered_labels, rotation=0, ha='center', fontsize=11, fontweight='bold')
    ax.set_yticklabels(numbered_labels, rotation=0, fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', which='both', length=0)

    # 设置主标题
    # ax.set_title('Feature Correlation Matrix', fontsize=18, fontweight='bold', pad=20)

    # 美化色带
    cbar = ax.collections[1].colorbar  # collections[1] 是指第二层(真正的数据层)的色带
    cbar.set_label("Pearson Correlation Coefficient ($r$)", fontsize=13, fontweight='bold', labelpad=12)
    cbar.ax.tick_params(labelsize=11)

    # ---------------------------------------------------------
    # 【核心技巧 3】强行压缩主图，在正下方开辟空白图例区
    # ---------------------------------------------------------
    # bottom=0.25 意味着图表的下缘只延伸到画布 25% 的高度，下面 0~25% 全部留白！
    plt.subplots_adjust(bottom=0.26)

    # ---------------------------------------------------------
    # 【核心技巧 4】在底部空白区绘制 4列排版的图例阵列
    # ---------------------------------------------------------
    # 定义 4 列的 X 坐标 (均匀分布在 0.1 到 0.9 之间)
    col_xs = [0.10, 0.32, 0.54, 0.76]
    start_y = 0.29  # 图例标题的 Y 坐标
    row_step = 0.02 # 每行文字的垂直间距

    # 画图例标题
    # fig.text(0.10, start_y, "Feature Legend:", fontsize=13, fontweight='bold', ha='left')

    # 20 个特征分成 4 列，每列 5 行
    rows_per_col = 5
    for i in range(n_features):
        col_idx = i // rows_per_col
        row_idx = i % rows_per_col
        
        x_pos = col_xs[col_idx]
        y_pos = start_y - 0.04 - (row_idx * row_step)

        # 把 "F1:" 和 "真实标签" 分开画，这样可以保证哪怕包含 LaTeX，文字也能绝对垂直对齐
        fig.text(x_pos, y_pos, f"F{i+1}:", fontsize=11.5, fontweight='bold', ha='left', color='#333333')
        fig.text(x_pos + 0.045, y_pos, feature_labels[i], fontsize=11.5, ha='left', color='#333333')

    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"🎉 高保真满格底纹 + 底部四列图例排版已成功导出至:\n{output_path}")

# ==========================================
# 3. 生产流主执行引擎
# ==========================================
def main():
    DATA_FILE = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    OUTPUT_DIR = '/home/whdong/dl/figures/scatter_density_A01' 
    
    print("\n📂 1/3: 加载数据与特征工程复原...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"未找到数据文件: {DATA_FILE}")
        
    df = pd.read_pickle(DATA_FILE)
    df_clean = df.dropna().copy()
    
    # ---------------------------------------------------------
    # 【与 fig2.py 100% 同步的特征工程】
    # ---------------------------------------------------------
    df_clean['no2_trop_log'] = np.log(df_clean['no2_trop'])
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    
    # 补全周期时间特征
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    
    # 交互特征
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']

    print("⚖️ 2/3: 提取核心特征并映射学术标签 (按物理属性重排聚类)...")
    
    features_to_plot = [
        'no2_trop_mean', 'no2_trop_log',
        'era5_blh', 'era5_blh_lag1', 'era5_blh_lead1',
        'era5_t2m', 'era5_t2m_lead1', 'era5_d2m_lag1', 'era5_sp',
        'era5_ssrd_lag1', 'era5_ssrd_lead1', 'ndvi', 'dem_mean',
        'ssrd_t2m_cross', 'ndvi_t2m_cross',
        'grid_lon', 'grid_lat', 'month_sin', 'doy_sin', 'doy_cos'
    ]
    
    academic_labels = [
        r'NO$_2$ Mean', r'Log(NO$_2$)',
        r'PBLH', r'PBLH (Lag 1)', r'PBLH (Lead 1)',
        r'T$_{2m}$', r'T$_{2m}$ (Lead 1)', r'D$_{2m}$ (Lag 1)', r'Surface Pressure',
        r'SSRD (Lag 1)', r'SSRD (Lead 1)', r'NDVI', r'Elevation',
        r'SSRD $\times$ T$_{2m}$', r'NDVI $\times$ T$_{2m}$',
        r'Longitude', r'Latitude', r'Month (sin)', r'DOY (sin)', r'DOY (cos)'
    ]

    print("\n🎨 3/3: 开始渲染学术热力图...")
    save_path = os.path.join(OUTPUT_DIR, 'FIG7-Feature_Correlation.png')
    
    plot_correlation_matrix(
        df=df_clean, 
        features=features_to_plot, 
        feature_labels=academic_labels, 
        output_path=save_path
    )

if __name__ == "__main__":
    main()