import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 0. 路径配置 (请根据你的实际路径修改)
# ==========================================
# 刚才训练好的模型和参数文件路径
MODEL_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_10fold_model.pkl' 
SCALER_PATH = '/home/whdong/dl/models/XCO2en_SHP-xgb_10fold_scaler.pkl' 
FEATURES_JSON = '/home/whdong/dl/best_params/selected_features_10fold.json'

# 我们之前生成的每日推理数据集 (以某一天为例)
INFERENCE_DATA_PATH = '/home/whdong/dl/post_data/inference_features_China_0p1deg_20230101.pkl'

# 输出地图保存路径
OUTPUT_MAP_PATH = '/home/whdong/dl/figures/01.png'

# ==========================================
# 1. 加载模型与资产
# ==========================================
print("📂 正在加载模型、Scaler 和特征列表...")
xgb_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_JSON, 'r', encoding='utf-8') as f:
    selected_features = json.load(f)

# ==========================================
# 2. 加载数据并执行严格对齐的特征工程
# ==========================================
print(f"📊 正在处理推理数据: {INFERENCE_DATA_PATH}")
df = pd.read_pickle(INFERENCE_DATA_PATH)

# (1) 时间特征计算 (必须与训练时一致)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['doy'] = df['date'].dt.dayofyear

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365.25)
df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365.25)

# (2) 物理交叉与衍生特征
# 注意：前提是 np.log 里面不能有负数或0，之前我们的QC函数已经过滤了 no2_trop > 0
if 'no2_trop' in df.columns:
    df['no2_trop_log'] = np.log(df['no2_trop'])
    
df['ndvi_t2m_cross'] = df['ndvi'] * df['era5_t2m']
df['ssrd_t2m_cross'] = df['era5_ssrd'] * df['era5_t2m']
df['ntl_nox_cross'] = df['ntl'] * df['meic_nox']
df['era5_wind_speed'] = np.sqrt(df['era5_u100']**2 + df['era5_v100']**2)

# (3) 过滤缺失值，提取模型需要的列
# 卫星数据有云遮挡，肯定会有 NaN。我们只保留所需特征全都齐全的网格像元
# 顺便把经纬度也保留下来，因为画图需要！
required_columns = selected_features + ['grid_lat', 'grid_lon']

# 检查是否有缺失的列
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"严重错误：推理数据中缺少训练时的必要特征: {missing_cols}")

# 剔除 NaN
df_valid = df.dropna(subset=required_columns).copy()
print(f"✅ 有效像元过滤完毕: 共计 {len(df_valid)} 个网格具备完整特征。")

# ==========================================
# 3. 缩放与预测
# ==========================================
print("🧠 正在进行模型预测...")
# 严格按照 JSON 里的特征顺序提取数据
X_infer_raw = df_valid[selected_features].values

# 使用训练好的 Scaler 进行标准化 transform (千万不能再 fit！)
X_infer_scaled = scaler.transform(X_infer_raw)

# 预测结果
preds = xgb_model.predict(X_infer_scaled)
df_valid['pred_XCO2en'] = preds

# ==========================================
# 4. Cartopy 空间制图
# ==========================================
print("🗺️ 正在绘制空间分布图...")
fig = plt.figure(figsize=(12, 8))

# 设置投影方式 (这里用最常见的 PlateCarree)
ax = plt.axes(projection=ccrs.PlateCarree())

# 添加地图底图特征
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8)
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree()) # 中国区域大致边界

# 绘制散点图 (因为是网格数据，散点足够密集时看起来就是连片的)
# 如果是 0.1 度分辨率，调整 s=2 左右，0.25 度可以调大到 s=5
scatter = ax.scatter(
    df_valid['grid_lon'], 
    df_valid['grid_lat'], 
    c=df_valid['pred_XCO2en'], 
    cmap='Spectral_r',      # 颜色条：红高蓝低
    s=3,                    # 点的大小
    transform=ccrs.PlateCarree(),
    vmin=np.percentile(preds, 2),   # 去除极值影响，用 2% 和 98% 分位数作为色标上下限
    vmax=np.percentile(preds, 98)
)

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.08, fraction=0.04)
cbar.set_label('Predicted ΔXCO$_2$ (ppm)', fontsize=12)

# 添加标题和网格线
plt.title('Daily Predicted XCO$_2$ Enhancement', fontsize=16, pad=15)
gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

# 保存与展示
plt.savefig(OUTPUT_MAP_PATH, dpi=300, bbox_inches='tight')
print(f"🎉 大功告成！地图已保存至: {OUTPUT_MAP_PATH}")
plt.show() # 如果你在 Jupyter 运行，可以取消注释直接查看