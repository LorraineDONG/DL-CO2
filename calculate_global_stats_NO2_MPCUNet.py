import os
import torch
import xarray as xr
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 路径配置 (请修改为你的实际路径)
# ==========================================
SHARDED_DIR = "/home/whdong/dl/data/sharded_daily_tensors-no2"
NC_MASK_PATH = "/home/whdong/dl/data/SHPshapeforNO2gapfilling_mask_3x3_025deg.nc"
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022]

def calculate_global_statistics():
    print(">>> 正在加载静态掩模 (仅在物理有效区域内统计气象场) <<<")
    nc_data = xr.open_dataset(NC_MASK_PATH)
    raw_mask_values = nc_data['mask_status'].values
    
    # 只要是有颜色(非NaN)的区域（红+蓝），我们都算作有效计算区域
    expanded_mask = torch.tensor(~np.isnan(raw_mask_values), dtype=torch.float32)
    # 统计单张图里的有效像素个数
    pixels_per_day = expanded_mask.sum().item() 
    print(f"每张图的有效像素数量: {pixels_per_day}")

    # 获取所有训练集文件
    file_paths = []
    for year in TRAIN_YEARS:
        year_prefix = f"NO2_{year}_"
        files = [os.path.join(SHARDED_DIR, f) for f in os.listdir(SHARDED_DIR) 
                 if f.startswith(year_prefix) and f.endswith(".pt")]
        file_paths.extend(files)
        
    file_paths.sort()
    total_days = len(file_paths)
    print(f"共找到 {total_days} 天的训练集数据，开始增量计算...")

    # 探测特征通道数
    sample_data = torch.load(file_paths[0], map_location='cpu')
    num_features = sample_data['x'].shape[0]
    
    # 初始化累加器
    channel_sum = torch.zeros(num_features, dtype=torch.float64)
    channel_sum_sq = torch.zeros(num_features, dtype=torch.float64)
    total_valid_pixels = 0

    # 遍历所有天数进行累加 (极低内存占用)
    for path in tqdm(file_paths, desc="扫描数据中"):
        day_data = torch.load(path, map_location='cpu')
        x_day = day_data['x'] # [C, H, W]
        
        # 将无数据区域(NaN)填为 0，防止报错
        x_day = torch.nan_to_num(x_day, nan=0.0)
        
        # 仅截取掩模覆盖的有效区域进行统计
        # expanded_mask 是 [H, W], x_day 是 [C, H, W]
        x_valid = x_day * expanded_mask
        
        # 累加每个通道的数值总和
        # 对 H 和 W 维度求和，保留 C 维度
        channel_sum += x_valid.sum(dim=(1, 2)).double()
        
        # 累加每个通道的平方和 (用于后面算方差)
        channel_sum_sq += (x_valid ** 2).sum(dim=(1, 2)).double()
        
        total_valid_pixels += pixels_per_day

    # ==========================================
    # 3. 计算最终的均值和标准差
    # 方差公式: E[X^2] - (E[X])^2
    # ==========================================
    global_means = channel_sum / total_valid_pixels
    
    # 防止浮点数精度问题导致极微小的负数
    variance = (channel_sum_sq / total_valid_pixels) - (global_means ** 2)
    variance = torch.clamp(variance, min=1e-8) 
    global_stds = torch.sqrt(variance)

    print("\n==========================================")
    print("✅ 计算完成！请将以下代码复制并替换掉主程序中的占位符：")
    print("==========================================\n")
    
    print(f"NUM_FEATURES = {num_features}")
    
    # 格式化输出为可以复制的 Python 代码
    means_str = ", ".join([f"{val:.4f}" for val in global_means])
    stds_str = ", ".join([f"{val:.4f}" for val in global_stds])
    
    print(f"GLOBAL_MEANS = torch.tensor([{means_str}]).view(-1, 1, 1)")
    print(f"GLOBAL_STDS = torch.tensor([{stds_str}]).view(-1, 1, 1)")

if __name__ == "__main__":
    calculate_global_statistics()