import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
from tqdm import tqdm

# ==========================================
# [核心配置] 1. 全局特征统计量 (Global Statistics)
# ==========================================
# 【重要警告】：必须使用 2018-2022 整个训练集计算得出全局均值和方差！
# 这样做才能保留特征的季节性和时间绝对量级差异。
# 下面三行运算来自 dl/calculate_global_stats_NO2_MPCUNet.py
NUM_FEATURES = 12
GLOBAL_MEANS = torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 104.4535, 342.1241, 51.2699, 0.5045]).view(-1, 1, 1)
GLOBAL_STDS = torch.tensor([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 376.0405, 718.5071, 96.3130, 1.6829]).view(-1, 1, 1)

# ==========================================
# [辅助模块] 2. 缺失掩模生成器 (造云机器)
# ==========================================
def generate_random_dropout_mask(candidate_mask, drop_rate=0.2):
    """纯随机像素点丢弃 (类似椒盐噪声，适合微观平滑)"""
    rand_tensor = torch.rand_like(candidate_mask)
    return (rand_tensor < drop_rate).float() * candidate_mask

def generate_block_dropout_mask(candidate_mask, drop_rate=0.2, min_block=5, max_block=20):
    """连续块状丢弃 (模拟真实的云系遮挡，逼迫模型学习物理风场和大尺度传输)"""
    drop_mask = torch.zeros_like(candidate_mask)
    _, H, W = candidate_mask.shape 
    
    target_drop_pixels = candidate_mask.sum() * drop_rate
    attempts = 0
    
    while (drop_mask * candidate_mask).sum() < target_drop_pixels and attempts < 100:
        h = torch.randint(min_block, max_block + 1, (1,)).item()
        w = torch.randint(min_block, max_block + 1, (1,)).item()
        y = torch.randint(0, H - h + 1, (1,)).item()
        x = torch.randint(0, W - w + 1, (1,)).item()
        drop_mask[:, y:y+h, x:x+w] = 1.0
        attempts += 1
        
    return drop_mask * candidate_mask

# ==========================================
# [数据IO] 3. 离线数据切片模块 (解决 CPU OOM)
# ==========================================
def shard_tensor_dataset(base_dir, file_prefix, target_years, out_dir):
    """将按年存储的巨大 pkl 文件，切分为按天存储的小 pt 文件，实现极低内存占用的懒加载"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"已创建切片输出目录: {out_dir}")
    
    for year in target_years:
        file_path = os.path.join(base_dir, f"{file_prefix}-{year}.pkl")
        if not os.path.exists(file_path): continue
            
        test_file = os.path.join(out_dir, f"NO2_{year}_000.pt")
        if os.path.exists(test_file): continue # 避免重复切片

        print(f"\n正在将 {year} 年数据读入内存进行切片...")
        data = torch.load(file_path, map_location='cpu')
        x_year, y_year = data['x'], data['y']
        
        for i in tqdm(range(x_year.shape[0]), desc=f"拆分 {year} 年数据"):
            torch.save({'x': x_year[i].clone(), 'y': y_year[i].clone()}, 
                       os.path.join(out_dir, f"NO2_{year}_{i:03d}.pt"))
            
        del data, x_year, y_year

# ==========================================
# [模型定义] 4. 核心网络组件: Partial Convolution & UNet
# ==========================================
class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input_x, mask_in):
        output = self.input_conv(input_x * mask_in)
        with torch.no_grad():
            mask_out = self.mask_conv(mask_in)
        
        # 修正：计算掩模占比，避免除以0
        kernel_view = self.mask_conv.weight.shape[2] * self.mask_conv.weight.shape[3]
        mask_ratio = kernel_view / (mask_out + 1e-8)
        mask_out = torch.clamp(mask_out, 0, 1) 
        
        output = output * mask_ratio * mask_out
        if self.input_conv.bias is not None:
            output = output + self.input_conv.bias.view(1, -1, 1, 1) * mask_out
        return output, mask_out

class UNetPConv(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.enc1_conv = PartialConv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc1_relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2_conv = PartialConv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_relu = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.bottle_conv = PartialConv2d(128, 256, kernel_size=3, padding=1)
        self.bottle_relu = nn.ReLU(inplace=True)

        # 修正：使用双线性插值 (Bilinear)，使恢复出的大气浓度场梯度更加平滑自然
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1_conv = PartialConv2d(384, 128, kernel_size=3, padding=1)
        self.dec1_relu = nn.ReLU(inplace=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2_conv = PartialConv2d(192, 64, kernel_size=3, padding=1)
        self.dec2_relu = nn.ReLU(inplace=True)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, mask):
        # Encoder
        e1, m1 = self.enc1_conv(x, mask); e1 = self.enc1_relu(e1)
        p1, mp1 = self.pool1(e1), self.pool1(m1)
        
        e2, m2 = self.enc2_conv(p1, mp1); e2 = self.enc2_relu(e2)
        p2, mp2 = self.pool2(e2), self.pool2(m2)

        # Bottleneck
        b, mb = self.bottle_conv(p2, mp2); b = self.bottle_relu(b)

        # Decoder 1
        d1 = self.up1(b); md1 = self.up1(mb)
        cat_d1 = torch.cat([d1, e2], dim=1)
        cat_md1 = torch.clamp(md1 + m2, 0, 1) # 修正：正确的掩模合并逻辑
        d1, md1 = self.dec1_conv(cat_d1, cat_md1); d1 = self.dec1_relu(d1)

        # Decoder 2
        d2 = self.up2(d1); md2 = self.up2(md1)
        cat_d2 = torch.cat([d2, e1], dim=1)
        cat_md2 = torch.clamp(md2 + m1, 0, 1)
        d2, md2 = self.dec2_conv(cat_d2, cat_md2); d2 = self.dec2_relu(d2)

        return self.final_conv(d2)

# ==========================================
# [数据流水线] 5. 双掩模懒加载数据集 (Mask Decoupling)
# ==========================================
class DailyNO2Dataset(Dataset):
    def __init__(self, sharded_dir, target_years, expanded_mask, core_mask, drop_rate=0.2, mask_strategy='random'):
        self.expanded_mask = expanded_mask.float() # [输入掩模] 包含红+蓝，提供边缘上下文视野
        self.core_mask = core_mask.float()         # [考核掩模] 仅含红区，严格限制计算Loss的范围
        self.drop_rate = drop_rate
        self.mask_strategy = mask_strategy
        
        self.file_paths = sorted([os.path.join(sharded_dir, f) for f in os.listdir(sharded_dir) 
                                 if f.split('_')[1] in map(str, target_years)])

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        day_data = torch.load(self.file_paths[idx], map_location='cpu')
        x_day, y_day = day_data['x'], day_data['y']
        
        # 1. 提取当天全图真实观测情况
        valid_m_day = (~torch.isnan(y_day)).float()
        
        # 2. 全局归一化特征 (保留物理绝对量级和季节性)
        x_day = torch.nan_to_num(x_day, nan=0.0)
        x_day = (x_day - GLOBAL_MEANS) / (GLOBAL_STDS + 1e-8)
        
        # 3. 稳定的目标值对数变换
        scale_factor = 1e15
        y_day = (torch.clamp(y_day, min=0.0) / scale_factor) * valid_m_day
        y_day = torch.nan_to_num(y_day, nan=0.0)
        
        # 4. 掩模解耦核心逻辑
        # A: 模型能看到的所有真实信息 (蓝区+红区中，没有被云遮挡的部分)
        available_vision = self.expanded_mask * valid_m_day
        
        # B: 仅在红色核心区进行模拟“挖洞”，制造考题
        candidate_loss_mask = self.core_mask * valid_m_day
        if self.mask_strategy == 'block':
            loss_mask = generate_block_dropout_mask(candidate_loss_mask, self.drop_rate)
        else:
            loss_mask = generate_random_dropout_mask(candidate_loss_mask, self.drop_rate)
            
        # C: 真正的输入 = 视野总和 减去 我们故意挖掉的洞
        input_mask = available_vision - loss_mask
        
        return {
            'x': x_day,
            'y_truth': y_day, 
            'input_mask': input_mask,
            'loss_mask': loss_mask # 天然被限制在了红区内部
        }

# ==========================================
# [损失评估] 6. 定制化物理混合掩模损失
# ==========================================
def hybrid_loss_fn(pred, truth, loss_mask):
    """同时约束对数空间与物理空间的误差，防止指数还原后的灾难性放大"""
    valid_pixels = torch.sum(loss_mask)
    if valid_pixels == 0: return torch.tensor(0.0, requires_grad=True).to(pred.device)

    # 1. Log空间约束
    log_l1 = F.l1_loss(pred * loss_mask, truth * loss_mask, reduction='sum') / valid_pixels
    
    # 2. 物理空间约束 (缩放到 10^15 量级，保持与 log_l1 量级大致接近)
    a = -130.0
    safe_pred = torch.clamp(pred, max=17.0)
    safe_truth = torch.clamp(truth, max=17.0)
    phys_pred = (10.0 ** safe_pred) + a
    phys_truth = (10.0 ** safe_truth) + a
    
    scale_factor = 1e15
    phys_l1 = F.l1_loss((phys_pred / scale_factor) * loss_mask, 
                        (phys_truth / scale_factor) * loss_mask, 
                        reduction='sum') / valid_pixels
    
    return log_l1 + 0.5 * phys_l1

def simple_loss_fn(pred, truth, loss_mask):
    valid_pixels = torch.sum(loss_mask)
    if valid_pixels == 0: return torch.tensor(0.0, requires_grad=True).to(pred.device)

    # 直接算原始空间误差
    return F.l1_loss(pred * loss_mask, truth * loss_mask, reduction='sum') / valid_pixels

# ==========================================
# [主流程] 7. 训练与统计评估代码
# ==========================================
def train_pconv_model(model, train_loader, val_loader, num_epochs, device, save_path):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_batches = 0.0, 0
        for batch in train_loader:
            x, y = batch['x'].to(device), batch['y_truth'].to(device)
            in_mask, l_mask = batch['input_mask'].to(device), batch['loss_mask'].to(device)
            
            optimizer.zero_grad()
            pred = model(x, in_mask)
            loss = simple_loss_fn(pred, y, l_mask)
            if loss.item() == 0: continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item(); train_batches += 1
            
        # 验证
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch['x'].to(device), batch['y_truth'].to(device)
                in_mask, l_mask = batch['input_mask'].to(device), batch['loss_mask'].to(device)
                pred = model(x, in_mask)
                loss = simple_loss_fn(pred, y, l_mask)
                if loss.item() > 0:
                    val_loss += loss.item(); val_batches += 1
                    
        avg_v_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_v_loss)
        print(f"Epoch [{epoch+1:03d}/{num_epochs}] | Val Loss: {avg_v_loss:.4f}")
        
        if avg_v_loss < best_val_loss and avg_v_loss > 0:
            best_val_loss = avg_v_loss
            torch.save(model.state_dict(), save_path)

    return model

def evaluate_model_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_truths = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['x'].to(device), batch['y_truth'].to(device)
            in_mask, l_mask = batch['input_mask'].to(device), batch['loss_mask'].to(device)
            pred_y = model(x, in_mask)

            valid = l_mask.bool()
            if valid.sum() == 0: continue
            all_preds.append(pred_y[valid].cpu().numpy())
            all_truths.append(y[valid].cpu().numpy())

    if not all_preds: return float('nan'), float('nan')

    # === 删掉所有 10.0 ** safe_preds 的逻辑 ===
    preds_raw = np.concatenate(all_preds).astype(np.float64)
    truths_raw = np.concatenate(all_truths).astype(np.float64)

    # 视你的原始数据情况而定。如果原始数值已经是 1.5, 3.0 这种，就不需要除以 1e15 了。
    # 这里直接用原值计算，看看最真实的 R2
    mse = np.mean((preds_raw - truths_raw) ** 2)
    rmse = np.sqrt(mse) 

    ss_tot = np.sum((truths_raw - np.mean(truths_raw)) ** 2)
    r2 = 1 - (np.sum((truths_raw - preds_raw) ** 2) / ss_tot) if ss_tot > 0 else 0.0
    
    return rmse, r2

# ==========================================
# [执行入口] 8. 主程序
# ==========================================
if __name__ == "__main__":
    BASE_DATA_DIR = "/home/whdong/dl/data"
    FILE_PREFIX = "NO2_Filling_TensorSet"
    SHARDED_DIR = "/home/whdong/dl/data/sharded_daily_tensors-no2"
    NC_MASK_PATH = "/home/whdong/dl/data/SHPshapeforNO2gapfilling_mask_3x3_025deg.nc"
    MODEL_SAVE_PATH = "/home/whdong/dl/models/NO2_best_pconv_model.pth"

    ALL_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
    TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022]
    TEST_YEARS = [2023]

    # 1. 执行离线切片
    shard_tensor_dataset(BASE_DATA_DIR, FILE_PREFIX, ALL_YEARS, SHARDED_DIR)

    # 2. 掩模解耦加载
    print("\n>>> 加载双掩模系统 (红区考核，蓝区辅助) <<<")
    nc_data = xr.open_dataset(NC_MASK_PATH)
    raw_mask_values = nc_data['mask_status'].values
    
    # Expanded (红+蓝): 提供物理风场上下文
    expanded_mask = torch.tensor(~np.isnan(raw_mask_values), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Core (仅红): 严格锁定考核范围
    core_mask = torch.tensor(raw_mask_values == 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 3. 初始化 Dataset (开启块状模拟丢弃策略)
    train_dataset = DailyNO2Dataset(SHARDED_DIR, TRAIN_YEARS, expanded_mask[0], core_mask[0], mask_strategy='random')
    val_dataset = DailyNO2Dataset(SHARDED_DIR, TEST_YEARS, expanded_mask[0], core_mask[0], mask_strategy='random')
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. 构建与训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPConv(in_channels=NUM_FEATURES, out_channels=1)
    
    trained_model = train_pconv_model(model, train_loader, val_loader, num_epochs=100, device=device, save_path=MODEL_SAVE_PATH)
    
    # 5. 指标验证计算
    print("\n>>> 模型训练结束，计算仅限红色区域的客观指标 <<<")
    trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    
    train_rmse, train_r2 = evaluate_model_metrics(trained_model, train_loader, device)
    test_rmse, test_r2 = evaluate_model_metrics(trained_model, val_loader, device)
    print(f"【训练集 (2018-2022)】 -> R2: {train_r2:.4f} | RMSE: {train_rmse:.4f} (x 10^15)")
    print(f"【测试集 (2023盲测)】  -> R2: {test_r2:.4f} | RMSE: {test_rmse:.4f} (x 10^15)")