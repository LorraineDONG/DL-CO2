import pandas as pd
import numpy as np
import optuna
import logging
import os
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 全局配置与日志初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/tabnet_training.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/optuna_tabnet_study.db'

if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 数据加载与特征工程
# ==========================================
def load_and_preprocess(file_path):
    logger.info(f"📂 正在加载数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    
    # 特征工程
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    return df_clean

# ==========================================
# 2. Optuna 深度超参数优化 (TabNet 专属版)
# ==========================================
def optimize_tabnet(X_train, y_train, X_val, y_val, n_trials=50):
    def objective(trial):
        # --- TabNet 核心超参数空间 ---
        
        # 1. 网络宽度 (n_d 和 n_a 通常设置相等，控制决策和注意力的维度)
        n_da = trial.suggest_int('n_da', 8, 64, step=8)
        
        # 2. 网络深度 (决策步数，特征越复杂步数可以越深)
        n_steps = trial.suggest_int('n_steps', 3, 8)
        
        # 3. 稀疏性正则化 (非常关键！控制特征选择的严厉程度)
        lambda_sparse = trial.suggest_float('lambda_sparse', 1e-6, 1e-2, log=True)
        
        # 4. 特征重用衰减因子 (1.0 到 2.0)
        gamma = trial.suggest_float('gamma', 1.0, 2.0)
        
        # 5. 学习率与优化器权重衰减
        lr = trial.suggest_float('lr', 1e-3, 5e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        # 初始化模型
        model = TabNetRegressor(
            n_d=n_da, 
            n_a=n_da,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr, weight_decay=weight_decay),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax', # entmax 比 sparsemax 更平滑
            verbose=0,
            seed=42
        )
        
        # 训练过程 (使用 patience 实现早停)
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['valid'],
            eval_metric=['rmse'],
            max_epochs=100,      # 调参时限制最大轮数以节省时间
            patience=15,         # 15轮不降则早停
            batch_size=1024,     # 表格数据一般用大 batch
            virtual_batch_size=128
        )
        
        # TabNet 会自动加载 early stopping 时的最佳权重
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    logger.info("🚀 开始 TabNet 深度参数模拟 (Optuna)... 预计较慢，请耐心等待。")
    study = optuna.create_study(direction='minimize', storage=DB_FILE, load_if_exists=True)
    # 因为深度学习慢，先设 n_trials=50 试水
    study.optimize(objective, n_trials=n_trials) 
    
    logger.info("="*40)
    logger.info(f"🏆 TabNet 最佳 Val RMSE: {study.best_value:.4f}")
    logger.info(f"最佳参数: {study.best_params}")
    return study.best_params

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/gridded0.25_xco2sf_sif_no2_era5_ndvi_meic_ntl.pkl'
    target = 'xco2_enhanced'
    
    golden_features = [
        'era5_wind_dir_100m', 'era5_wind_speed_100m', 'era5_tcwv', 'era5_ssrd', 'era5_blh', 
        'era5_v100', 'era5_u100', 'era5_t2m', 'ndvi', 'ndvi_std', 'sif_time_tai93', 'sif_sza', 
        'sif_uncertainty', 'sif_vza', 'sif_raz', 'no2_trop', 'no2_vaa', 'no2_vza', 'no2_sza', 
        'no2_amf_trop', 'no2_trop_std', 'no2_time', 'mean_hour', 'month', 'season', 
        'ssrd_t2m_cross', 'ntl_nox_cross'
    ]

    # 1. 准备数据
    df = load_and_preprocess(file_path)
    
    train_df = df[df['year'] <= 2020]
    val_df   = df[df['year'] == 2021]
    test_df  = df[df['year'] >= 2022]

    X_train_raw = train_df[golden_features].values
    X_val_raw   = val_df[golden_features].values
    X_test_raw  = test_df[golden_features].values

    # ⚠️ 深度学习必做：特征标准化 StandardScaler
    logger.info("⚖️ 正在进行特征标准化 (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val   = scaler.transform(X_val_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    # ⚠️ TabNet 要求目标变量必须是二维列向量 (N, 1)
    y_train = train_df[target].values.reshape(-1, 1).astype(np.float32)
    y_val   = val_df[target].values.reshape(-1, 1).astype(np.float32)
    y_test  = test_df[target].values.reshape(-1, 1).astype(np.float32)

    # 2. 自动调参
    best_params = optimize_tabnet(X_train, y_train, X_val, y_val, n_trials=100)

    # 3. 终极盲测阶段
    logger.info("🏁 执行 TabNet 盲测集评估...")
    
    # 提取调参结果并展开
    final_n_da = best_params.pop('n_da')
    final_lr = best_params.pop('lr')
    final_wd = best_params.pop('weight_decay')
    
    final_model = TabNetRegressor(
        n_d=final_n_da, n_a=final_n_da,
        **best_params,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=final_lr, weight_decay=final_wd),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=1,
        seed=42
    )
    
    # 盲测阶段增加 epoch，让它充分收敛
    final_model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['valid'],
        eval_metric=['rmse'],
        max_epochs=200, 
        patience=30,
        batch_size=1024, virtual_batch_size=128
    )
    
    y_pred = final_model.predict(X_test)
    
    # --- 指标计算 (四维体系) ---
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_bias = np.mean(y_pred - y_test)

    # --- 提取 TabNet 特征重要性 ---
    # TabNet 内部通过 Attention Mask 提取重要性，且默认已归一化（总和为 1.0）
    importance_df = pd.DataFrame({
        'Feature': golden_features,
        'Importance (Normalized)': final_model.feature_importances_
    }).sort_values(by='Importance (Normalized)', ascending=False)
    
    # 输出报告
    logger.info("="*30 + " TABNET FINAL REPORT " + "="*30)
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    logger.info("-" * 25 + " 气象与卫星因子贡献度排名前 10 " + "-" * 25)
    
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['Feature']:>20} : {row['Importance (Normalized)']:.4f}")
        
    logger.info("="*81)

