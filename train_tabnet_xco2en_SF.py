import pandas as pd
import numpy as np
import optuna
import logging
import os
import torch
import joblib
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 全局配置与日志初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SHP_tabnet_training.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SHP_optuna_tabnet_study.db'
PARAMS_JSON = '/home/whdong/dl/train_tabnet_xco2en_SHP_best_params.json'   # 新增：最优参数保存路径
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-tabnet_model.zip'     # 新增：模型保存路径 (TabNet推荐用zip)
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-tabnet_scaler.pkl'   # 新增：必须同时保存标准化器

# 自动创建不存在的文件夹
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE.replace('sqlite:///', '')), exist_ok=True)
os.makedirs(os.path.dirname(PARAMS_JSON), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)              # 新增：统一创建 models 文件夹

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

    # --- 1. 时间周期性特征 ---
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    
    # --- 2. 大气动力学与扩散条件交叉 ---
    # 计算绝对风速 (m/s)
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    # 通风系数 (Ventilation Coefficient) = 边界层高度 * 风速
    df_clean['ventilation_coef'] = df_clean['era5_blh'] * df_clean['era5_wind_speed']
    
    # --- 3. 人为碳排放源复合交叉 ---
    # 灯光与NOx清单的交叉 (原有的)
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    # 卫星观测NO2与灯光的交叉 (捕捉清单未涵盖的动态人为源)
    df_clean['ntl_no2_cross'] = df_clean['ntl'] * df_clean['no2_trop']
    
    # --- 4. 生物圈碳汇活动交叉 ---
    # 温度与辐射的交叉 (原有的)
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    # SIF与温度的交叉 (高温可能带来水分胁迫，或者适宜温度促进光合)
    df_clean['sif_t2m_cross'] = df_clean['sif_740'] * df_clean['era5_t2m']

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
            # 🌟 关键修改 3：更灵活的学习率调度器
            scheduler_params={"mode": "min", "patience": 5, "factor": 0.5},
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            mask_type='entmax',
            verbose=0,
            seed=29
        )
        
        # 训练过程 (使用 patience 实现早停)
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['valid'],
            eval_metric=['rmse'],
            loss_fn=torch.nn.SmoothL1Loss(),
            max_epochs=100,      # 调参时限制最大轮数以节省时间
            patience=15,         # 15轮不降则早停
            batch_size=512,     # 表格数据一般用大 batch
            virtual_batch_size=64
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
    file_path = '/home/whdong/dl/TABLE-WLGXCO2en_sif_no2_era5_ndvi_meic_ntl_dem.pkl'
    target = 'xco2_enhanced'
    
    golden_features = ['month_sin', 'month_cos', #'doy_sin', 'doy_cos',
                       'era5_t2m', 'era5_u100', 'era5_v100', 'era5_blh', 'era5_ssrd', 'era5_tcwv', 
                       'ntl', 'grid_lon', 'grid_lat', 'dem_mean', 'dem_std', 'ndvi', 'ndvi_std', 
                       'sif_variance', 'sif_740', 'meic_nox', 'no2_amf_trop', 'no2_variance', 'no2_trop',
                       'ventilation_coef', 'ntl_nox_cross', 'ntl_no2_cross', 'ssrd_t2m_cross', 'sif_t2m_cross'
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

    # 🌟 关键修改 1：目标变量 y 同样需要标准化！
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(train_df[target].values.reshape(-1, 1)).astype(np.float32)
    y_val   = y_scaler.transform(val_df[target].values.reshape(-1, 1)).astype(np.float32)
    
    # 测试集 y 保持原始值，用于最终计算指标
    y_test_raw = test_df[target].values.reshape(-1, 1).astype(np.float32)

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
        scheduler_params={"mode": "min", "patience": 10, "factor": 0.5}, # 盲测时耐心可以稍微给多一点
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax',
        verbose=1,
        seed=29
    )
    
    # 盲测阶段增加 epoch，让它充分收敛
    final_model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['valid'],
        eval_metric=['rmse'],
        loss_fn=torch.nn.SmoothL1Loss(),
        max_epochs=200, 
        patience=30,
        batch_size=512, virtual_batch_size=64
    )
    
    y_pred_scaled = final_model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test = y_test_raw # 使用前面保留的原始真实值
    
    # --- 指标计算 (四维体系) ---
    test_r2 = r2_score(y_test, y_pred)
    
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


    joblib.dump(scaler, SCALER_SAVE_PATH)
    save_path_without_ext = MODEL_SAVE_PATH.replace('.zip', '')
    final_model.save_model(save_path_without_ext)
