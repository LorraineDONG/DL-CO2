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
from sklearn.model_selection import train_test_split, KFold
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局配置、设备与日志初始化
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SHP_tabnet_training(A01-No lead and lag).log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SHP_optuna_tabnet_study-A01.db'
PARAMS_JSON = '/home/whdong/dl/best_params/train_tabnet_xco2en_SHP_best_params-A01.json'   
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-tabnet_model-A01.zip'     
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-tabnet_scaler-A01.pkl'
Y_SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-tabnet_y_scaler-A01.pkl'

# 自动创建不存在的文件夹
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE.replace('sqlite:///', '')), exist_ok=True)
os.makedirs(os.path.dirname(PARAMS_JSON), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)              

if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 自动检测设备
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🖥️ 当前计算设备已自动设置为: {device.upper()}")

# ==========================================
# 1. 数据加载与特征工程 (与 RF/LGB 完全对齐)
# ==========================================
def load_and_preprocess(file_path):
    logger.info(f"📂 正在加载数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    
    # 确保数据排序
    df_clean.sort_values('date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    # --- 时间周期性编码 ---
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    
    # --- 交叉特征计算 ---
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    
    df_clean['no2_trop_log'] = np.log1p(np.maximum(df_clean['no2_trop'], 0))
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    
    df_clean = df_clean.astype({col: 'float32' for col in df_clean.select_dtypes(include='float64').columns})

    return df_clean

# ==========================================
# 2. Optuna + KFold 深度超参数优化 (注入 RF CV 哲学)
# ==========================================
def optimize_tabnet(X_pool, y_pool, n_trials=50):
    def objective(trial):
        # TabNet 专属超参数空间
        n_da = trial.suggest_int('n_da', 8, 32, step=8)
        n_steps = trial.suggest_int('n_steps', 3, 8)
        lambda_sparse = trial.suggest_float('lambda_sparse', 1e-6, 1e-2, log=True)
        gamma = trial.suggest_float('gamma', 1.0, 2.0)
        lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_rmses = []
        
        # 使用滑窗交叉验证与 RF/LGB 对齐
        for train_index, val_index in kf.split(X_pool):
            X_tr_raw, X_va_raw = X_pool[train_index], X_pool[val_index]
            y_tr_raw, y_va_raw = y_pool[train_index], y_pool[val_index]
            
            # 严格在每次 CV 内独立进行标准化 (特征和目标)
            cv_x_scaler = StandardScaler()
            X_tr = cv_x_scaler.fit_transform(X_tr_raw).astype(np.float32)
            X_va = cv_x_scaler.transform(X_va_raw).astype(np.float32)
            
            cv_y_scaler = StandardScaler()
            y_tr = cv_y_scaler.fit_transform(y_tr_raw.reshape(-1, 1)).astype(np.float32)
            y_va = cv_y_scaler.transform(y_va_raw.reshape(-1, 1)).astype(np.float32)

            model = TabNetRegressor(
                n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=lr, weight_decay=weight_decay),
                scheduler_params={"mode": "min", "patience": 5, "factor": 0.5},
                scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                mask_type='entmax', verbose=0, seed=29, device_name=device
            )
            
            model.fit(
                X_train=X_tr, y_train=y_tr,
                eval_set=[(X_va, y_va)],
                eval_name=['valid'], eval_metric=['rmse'],
                loss_fn=torch.nn.SmoothL1Loss(),
                max_epochs=100, patience=8,         
                batch_size=4096, virtual_batch_size=1024
            )
            
            # 使用预测结果反标准化，以真实量纲评估 RMSE
            preds_scaled = model.predict(X_va)
            preds = cv_y_scaler.inverse_transform(preds_scaled)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va_raw, preds)))
            
        return np.mean(cv_rmses)

    logger.info(f"🚀 开始 TabNet KFold 深度参数模拟 (Optuna) | 设备: {device}...")
    study = optuna.create_study(direction='minimize', storage=DB_FILE, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials) 
    
    return study.best_params

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    target = 'xco2_enhanced'
    
    # 🌟 直接使用与 RF/LGB 完全一致的特征集合
    # golden_features = [
    #     'era5_u100', 'era5_v100', 'grid_lon', 'grid_lat',
    #     'sif_740', 'no2_trop', 'meic_nox', 'dem_mean',
    #     'era5_tcwv', 'era5_ssrd', 'era5_blh', 'era5_t2m', 
    #     'ntl', 'ndvi', 'ndvi_std', 
    #     'month_sin', 'month_cos', 
    #     'sif_variance', 'era5_wind_speed',
    #     'no2_amf_trop', 'no2_variance', 
    #     'ssrd_t2m_cross', 'ntl_nox_cross', 'ndvi_t2m_cross'
    # ]

    golden_features= [
        'era5_blh', 'era5_d2m', 'era5_sp', 'era5_ssrd', 'era5_t2m', 'era5_tcwv', 
        'era5_u100', 'era5_v100', 'era5_u10', 'era5_v10','era5_wind_speed',
        'grid_lat', 'grid_lon', 'dem_mean', 
        'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
        'ndvi_t2m_cross', 'ssrd_t2m_cross', 'ntl_nox_cross',
        'meic_nox', 'ntl', 'ndvi','no2_trop_log',
    ]

    # 1. 准备全局数据
    df = load_and_preprocess(file_path)
    X_full_raw = df[golden_features].values
    y_full_raw = df[target].values

    # 2. 数据划分：切分出 20% 绝对盲测集 (完全冻结)
    X_pool_raw, X_test_raw, y_pool_raw, y_test_raw = train_test_split(
        X_full_raw, y_full_raw, test_size=0.2, random_state=42
    )

    logger.info(f"✨ 输入特征组合 ({len(golden_features)}个): {golden_features}")
    
    # 3. 自动调参 (仅使用 Pool 数据，函数内部会处理标准化)
    best_params = optimize_tabnet(X_pool_raw, y_pool_raw, n_trials=500)

    # 保存最优参数
    import json
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)

    # ==========================================
    # 4. 终极盲测评估 (严格修复早停验证集数据泄露)
    # ==========================================
    logger.info("🏁 阶段三：执行终极盲测集评估，并对比训练集性能...")
    
    # 【修复点】：先切分 early stopping 验证集，然后再分别进行标准化
    X_train_es_raw, X_val_es_raw, y_train_es_raw, y_val_es_raw = train_test_split(
        X_pool_raw, y_pool_raw, test_size=0.1, random_state=42
    )

    # 严格按照 Train (ES) 数据拟合 Scaler
    eval_x_scaler = StandardScaler()
    X_train_es = eval_x_scaler.fit_transform(X_train_es_raw).astype(np.float32)
    X_val_es = eval_x_scaler.transform(X_val_es_raw).astype(np.float32)
    X_test = eval_x_scaler.transform(X_test_raw).astype(np.float32)
    # 转换整个 Pool 以便计算最终 Train 指标
    X_pool_for_eval = eval_x_scaler.transform(X_pool_raw).astype(np.float32)

    eval_y_scaler = StandardScaler()
    y_train_es = eval_y_scaler.fit_transform(y_train_es_raw.reshape(-1, 1)).astype(np.float32)
    y_val_es = eval_y_scaler.transform(y_val_es_raw.reshape(-1, 1)).astype(np.float32)

    final_n_da = best_params.pop('n_da')
    final_lr = best_params.pop('lr')
    final_wd = best_params.pop('weight_decay')
    
    eval_model = TabNetRegressor(
        n_d=final_n_da, n_a=final_n_da,
        **best_params,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=final_lr, weight_decay=final_wd),
        scheduler_params={"mode": "min", "patience": 10, "factor": 0.5},
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax', verbose=0, seed=29, device_name=device
    )

    # 拟合盲测模型
    eval_model.fit(
        X_train=X_train_es, y_train=y_train_es,
        eval_set=[(X_val_es, y_val_es)],
        eval_name=['valid'], eval_metric=['rmse'],
        loss_fn=torch.nn.SmoothL1Loss(),
        max_epochs=300, patience=30,
        batch_size=1024, virtual_batch_size=128
    )
    
    # 预测并反标准化
    train_preds_scaled = eval_model.predict(X_pool_for_eval)
    train_preds = eval_y_scaler.inverse_transform(train_preds_scaled).flatten()
    
    test_preds_scaled = eval_model.predict(X_test)
    test_preds = eval_y_scaler.inverse_transform(test_preds_scaled).flatten()

    # 计算指标
    train_r2 = r2_score(y_pool_raw, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_pool_raw, train_preds))

    test_r2 = r2_score(y_test_raw, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test_raw, test_preds))
    test_mae = mean_absolute_error(y_test_raw, test_preds)
    test_bias = np.mean(test_preds - y_test_raw)
    
    # 输出对齐的报告格式
    logger.info("="*30 + " TABNET FINAL REPORT " + "="*30)
    logger.info(f"Train R²  : {train_r2:.4f}")
    logger.info(f"Train RMSE: {train_rmse:.4f} ppm")
    logger.info("-" * 25)
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    logger.info("="*85)

    # ==========================================
    # 5. 训练最终的生产模型 (使用 100% 数据)
    # ==========================================
    logger.info("💾 阶段四：使用 100% 数据训练最终生产模型并固化...")

    # 切分生产模型的 early stopping 验证集 (5% 即可)
    X_prod_tr_raw, X_prod_val_raw, y_prod_tr_raw, y_prod_val_raw = train_test_split(
        X_full_raw, y_full_raw, test_size=0.05, random_state=42
    )

    # 生产模型的独立 Scaler
    production_x_scaler = StandardScaler()
    X_prod_tr = production_x_scaler.fit_transform(X_prod_tr_raw).astype(np.float32)
    X_prod_val = production_x_scaler.transform(X_prod_val_raw).astype(np.float32)

    production_y_scaler = StandardScaler()
    y_prod_tr = production_y_scaler.fit_transform(y_prod_tr_raw.reshape(-1, 1)).astype(np.float32)
    y_prod_val = production_y_scaler.transform(y_prod_val_raw.reshape(-1, 1)).astype(np.float32)

    production_model = TabNetRegressor(
        n_d=final_n_da, n_a=final_n_da,
        **best_params,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=final_lr, weight_decay=final_wd),
        scheduler_params={"mode": "min", "patience": 10, "factor": 0.5},
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax', verbose=1, seed=29, device_name=device
    )

    production_model.fit(
        X_train=X_prod_tr, y_train=y_prod_tr,
        eval_set=[(X_prod_val, y_prod_val)],
        eval_name=['valid'], eval_metric=['rmse'],
        loss_fn=torch.nn.SmoothL1Loss(),
        max_epochs=300, patience=30,
        batch_size=1024, virtual_batch_size=128
    )

    # 保存生产级模型和对应Scaler
    joblib.dump(production_x_scaler, SCALER_SAVE_PATH)
    joblib.dump(production_y_scaler, Y_SCALER_SAVE_PATH)
    save_path_without_ext = MODEL_SAVE_PATH.replace('.zip', '')
    production_model.save_model(save_path_without_ext)
    logger.info(f"✅ 生产级模型与全量标准化器已持久化至: {os.path.dirname(MODEL_SAVE_PATH)}")