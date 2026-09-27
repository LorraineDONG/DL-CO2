import pandas as pd
import numpy as np
import optuna
import logging
import os
import json
import joblib
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局配置、设备与日志初始化 (增加 PINN 后缀)
# ==========================================
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SHP_tabnet_training_PINN.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SHP_optuna_tabnet_study-PINN.db'
PARAMS_JSON = '/home/whdong/dl/best_params/train_tabnet_xco2en_SHP_best_params-PINN.json'   
MODEL_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-tabnet_model-PINN.zip'     
SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-tabnet_scaler-PINN.pkl'
Y_SCALER_SAVE_PATH = '/home/whdong/dl/models/XCO2en_SHP-tabnet_y_scaler-PINN.pkl'

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

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🖥️ 当前计算设备已自动设置为: {device.upper()}")


# ==========================================
# 1. 核心大招：物理约束 TabNet (PINN-TabNet)
# ==========================================
class PINNTabNetRegressor(TabNetRegressor):
    """
    继承原生 TabNet，注入基于梯度的单调性物理约束 (Physics-Informed Neural Network)
    """
    def __init__(self, physics_features_idx=None, lambda_physics=0.1, **kwargs):
        super().__init__(**kwargs)
        # 记录需要施加单调约束的特征索引 (例如 NO2 和 排放量)
        self.physics_features_idx = physics_features_idx if physics_features_idx is not None else []
        # 惩罚强度系数
        self.lambda_physics = lambda_physics

    def _train_batch(self, X, y):
        """
        重写底层 Batch 训练逻辑，加入梯度惩罚
        """
        self.network.train()
        self._optimizer.zero_grad()
        
        # 关键点 1：将 X 传入 GPU，并开启梯度追踪 (requires_grad)
        X = X.to(self.device).float().requires_grad_(True)
        y = y.to(self.device).float()
        
        # 前向传播
        output, M_loss = self.network(X)
        
        # 计算基础数据 Loss (例如 SmoothL1)
        loss = self.loss_fn(output, y)
        
        # 加上 TabNet 原生的稀疏性特征选择惩罚
        loss -= self.lambda_sparse * M_loss

        # ====================================================
        # 关键点 2：物理约束 (单调性损失计算)
        # ====================================================
        if len(self.physics_features_idx) > 0 and self.lambda_physics > 0:
            # 求解预测值对输入 X 的偏导数 (梯度)
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=X,
                grad_outputs=torch.ones_like(output),
                create_graph=True,   # 必须为 True，允许梯度反向传播
                retain_graph=True
            )[0]
            
            physics_loss = 0.0
            for idx in self.physics_features_idx:
                # 提取目标特征的梯度
                feat_grad = gradients[:, idx]
                # 单调递增约束：如果梯度 < 0 (NO2上升导致碳下降)，则违反物理常识
                # 取反后用 ReLU 激活：当负梯度出现时，产生正向的惩罚值
                physics_loss += torch.mean(torch.relu(-feat_grad))
                
            # 将物理惩罚加入总 Loss
            loss += self.lambda_physics * physics_loss

        # 反向传播与优化
        loss.backward()
        if self.clip_value:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        loss_value = loss.item()
        # 更新训练记录 (兼容原生接口)
        if hasattr(self, '_train_history'):
            self._train_history['loss'].append(loss_value)
            
        return loss_value


# ==========================================
# 2. 数据加载与特征工程
# ==========================================
def load_and_preprocess(file_path):
    logger.info(f"📂 正在加载数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    
    # 引入代码3的经验：剔除极端值，防止网络强行拟合离群点
    target = 'xco2_enhanced'
    df_clean = df_clean[df_clean[target] <= 10].reset_index(drop=True)
    logger.info(f"🧹 经过筛选 (xco2_enhanced <= 10) 后，当前样本量为: {len(df_clean)}")

    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    df_clean.sort_values('date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    
    df_clean['no2_trop_log'] = np.log1p(np.maximum(df_clean['no2_trop'], 0))
    if 'no2_trop_mean' not in df_clean.columns:
        df_clean['no2_trop_mean'] = df_clean['no2_trop'] # 兜底逻辑
        
    df_clean['era5_wind_speed'] = np.sqrt(df_clean['era5_u100']**2 + df_clean['era5_v100']**2)
    df_clean = df_clean.astype({col: 'float32' for col in df_clean.select_dtypes(include='float64').columns})

    return df_clean

# ==========================================
# 3. Optuna 深度超参数优化 (注入 PINN)
# ==========================================
def optimize_tabnet_pinn(X_pool, y_pool, physics_idx, n_trials=50):
    def objective(trial):
        n_da = trial.suggest_int('n_da', 16, 64, step=8) 
        n_steps = trial.suggest_int('n_steps', 3, 7)      
        lambda_sparse = trial.suggest_float('lambda_sparse', 1e-4, 1e-1, log=True) 
        # 新增物理惩罚系数搜索范围
        lambda_physics = trial.suggest_float('lambda_physics', 1e-3, 0.5, log=True)
        gamma = trial.suggest_float('gamma', 1.0, 2.0)
        lr = trial.suggest_float('lr', 1e-3, 5e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42) # 考虑到PINN计算慢，折数降到5
        cv_rmses = []
        
        for train_index, val_index in kf.split(X_pool):
            X_tr_raw, X_va_raw = X_pool[train_index], X_pool[val_index]
            y_tr_raw, y_va_raw = y_pool[train_index], y_pool[val_index]
            
            cv_x_scaler = StandardScaler()
            X_tr = cv_x_scaler.fit_transform(X_tr_raw).astype(np.float32)
            X_va = cv_x_scaler.transform(X_va_raw).astype(np.float32)
            
            cv_y_scaler = StandardScaler()
            y_tr = cv_y_scaler.fit_transform(y_tr_raw.reshape(-1, 1)).astype(np.float32)
            y_va = cv_y_scaler.transform(y_va_raw.reshape(-1, 1)).astype(np.float32)

            # 实例化我们自定义的物理约束网络
            model = PINNTabNetRegressor(
                physics_features_idx=physics_idx,
                lambda_physics=lambda_physics,
                n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=lr, weight_decay=weight_decay),
                scheduler_params={"mode": "min", "patience": 8, "factor": 0.5},
                scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                mask_type='entmax', verbose=0, seed=42, device_name=device
            )
            
            model.fit(
                X_train=X_tr, y_train=y_tr,
                eval_set=[(X_va, y_va)],
                eval_name=['valid'], eval_metric=['rmse'],
                loss_fn=torch.nn.SmoothL1Loss(),
                max_epochs=150, patience=20,         
                batch_size=4096, virtual_batch_size=1024
            )
            
            preds_scaled = model.predict(X_va)
            preds = cv_y_scaler.inverse_transform(preds_scaled)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va_raw, preds)))
            
        return np.mean(cv_rmses)

    logger.info(f"🚀 开始 PINN-TabNet KFold 物理约束参数模拟 | 设备: {device}...")
    study = optuna.create_study(direction='minimize', storage=DB_FILE, load_if_exists=True, study_name='pinn_tabnet_01')
    study.optimize(objective, n_trials=n_trials) 
    
    return study.best_params


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 使用代码3的数据集 (包含了 no2_trop_mean 等特征)
    file_path = '/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl'
    target = 'xco2_enhanced'
    
    # 结合代码3的经验，去掉冗余的 lag 和 lead 特征，保留 Top 物理贡献特征
    golden_features = [
        'era5_blh', 'era5_d2m', 'era5_sp', 'era5_ssrd', 'era5_t2m', 'era5_tcwv', 
        'era5_u100', 'era5_u10', 'era5_v100', 'era5_v10', 'era5_wind_speed', 
        'no2_trop_log', 'no2_trop_mean', 
        'ndvi_t2m_cross', 'ssrd_t2m_cross', 'ntl_nox_cross',
        'meic_nox', 'ndvi', 'ntl', 'dem_mean',
        'doy_sin', 'doy_cos', 'month_sin', 'month_cos', 
        'grid_lon', 'grid_lat'
    ]

    # 确定需要进行单调性约束的特征在列表中的索引
    monotone_vars = ['no2_trop_mean', 'no2_trop_log', 'meic_nox']
    physics_idx = [golden_features.index(f) for f in monotone_vars if f in golden_features]
    logger.info(f"📐 物理约束监控特征索引已锁定: {dict(zip(monotone_vars, physics_idx))}")

    df = load_and_preprocess(file_path)
    X_full_raw = df[golden_features].values
    y_full_raw = df[target].values

    # 切分出 20% 绝对盲测集 (完全冻结)
    X_pool_raw, X_test_raw, y_pool_raw, y_test_raw = train_test_split(
        X_full_raw, y_full_raw, test_size=0.2, random_state=42
    )
    
    # 自动调参
    best_params = optimize_tabnet_pinn(X_pool_raw, y_pool_raw, physics_idx, n_trials=50)

    # 保存最优参数
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)

    # ==========================================
    # 5. 终极盲测评估 
    # ==========================================
    logger.info("🏁 阶段三：执行终极盲测集评估...")
    
    X_train_es_raw, X_val_es_raw, y_train_es_raw, y_val_es_raw = train_test_split(
        X_pool_raw, y_pool_raw, test_size=0.1, random_state=42
    )

    eval_x_scaler = StandardScaler()
    X_train_es = eval_x_scaler.fit_transform(X_train_es_raw).astype(np.float32)
    X_val_es = eval_x_scaler.transform(X_val_es_raw).astype(np.float32)
    X_test = eval_x_scaler.transform(X_test_raw).astype(np.float32)
    X_pool_for_eval = eval_x_scaler.transform(X_pool_raw).astype(np.float32)

    eval_y_scaler = StandardScaler()
    y_train_es = eval_y_scaler.fit_transform(y_train_es_raw.reshape(-1, 1)).astype(np.float32)
    y_val_es = eval_y_scaler.transform(y_val_es_raw.reshape(-1, 1)).astype(np.float32)

    final_n_da = best_params.pop('n_da')
    final_lr = best_params.pop('lr')
    final_wd = best_params.pop('weight_decay')
    final_lambda_physics = best_params.pop('lambda_physics')
    
    eval_model = PINNTabNetRegressor(
        physics_features_idx=physics_idx,
        lambda_physics=final_lambda_physics,
        n_d=final_n_da, n_a=final_n_da,
        **best_params,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=final_lr, weight_decay=final_wd),
        scheduler_params={"mode": "min", "patience": 10, "factor": 0.5},
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax', verbose=0, seed=42, device_name=device
    )

    eval_model.fit(
        X_train=X_train_es, y_train=y_train_es,
        eval_set=[(X_val_es, y_val_es)],
        eval_name=['valid'], eval_metric=['rmse'],
        loss_fn=torch.nn.SmoothL1Loss(),
        max_epochs=200, patience=25,
        batch_size=1024, virtual_batch_size=128
    )
    
    train_preds_scaled = eval_model.predict(X_pool_for_eval)
    train_preds = eval_y_scaler.inverse_transform(train_preds_scaled).flatten()
    
    test_preds_scaled = eval_model.predict(X_test)
    test_preds = eval_y_scaler.inverse_transform(test_preds_scaled).flatten()

    train_r2 = r2_score(y_pool_raw, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_pool_raw, train_preds))

    test_r2 = r2_score(y_test_raw, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test_raw, test_preds))
    test_mae = mean_absolute_error(y_test_raw, test_preds)
    test_bias = np.mean(test_preds - y_test_raw)
    
    logger.info("="*30 + " PINN-TABNET FINAL REPORT " + "="*30)
    logger.info(f"Train R²  : {train_r2:.4f}")
    logger.info(f"Train RMSE: {train_rmse:.4f} ppm")
    logger.info("-" * 25)
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    logger.info("="*86)

    # ==========================================
    # 6. 训练最终的生产模型 (使用 100% 数据)
    # ==========================================
    logger.info("💾 阶段四：使用 100% 数据训练最终生产模型并固化...")

    X_prod_tr_raw, X_prod_val_raw, y_prod_tr_raw, y_prod_val_raw = train_test_split(
        X_full_raw, y_full_raw, test_size=0.05, random_state=42
    )

    production_x_scaler = StandardScaler()
    X_prod_tr = production_x_scaler.fit_transform(X_prod_tr_raw).astype(np.float32)
    X_prod_val = production_x_scaler.transform(X_prod_val_raw).astype(np.float32)

    production_y_scaler = StandardScaler()
    y_prod_tr = production_y_scaler.fit_transform(y_prod_tr_raw.reshape(-1, 1)).astype(np.float32)
    y_prod_val = production_y_scaler.transform(y_prod_val_raw.reshape(-1, 1)).astype(np.float32)

    production_model = PINNTabNetRegressor(
        physics_features_idx=physics_idx,
        lambda_physics=final_lambda_physics,
        n_d=final_n_da, n_a=final_n_da,
        **best_params,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=final_lr, weight_decay=final_wd),
        scheduler_params={"mode": "min", "patience": 10, "factor": 0.5},
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax', verbose=1, seed=42, device_name=device
    )

    production_model.fit(
        X_train=X_prod_tr, y_train=y_prod_tr,
        eval_set=[(X_prod_val, y_prod_val)],
        eval_name=['valid'], eval_metric=['rmse'],
        loss_fn=torch.nn.SmoothL1Loss(),
        max_epochs=300, patience=30,
        batch_size=1024, virtual_batch_size=128
    )

    joblib.dump(production_x_scaler, SCALER_SAVE_PATH)
    joblib.dump(production_y_scaler, Y_SCALER_SAVE_PATH)
    save_path_without_ext = MODEL_SAVE_PATH.replace('.zip', '')
    production_model.save_model(save_path_without_ext)
    logger.info(f"✅ 生产级 PINN-TabNet 模型已持久化至: {os.path.dirname(MODEL_SAVE_PATH)}")

    # 打印特征重要性
    feature_importances = production_model.feature_importances_
    if feature_importances is not None and len(feature_importances) == len(golden_features):
        importance_df = pd.DataFrame({
            'Feature': golden_features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        logger.info("="*30 + " 特征贡献度排名 (Top 20) " + "="*30)
        for i, row in importance_df.head(20).iterrows():
            logger.info(f"{i+1:2d}. {row['Feature']:<25}: {row['Importance']:.6f}")
        logger.info("="*85)