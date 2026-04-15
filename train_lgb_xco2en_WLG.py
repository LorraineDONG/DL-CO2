import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import logging
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV

# ==========================================
# 0. 全局配置与路径初始化
# ==========================================
# 修正了文件名，防止和 Random Forest 的日志混淆冲突
LOG_FILE = '/home/whdong/dl/logfile/XCO2en_SF_lgb_training.log'
DB_FILE = 'sqlite:////home/whdong/dl/dbfile/XCO2en_SF_optuna_lgb_study.db' 
PARAMS_JSON = '/home/whdong/dl/train_lgb_xco2en_SF_best_params.json'

# 自动创建不存在的文件夹，防止 SQLite 或日志报错
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE.replace('sqlite:///', '')), exist_ok=True)
os.makedirs(os.path.dirname(PARAMS_JSON), exist_ok=True)

if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 数据加载与时间序列排序
# ==========================================
def load_and_preprocess(file_path):
    logger.info(f"📂 正在加载数据: {file_path}...")
    df = pd.read_pickle(file_path)
    df_clean = df.dropna().copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    
    # ⚠️ 关键修改：为了使用 TimeSeriesSplit，必须确保数据严格按时间顺序排列
    df_clean.sort_values('date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    # --- 时间特征工程：提取基础周期 ---
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['doy'] = df_clean['date'].dt.dayofyear # 提取一年中的第几天 (1-365)
    
    # --- 🌟 时间周期性编码 (Sine / Cosine Transformation) ---
    # 1. 月份的周期性编码 (周期为 12)
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12.0)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12.0)
    
    # 2. DOY 的周期性编码 (周期为 365.25，考虑闰年)
    df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
    df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)

    # 保留原来的季节划分，有时树模型依然喜欢这种粗粒度的分类
    df_clean['season'] = (df_clean['month'] % 12 + 3) // 3
    
    # --- 物理交叉特征 ---
    logger.info("🛠️ 执行特征工程: 周期性编码 + 物理交叉项...")
    df_clean['ndvi_t2m_cross'] = df_clean['ndvi'] * df_clean['era5_t2m']
    df_clean['ssrd_t2m_cross'] = df_clean['era5_ssrd'] * df_clean['era5_t2m']
    df_clean['ntl_nox_cross'] = df_clean['ntl'] * df_clean['meic_nox']
    
    return df_clean

# ==========================================
# 2. 受限特征淘汰赛 (RFE) - 保护核心物理变量
# ==========================================
def recursive_feature_elimination(X_train, y_train, X_val, y_val, candidate_features, must_keep):
    logger.info("🕵️‍♂️ 启动特征淘汰赛 (仅筛选非核心特征)...")
    current_candidates = list(candidate_features)
    best_r2 = -np.inf
    best_subset = current_candidates.copy()

    # 只要候选特征还多于16个，就继续淘汰
    while len(current_candidates) > 18:
        # 每次训练都带上必须保留的特征
        features_to_test = must_keep + current_candidates
        
        model = lgb.LGBMRegressor(
                n_estimators=2000,  # 修改特征淘汰赛 (RFE) 的训练轮次
                learning_rate=0.03,  
                random_state=42, 
                n_jobs=-1, 
                verbose=-1
                )
        model.fit(X_train[features_to_test], y_train, eval_set=[(X_val[features_to_test], y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        
        preds = model.predict(X_val[features_to_test])
        curr_r2 = r2_score(y_val, preds)
        
        # 记录验证集效果最好的那一次候选特征组合
        if curr_r2 > best_r2:
            best_r2 = curr_r2
            best_subset = current_candidates.copy()
            
        # 提取特征重要性
        importances = model.booster_.feature_importance(importance_type='gain')
        
        # ⚠️ 关键：分离出只属于 candidate_features 的重要性 (因为 must_keep 排在前面)
        candidate_importances = importances[len(must_keep):]
        
        # 剔除候选特征中最弱的一个
        worst_idx = np.argmin(candidate_importances)
        worst_feat = current_candidates.pop(worst_idx)
        logger.info(f"剩余候选特征数: {len(current_candidates)} | Val R²: {curr_r2:.4f} | 剔除: {worst_feat}")

    logger.info(f"🏆 筛选完成！最佳候选特征数量: {len(best_subset)}")
    return must_keep + best_subset


def recursive_feature_elimination_tscv(X_pool, y_pool, candidate_features, must_keep, n_splits=3):
    logger.info(f"🕵️‍♂️ 启动带时间滑窗的特征淘汰赛 (CV={n_splits})...")
    current_candidates = list(candidate_features)
    best_r2 = -np.inf
    best_subset = current_candidates.copy()

    # 初始化时间序列滑窗
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 只要候选特征还多于18个，就继续淘汰 (可根据你的实际情况调整阈值)
    while len(current_candidates) > 18:
        features_to_test = must_keep + current_candidates
        X_curr = X_pool[features_to_test]
        
        cv_r2s = []
        # 初始化一个全零数组，用于累加每一折的特征重要性 (Gain)
        total_importances = np.zeros(len(features_to_test))
        
        # 🌟 对当前特征组合进行 TimeSeriesSplit 多折验证
        for train_index, val_index in tscv.split(X_curr):
            X_tr, X_va = X_curr.iloc[train_index], X_curr.iloc[val_index]
            y_tr, y_va = y_pool.iloc[train_index], y_pool.iloc[val_index]
            
            model = lgb.LGBMRegressor(
                n_estimators=2000,  
                learning_rate=0.03,  
                random_state=42, 
                n_jobs=-1, 
                verbose=-1
            )
            
            model.fit(
                X_tr, y_tr, 
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            preds = model.predict(X_va)
            cv_r2s.append(r2_score(y_va, preds))
            
            # 累加这一折的特征重要性
            total_importances += model.booster_.feature_importance(importance_type='gain')
            
        # 计算多折的平均指标
        avg_r2 = np.mean(cv_r2s)
        avg_importances = total_importances / n_splits
        
        # 记录跨年份平均效果最好的那一次候选特征组合
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_subset = current_candidates.copy()
            
        # ⚠️ 分离出只属于 candidate_features 的重要性 
        candidate_importances = avg_importances[len(must_keep):]
        
        # 依据平均重要性，剔除候选特征中最弱的一个
        worst_idx = np.argmin(candidate_importances)
        worst_feat = current_candidates.pop(worst_idx)
        
        logger.info(f"剩余候选特征数: {len(current_candidates)} | CV 平均 R²: {avg_r2:.4f} | 剔除: {worst_feat}")

    logger.info(f"🏆 筛选完成！最佳 CV R²: {best_r2:.4f} | 最佳候选特征数量: {len(best_subset)}")
    return must_keep + best_subset


# ==========================================
# 3. Optuna + TimeSeriesSplit 深度调参
# ==========================================
def optimize_hyperparameters(X_pool, y_pool, n_trials=100):
    def objective(trial):
        param = {
            'objective': 'huber',  # 使用 Huber 损失函数抵抗极端异常值
            'n_estimators': 5000,   # 修改 Optuna 调参时的训练轮次
            # Optuna 寻找最优的“初始学习率”
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # ⚠️ 关键修改：时间序列滑窗验证 (3折)，提升对未来年际变化的泛化能力
        tscv = TimeSeriesSplit(n_splits=3)
        cv_rmses = []
        
        for train_index, val_index in tscv.split(X_pool):
            X_tr, X_va = X_pool.iloc[train_index], X_pool.iloc[val_index]
            y_tr, y_va = y_pool.iloc[train_index], y_pool.iloc[val_index]
            
            model = lgb.LGBMRegressor(**param)
            model.fit(
                X_tr, y_tr, 
                eval_set=[(X_va, y_va)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            preds = model.predict(X_va)
            cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
            
        # 返回多次滑窗的平均 RMSE
        return np.mean(cv_rmses)

    logger.info("🚀 开始带时间滑窗的深度参数模拟 (Optuna)...")
    study = optuna.create_study(direction='minimize', storage=DB_FILE, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    file_path = '/home/whdong/dl/TABLE-WLGXCO2en_sif_no2_era5_ndvi_meic_ntl.pkl'
    target = 'xco2_enhanced'

    # 🌟 1. 物理上绝对不能踢掉的核心特征（强制保留）
    must_keep_features = [
            'era5_tcwv', 'era5_ssrd', 
            'era5_blh', 'era5_t2m', 
            'era5_u100', 'era5_v100', 
            'sif_740', 'no2_trop',    
            'meic_nox', 'grid_lat', 'grid_lon'                
    ]

    # 🌟 2. 参与“淘汰赛”的候选特征
    candidate_features = [
        'era5_wind_dir_100m', 'era5_wind_speed_100m', 
        'ntl', 'ndvi', 'ndvi_std', 'year',
        # 'month_sin', 'month_cos', 
        'doy_sin', 'doy_cos',
        'sif_sza', 'sif_variance', 'sif_vza',
        'no2_vza', 'no2_sza', 'no2_amf_trop', 'no2_variance', 
        'ndvi_t2m_cross', 'ssrd_t2m_cross', 'ntl_nox_cross'
    ]

    # 1. 准备数据
    df = load_and_preprocess(file_path)
    
    # 划分调参池和盲测集
    train_val_pool = df[df['year'] <= 2021]
    test_df        = df[df['year'] >= 2022]

    # 2. 受限特征淘汰赛
    # rfe_train = train_val_pool[train_val_pool['year'] <= 2020]
    # rfe_val   = train_val_pool[train_val_pool['year'] == 2021]
    # golden_features = recursive_feature_elimination(
    #     rfe_train, rfe_train[target], rfe_val, rfe_val[target], candidate_features, must_keep_features
    # )  # 注释的这4行是之前recursive_feature_elimination的版本

    X_train_val = train_val_pool[must_keep_features + candidate_features]
    y_train_val = train_val_pool[target]
    golden_features = recursive_feature_elimination_tscv(
        X_train_val, y_train_val, candidate_features, must_keep_features, n_splits=3
    )

    logger.info(f"✨ 最终确定的输入特征组合 ({len(golden_features)}个): {golden_features}")

    X_pool, y_pool = train_val_pool[golden_features], train_val_pool[target]
    X_test, y_test = test_df[golden_features], test_df[target]

    # 3. TimeSeriesSplit 参数优化
    best_params = optimize_hyperparameters(X_pool, y_pool, n_trials=50)

    # ⚠️ 自动保存最优参数到本地 JSON 文件
    with open(PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"✅ 最优参数已安全保存至: {PARAMS_JSON}")

    # 4. 终极盲测阶段
    logger.info("🏁 执行盲测集评估与特征物理重要性分析...")
    
    initial_lr = best_params.pop('learning_rate')
    final_model = lgb.LGBMRegressor(**best_params, n_estimators=1000, objective='regression') # n_estimators 修改最终盲测模型的训练轮次
    
    # ⚠️ 动态学习率衰减
    lr_scheduler = lgb.reset_parameter(
        learning_rate=lambda iter: initial_lr * (0.999 ** iter) 
    )

    final_model.fit(
        X_pool, y_pool, 
        eval_set=[(X_test, y_test)],
        eval_metric=['rmse', 'mae'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),  # 早停的耐心值 越大越耐心
            lr_scheduler
        ]
    )
    
    y_pred = final_model.predict(X_test)
    
    # --- 指标计算 (四维体系) ---
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_bias = np.mean(y_pred - y_test)

    # --- 基于 gain 的特征重要性并归一化 ---
    # 1. 从底层 booster 显式获取原生的 gain（信息增益）数值
    raw_gain = final_model.booster_.feature_importance(importance_type='gain')
    
    # 2. 归一化：每个特征的 gain 除以全局 gain 的总和
    normalized_gain = raw_gain / raw_gain.sum()

    importance_df = pd.DataFrame({
        'Feature': final_model.feature_name_,
        'Importance (Normalized Gain)': normalized_gain
    }).sort_values(by='Importance (Normalized Gain)', ascending=False)
    
    # 输出报告
    logger.info("="*30 + " XCO2 ENHANCEMENT 最终评估报告 " + "="*30)
    logger.info(f"Test R²   : {test_r2:.4f}")
    logger.info(f"Test RMSE : {test_rmse:.4f} ppm")
    logger.info(f"Test MAE  : {test_mae:.4f} ppm")
    logger.info(f"Test BIAS : {test_bias:.4f} ppm")
    logger.info("-" * 25 + " 气象与卫星因子贡献度排名前 10 " + "-" * 25)
    
    # 打印时保留 4 位小数，与 Random Forest 保持一致
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['Feature']:>20} : {row['Importance (Normalized Gain)']:.4f}")
        
    logger.info("="*85)