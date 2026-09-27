import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置与版本注册表
# ==========================================
ASSETS_DIR = '/home/whdong/dl'
INPUT_PKL_DIR = os.path.join(ASSETS_DIR, 'ML-prediction_input_data')
OUTPUT_PRED_DIR = os.path.join(ASSETS_DIR, 'ML-prediction-output_result')

os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)

# 🎯 模型资产注册表
# model_suf: 模型文件名的后缀
# scaler_suf: 你的代码中由于历史原因，Scaler 后缀是单数字 (如 A1, A8)
# k_factor: 通过可靠性图 (brentq) 算出来的温度缩放系数。未计算的暂设 1.0。
MODEL_REGISTRY = {
    'A01': {'model_suf': 'A01', 'scaler_suf': 'A1', 'feat_suf': 'A01', 'k_factor': 11.99},
    'A02': {'model_suf': 'A02', 'scaler_suf': 'A2', 'feat_suf': 'A02', 'k_factor': 12.28},
    'A03': {'model_suf': 'A03', 'scaler_suf': 'A3', 'feat_suf': 'A03', 'k_factor': 10.42},
    'A04': {'model_suf': 'A04', 'scaler_suf': 'A4', 'feat_suf': 'A04', 'k_factor': 4.15},
    'A05': {'model_suf': 'A05', 'scaler_suf': 'A5', 'feat_suf': 'A05', 'k_factor': 16.84},
    'A06': {'model_suf': 'A06', 'scaler_suf': 'A6', 'feat_suf': 'A06', 'k_factor': 4.23},
    'A07': {'model_suf': 'A07', 'scaler_suf': 'A7', 'feat_suf': 'A07', 'k_factor': 4.16},
    'A08': {'model_suf': 'A08', 'scaler_suf': 'A8', 'feat_suf': 'A08', 'k_factor': 4.28}, 
    'A09': {'model_suf': 'A09', 'scaler_suf': 'A9', 'feat_suf': 'A09', 'k_factor': 4.51},
}

# 🚀 【唯一需要手动修改的地方】：你想应用哪个版本去生成 1 年的数据？
ACTIVE_VERSION = 'A01' 

# ==========================================
# 2. 大一统特征工厂 (适配 A01 - A09 所有的特征组合)
# ==========================================
def universal_feature_factory(df):
    df_out = df.copy()
    
    # 防止无用警告，且兼容历史数据可能出现的情况
    if 'no2_trop' in df_out.columns:
        # np.log 保持与训练脚本一致
        df_out['no2_trop_log'] = np.log(df_out['no2_trop'])
        
    if 'date' in df_out.columns:
        if not np.issubdtype(df_out['date'].dtype, np.datetime64):
            df_out['date'] = pd.to_datetime(df_out['date'])
        month = df_out['date'].dt.month
        doy = df_out['date'].dt.dayofyear
        df_out['month_sin'] = np.sin(2 * np.pi * month / 12.0)
        df_out['month_cos'] = np.cos(2 * np.pi * month / 12.0)
        df_out['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
        df_out['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
        
    if 'ndvi' in df_out.columns and 'era5_t2m' in df_out.columns:
        df_out['ndvi_t2m_cross'] = df_out['ndvi'] * df_out['era5_t2m']
        
    if 'era5_ssrd' in df_out.columns and 'era5_t2m' in df_out.columns:
        df_out['ssrd_t2m_cross'] = df_out['era5_ssrd'] * df_out['era5_t2m']
        
    if 'ntl' in df_out.columns and 'meic_nox' in df_out.columns:
        df_out['ntl_nox_cross'] = df_out['ntl'] * df_out['meic_nox']
        
    if 'era5_u100' in df_out.columns and 'era5_v100' in df_out.columns:
        df_out['era5_wind_speed'] = np.sqrt(df_out['era5_u100']**2 + df_out['era5_v100']**2)
        
    return df_out

# ==========================================
# 3. 动态流水线加载器
# ==========================================
def load_version_pipeline(version):
    config = MODEL_REGISTRY.get(version)
    if not config:
        raise ValueError(f"❌ 注册表中未找到版本: {version}")
        
    feat_path = os.path.join(ASSETS_DIR, f"best_params/selected_features-{config['feat_suf']}.json")
    scaler_path = os.path.join(ASSETS_DIR, f"models/XCO2en_SHP-xgb_scaler-{config['scaler_suf']}.pkl")
    model_pattern = os.path.join(ASSETS_DIR, f"models/XCO2en_SHP-xgb_model-{config['model_suf']}_seed*.pkl")
    
    model_paths = glob.glob(model_pattern)
    
    if not os.path.exists(feat_path): raise FileNotFoundError(f"找不到特征配置: {feat_path}")
    if not os.path.exists(scaler_path): raise FileNotFoundError(f"找不到 Scaler: {scaler_path}")
    if len(model_paths) != 10: 
        print(f"⚠️ 警告: 模型数量为 {len(model_paths)}，预期为 10 个！")

    print(f"✅ 成功加载架构 [{version}]")
    
    with open(feat_path, 'r', encoding='utf-8') as f:
        features = json.load(f)
        
    scaler = joblib.load(scaler_path)
    # 按 seed 文件名自然排序，确保模型顺序稳定
    model_paths.sort() 
    models = [joblib.load(mp) for mp in model_paths]
    
    return features, scaler, models, config['k_factor']

# ==========================================
# 4. 大规模并行推理引擎
# ==========================================
def run_batch_inference():
    print("=" * 60)
    print(f"🚀 启动应用期大规模推理任务 | 目标模型: {ACTIVE_VERSION}")
    print("=" * 60)
    
    features, scaler, ensemble_models, k_factor = load_version_pipeline(ACTIVE_VERSION)
    print(f"📊 使用温度缩放系数 (k_factor): {k_factor:.3f}")
    
    version_output_dir = os.path.join(OUTPUT_PRED_DIR, ACTIVE_VERSION)
    os.makedirs(version_output_dir, exist_ok=True)
    print(f"📂 结果将保存至子目录: {version_output_dir}")
    
    pkl_files = glob.glob(os.path.join(INPUT_PKL_DIR, "post_data_*.pkl"))
    pkl_files.sort()
    
    if not pkl_files:
        print(f"❌ 输入目录 {INPUT_PKL_DIR} 为空。")
        return

    total_files = len(pkl_files)
    
    for idx, pkl_file in enumerate(pkl_files):
        filename = os.path.basename(pkl_file)
        print(f"   [{idx+1}/{total_files}] 正在推断: {filename} ...", end="", flush=True)
        
        try:
            df_day = pd.read_pickle(pkl_file)
            if len(df_day) == 0: 
                print(" (跳过: 空文件)")
                continue
            
            # 清理潜在的空值，防止 Scaler 报错
            df_day = df_day.dropna().reset_index(drop=True)
                
            # 1. 工厂加工全量特征
            df_day = universal_feature_factory(df_day)
            
            # 2. 依照 JSON 清单精准提取
            missing_cols = [col for col in features if col not in df_day.columns]
            if missing_cols:
                raise KeyError(f"数据缺失必要的特征列: {missing_cols}")
            X_raw = df_day[features].values
            
            # 3. 标准化
            X_scaled = scaler.transform(X_raw)
            
            # 4. 10 个模型矩阵预测
            preds_matrix = np.zeros((len(ensemble_models), len(X_scaled)), dtype=np.float32)
            for i, model in enumerate(ensemble_models):
                preds_matrix[i, :] = model.predict(X_scaled)
            
            # 5. 计算期望与校准方差
            df_day['pred_xco2_enhanced'] = preds_matrix.mean(axis=0)
            raw_std = preds_matrix.std(axis=0)
            df_day['pred_uncertainty_1sigma'] = raw_std * k_factor
            
            df_day['pred_lower_95CI'] = df_day['pred_xco2_enhanced'] - 1.96 * df_day['pred_uncertainty_1sigma']
            df_day['pred_upper_95CI'] = df_day['pred_xco2_enhanced'] + 1.96 * df_day['pred_uncertainty_1sigma']
            
            # 6. 导出结果 (仅保留核心列，避免文件过大)
            # 可根据后续 GEOS-Chem 的需要调整导出的列
            export_cols = ['grid_lon', 'grid_lat', 'date', 
                           'pred_xco2_enhanced', 'pred_uncertainty_1sigma', 
                           'pred_lower_95CI', 'pred_upper_95CI']
            
            # 如果原始数据里有某些列丢失，做一层容错
            export_cols = [c for c in export_cols if c in df_day.columns]
            df_export = df_day[export_cols]
            
            output_name = filename.replace("post_data_", "pred_0.1deg_")
            output_path = os.path.join(version_output_dir, output_name)
            df_export.to_pickle(output_path)
            
            print(f" ✅ 完成! (生成 {len(df_export)} 个网格点)")
            
        except Exception as e:
            print(f" ❌ 失败! 错误: {e}")

    print("=" * 60)
    print(f"🎉 批处理任务结束。所有预测结果已保存在: {OUTPUT_PRED_DIR}")

# ==========================================
if __name__ == "__main__":
    run_batch_inference()