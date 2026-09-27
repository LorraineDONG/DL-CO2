# -*- coding: utf-8 -*-
"""
xco2en_qa_pipeline.py

面向 XGBoost ensemble ΔXCO2 重建产品的完整 QA 管线。

功能
----
1. build:
   - 复现训练时的 80/20 随机留出划分；
   - 仅用训练池建立 k-NN 特征空间参考分布；
   - 用留出测试样本固定 OOD 三分位阈值与 raw ensemble spread 的 P90 阈值；
   - 保存 QA 阈值和 k-NN 模型。

2. validate:
   - 对留出测试样本生成 ensemble mean、raw_std、sigma_cal、OOD 和 QA；
   - 按 QA=0/1/2 报告 N、比例、RMSE、MAE、Bias、95% PICP、CE、平均 sigma_cal；
   - 输出不同地区和季节的 QA 比例；
   - 检查 RMSE_QA0 < RMSE_QA1 < RMSE_QA2；
   - 检查 QA=0 的 CE 是否最低或距最低值不超过指定容差。

3. predict:
   - 对逐日 post_data_*.pkl 生成预测；
   - 输出类似卫星产品的 qa_flag、qa_bits、ood_group、sigma_cal 和 95% CI；
   - 保留输入无效行，并以 qa_flag=255 标记，而不是静默删除。

QA 定义
-------
qa_flag = 0: Recommended / 建议使用
qa_flag = 1: Use with caution / 谨慎使用
qa_flag = 2: Not recommended / 不建议用于定量应用
qa_flag = 255: Invalid / 无效输入或无法预测

注意
----
- OOD 阈值、raw_std 阈值必须从固定参考样本中计算，不能逐日或逐批重新标准化。
- 当前 validate 属于内部留出样本上的 QA 诊断。若将来需要完全独立的 QA 泛化验证，
  可在不改动 QA 逻辑的情况下，换用第二独立测试集或折外预测。

使用
----
1. 建立固定QA参考阈值
python generate_xgb_xco2en_prediction_with_flag_pkl.py build --version A01
2. 对测试样本应用QA并生成验证报告
python generate_xgb_xco2en_prediction_with_flag_pkl.py validate --version A01
3. 生成逐日预测产品
python generate_xgb_xco2en_prediction_with_flag_pkl.py predict --version A01
4. 一步完成所有流程
python xco2en_qa_pipeline.py all --version A01
或者右上小三角
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# =============================================================================
# 1. 全局配置：请先核对路径和区域定义
# =============================================================================
ASSETS_DIR = Path("/home/whdong/dl")
INPUT_PKL_DIR = ASSETS_DIR / "ML-prediction_input_data"
OUTPUT_PRED_DIR = ASSETS_DIR / "ML-prediction-output_result"
TRAINING_DATA = (
    ASSETS_DIR
    / "data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl"
)
QA_ARTIFACT_DIR = ASSETS_DIR / "qa_artifacts"
QA_VALIDATION_DIR = ASSETS_DIR / "qa_validation"

TARGET = "xco2_enhanced"
ACTIVE_VERSION = "A01"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_NEIGHBORS = 100
MAX_OOD_REFERENCE_SAMPLES = 50_000
QA_SCHEMA_VERSION = 2

# QA=0 的 CE 若不高于所有等级最小 CE + 1.0 个百分点，则视作“最低或接近最低”
CE_NEAR_BEST_TOLERANCE_PP = 1.0

MODEL_REGISTRY = {
    "A01": {"model_suf": "A01", "scaler_suf": "A1", "feat_suf": "A01", "k_factor": 11.99},
    "A02": {"model_suf": "A02", "scaler_suf": "A2", "feat_suf": "A02", "k_factor": 12.28},
    "A03": {"model_suf": "A03", "scaler_suf": "A3", "feat_suf": "A03", "k_factor": 10.42},
    "A04": {"model_suf": "A04", "scaler_suf": "A4", "feat_suf": "A04", "k_factor": 4.15},
    "A05": {"model_suf": "A05", "scaler_suf": "A5", "feat_suf": "A05", "k_factor": 16.84},
    "A06": {"model_suf": "A06", "scaler_suf": "A6", "feat_suf": "A06", "k_factor": 4.23},
    "A07": {"model_suf": "A07", "scaler_suf": "A7", "feat_suf": "A07", "k_factor": 4.16},
    "A08": {"model_suf": "A08", "scaler_suf": "A8", "feat_suf": "A08", "k_factor": 4.28},
    "A09": {"model_suf": "A09", "scaler_suf": "A9", "feat_suf": "A09", "k_factor": 4.51},
}

# 仅在数据没有现成 region 列时使用。
# 请改成与你论文 Table 2 完全一致的城市群边界或空间掩膜。
REGION_BOXES: Sequence[Tuple[str, float, float, float, float]] = (
    # name, lon_min, lon_max, lat_min, lat_max
    ("North_China_Plain", 112.0, 120.5, 34.0, 41.5),
    ("Yangtze_River_Delta", 117.0, 123.0, 28.0, 34.5),
    ("Pearl_River_Delta", 111.5, 115.5, 21.0, 24.5),
)

# bitwise QA 诊断
QA_BIT_DEFINITIONS: Mapping[int, str] = {
    0: "weak_signal_1_to_2ppm",
    1: "very_weak_signal_below_1ppm",
    2: "extreme_enhancement_above_5ppm",
    3: "low_ood_calibration_warning",
    4: "high_ood_extrapolation_warning",
    5: "high_ensemble_disagreement",
    6: "summer_condition",
    7: "outside_training_target_range_0_to_10ppm",
    8: "invalid_input_or_prediction",
}

QA_LABELS = {
    0: "recommended",
    1: "use_with_caution",
    2: "not_recommended",
    255: "invalid",
}


# =============================================================================
# 2. 路径、特征和模型
# =============================================================================
def artifact_paths(version: str) -> Dict[str, Path]:
    QA_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "thresholds": QA_ARTIFACT_DIR / f"quality_thresholds_{version}.json",
        "nn_model": QA_ARTIFACT_DIR / f"quality_knn_{version}.pkl",
    }


def universal_feature_factory(df: pd.DataFrame) -> pd.DataFrame:
    """与训练代码保持一致地构建派生特征。"""
    out = df.copy()

    if "no2_trop" in out.columns:
        no2 = pd.to_numeric(out["no2_trop"], errors="coerce")
        out["no2_trop_log"] = np.where(no2 > 0, np.log(no2), np.nan)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        month = out["date"].dt.month
        doy = out["date"].dt.dayofyear
        out["month"] = month
        out["doy"] = doy
        out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    if {"ndvi", "era5_t2m"}.issubset(out.columns):
        out["ndvi_t2m_cross"] = out["ndvi"] * out["era5_t2m"]

    if {"era5_ssrd", "era5_t2m"}.issubset(out.columns):
        out["ssrd_t2m_cross"] = out["era5_ssrd"] * out["era5_t2m"]

    if {"ntl", "meic_nox"}.issubset(out.columns):
        out["ntl_nox_cross"] = out["ntl"] * out["meic_nox"]

    if {"era5_u100", "era5_v100"}.issubset(out.columns):
        out["era5_wind_speed"] = np.sqrt(
            out["era5_u100"] ** 2 + out["era5_v100"] ** 2
        )

    return out


def load_version_pipeline(version: str, verbose: bool = True):
    if version not in MODEL_REGISTRY:
        raise ValueError(f"版本 {version!r} 不在 MODEL_REGISTRY 中。")

    cfg = MODEL_REGISTRY[version]
    feature_path = (
        ASSETS_DIR / f"best_params/selected_features-{cfg['feat_suf']}.json"
    )
    scaler_path = (
        ASSETS_DIR / f"models/XCO2en_SHP-xgb_scaler-{cfg['scaler_suf']}.pkl"
    )
    model_pattern = str(
        ASSETS_DIR
        / f"models/XCO2en_SHP-xgb_model-{cfg['model_suf']}_seed*.pkl"
    )
    model_paths = sorted(glob.glob(model_pattern))

    if not feature_path.exists():
        raise FileNotFoundError(f"缺少特征文件: {feature_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"缺少 scaler: {scaler_path}")
    if len(model_paths) < 2:
        raise FileNotFoundError(f"未找到足够的 ensemble 模型: {model_pattern}")

    with feature_path.open("r", encoding="utf-8") as f:
        features = json.load(f)

    scaler = joblib.load(scaler_path)
    models = [joblib.load(path) for path in model_paths]
    k_factor = float(cfg["k_factor"])

    if verbose:
        print(f"版本: {version}")
        print(f"选中特征数: {len(features)}")
        print(f"ensemble 模型数: {len(models)}")
        print(f"PICP scaling factor k: {k_factor:.4f}")
        if len(models) != 10:
            print("警告：当前模型数不是 10，请确认是否符合论文设置。")

    return features, scaler, models, k_factor


def load_modeling_dataset(features: Sequence[str]) -> pd.DataFrame:
    """
    按训练脚本的顺序重建建模数据。
    训练脚本先 df.dropna()，再构建派生特征，因此这里保持相同顺序。
    """
    if not TRAINING_DATA.exists():
        raise FileNotFoundError(f"缺少训练数据: {TRAINING_DATA}")

    df = pd.read_pickle(TRAINING_DATA)
    dc = df.dropna().copy()
    dc = universal_feature_factory(dc)

    missing = [name for name in features if name not in dc.columns]
    if missing:
        raise KeyError(f"训练数据缺少模型特征: {missing}")
    if TARGET not in dc.columns:
        raise KeyError(f"训练数据缺少目标变量: {TARGET}")

    finite_mask = np.isfinite(dc[list(features)].to_numpy(dtype=float)).all(axis=1)
    finite_mask &= np.isfinite(pd.to_numeric(dc[TARGET], errors="coerce"))
    if not finite_mask.all():
        print(f"警告：派生特征或目标存在非有限值，移除 {(~finite_mask).sum()} 行。")
        dc = dc.loc[finite_mask].copy()

    return dc.reset_index(drop=True)


def reproduce_holdout_split(dc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """复现 train_xgb_xco2en_GPU-A01.py 中 random_state=42 的 80/20 划分。"""
    all_idx = np.arange(len(dc))
    pool_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return dc.iloc[pool_idx].copy(), dc.iloc[test_idx].copy()


# =============================================================================
# 3. Ensemble、OOD 和固定参考阈值
# =============================================================================
def ensemble_predict(models: Sequence, X_scaled: np.ndarray):
    preds = np.vstack(
        [np.asarray(model.predict(X_scaled), dtype=np.float64) for model in models]
    )
    return preds.mean(axis=0), preds.std(axis=0), preds


def fit_ood_reference(
    X_pool_scaled: np.ndarray,
) -> Tuple[NearestNeighbors, float, float, int]:
    """
    只使用训练池建立特征空间参考。

    对参考样本查询自身时，第一个邻居是自身（距离 0），因此使用 K+1 个邻居并剔除第一个。
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n_ref = min(MAX_OOD_REFERENCE_SAMPLES, len(X_pool_scaled))
    if n_ref < len(X_pool_scaled):
        ref_idx = rng.choice(len(X_pool_scaled), n_ref, replace=False)
        X_ref = X_pool_scaled[ref_idx]
    else:
        X_ref = X_pool_scaled

    if len(X_ref) < 3:
        raise ValueError("OOD 参考样本太少。")

    k_query = min(N_NEIGHBORS, len(X_ref) - 1)
    nn_model = NearestNeighbors(
        n_neighbors=k_query + 1,
        metric="euclidean",
        n_jobs=-1,
    )
    nn_model.fit(X_ref)

    ref_distances, _ = nn_model.kneighbors(X_ref, n_neighbors=k_query + 1)
    ref_mean_distance = ref_distances[:, 1:].mean(axis=1)

    distance_mu = float(ref_mean_distance.mean())
    distance_sd = float(ref_mean_distance.std(ddof=0))
    if not np.isfinite(distance_sd) or distance_sd <= 0:
        raise ValueError("训练池 k-NN 距离标准差无效。")

    return nn_model, distance_mu, distance_sd, k_query


def compute_ood_score(
    nn_model: NearestNeighbors,
    X_scaled: np.ndarray,
    distance_mu: float,
    distance_sd: float,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用固定训练参考分布计算 OOD z-score。
    不允许按每日批次重新计算均值和标准差。
    """
    distances, _ = nn_model.kneighbors(
        X_scaled,
        n_neighbors=k_neighbors,
    )
    mean_distance = distances.mean(axis=1)
    ood_score = (mean_distance - distance_mu) / distance_sd
    return ood_score, mean_distance


def build_reference_artifacts(
    version: str = ACTIVE_VERSION,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    建立固定 QA 参考阈值。

    - OOD z-score 的中心与尺度：训练池自距离分布；
    - Low/Mid/High OOD 分界：留出测试样本 OOD 的 P33/P67；
    - 高模型分歧阈值：留出测试样本 raw_std 的 P90。
    """
    paths = artifact_paths(version)

    if not force and paths["thresholds"].exists() and paths["nn_model"].exists():
        with paths["thresholds"].open("r", encoding="utf-8") as f:
            existing = json.load(f)
        required = {
            "schema_version",
            "ood_reference_distance_mean",
            "ood_reference_distance_std",
            "ood_p33",
            "ood_p67",
            "raw_std_q90",
            "k_neighbors",
        }
        if (
            existing.get("schema_version") == QA_SCHEMA_VERSION
            and required.issubset(existing)
        ):
            if verbose:
                print(f"复用现有 QA 阈值: {paths['thresholds']}")
            return existing
        if verbose:
            print("现有阈值文件版本不兼容，重新生成。")

    features, scaler, models, k_factor = load_version_pipeline(version, verbose=verbose)
    dc = load_modeling_dataset(features)
    pool_df, test_df = reproduce_holdout_split(dc)

    X_pool = scaler.transform(pool_df[list(features)].to_numpy(dtype=float))
    X_test = scaler.transform(test_df[list(features)].to_numpy(dtype=float))

    nn_model, dist_mu, dist_sd, k_neighbors = fit_ood_reference(X_pool)
    test_ood, _ = compute_ood_score(
        nn_model,
        X_test,
        dist_mu,
        dist_sd,
        k_neighbors,
    )

    test_mean, test_raw_std, _ = ensemble_predict(models, X_test)

    thresholds = {
        "schema_version": QA_SCHEMA_VERSION,
        "version": version,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "k_factor": k_factor,
        "k_neighbors": int(k_neighbors),
        "n_modeling_samples": int(len(dc)),
        "n_training_pool": int(len(pool_df)),
        "n_holdout_test": int(len(test_df)),
        "n_ood_reference": int(nn_model.n_samples_fit_),
        "ood_reference_distance_mean": dist_mu,
        "ood_reference_distance_std": dist_sd,
        "ood_p33": float(np.percentile(test_ood, 33)),
        "ood_p67": float(np.percentile(test_ood, 67)),
        "raw_std_q90": float(np.percentile(test_raw_std, 90)),
        "signal_very_weak_upper_ppm": 1.0,
        "signal_core_lower_ppm": 2.0,
        "signal_core_upper_ppm": 5.0,
        "training_target_lower_ppm": 0.0,
        "training_target_upper_ppm": 10.0,
        "ce_near_best_tolerance_percentage_points": CE_NEAR_BEST_TOLERANCE_PP,
        "qa_bit_definitions": {str(k): v for k, v in QA_BIT_DEFINITIONS.items()},
        "notes": (
            "OOD z-score is standardized against training-pool self-distance. "
            "OOD P33/P67 and raw_std P90 are fixed from the held-out reference test set."
        ),
    }

    paths["thresholds"].parent.mkdir(parents=True, exist_ok=True)
    with paths["thresholds"].open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    joblib.dump(nn_model, paths["nn_model"])

    if verbose:
        print(f"保存 QA 阈值: {paths['thresholds']}")
        print(f"保存 k-NN 模型: {paths['nn_model']}")
        print(
            "阈值: "
            f"OOD_P33={thresholds['ood_p33']:.4f}, "
            f"OOD_P67={thresholds['ood_p67']:.4f}, "
            f"raw_std_Q90={thresholds['raw_std_q90']:.6f}"
        )
    return thresholds


def load_quality_artifacts(
    version: str,
    force_rebuild: bool = False,
    verbose: bool = True,
):
    thresholds = build_reference_artifacts(
        version,
        force=force_rebuild,
        verbose=verbose,
    )
    paths = artifact_paths(version)
    nn_model = joblib.load(paths["nn_model"])
    return thresholds, nn_model


# =============================================================================
# 4. QA 规则
# =============================================================================
def assign_qa(
    pred: np.ndarray,
    raw_std: np.ndarray,
    ood_score: np.ndarray,
    month: np.ndarray,
    thresholds: Mapping[str, float],
    valid_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    向量化生成 qa_flag、qa_bits、ood_group 和 qa_label。

    QA=0（建议使用）：
      2 <= pred <= 5 ppm；Mid OOD；raw_std < Q90。

    QA=1（谨慎使用）：
      未达到 QA=2 的其他有效预测，包括：
      1–2 ppm、>5 ppm、Low OOD、High OOD 等单一风险条件。

    QA=2（不建议用于定量应用）：
      pred < 1 ppm；
      raw_std >= Q90；
      High OOD 与弱信号共同出现；
      High OOD 与 >5 ppm 共同出现；
      预测超出训练目标 0–10 ppm。

    QA=255：
      输入无效或无法生成有限预测。
    """
    pred = np.asarray(pred, dtype=float)
    raw_std = np.asarray(raw_std, dtype=float)
    ood_score = np.asarray(ood_score, dtype=float)
    month = np.asarray(month, dtype=float)

    n = len(pred)
    if not (len(raw_std) == len(ood_score) == len(month) == n):
        raise ValueError("assign_qa 输入数组长度不一致。")

    finite = (
        np.isfinite(pred)
        & np.isfinite(raw_std)
        & np.isfinite(ood_score)
    )
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)

    weak = (pred >= 1.0) & (pred < 2.0)
    very_weak = pred < 1.0
    core = (pred >= 2.0) & (pred <= 5.0)
    extreme = pred > 5.0

    low_ood = ood_score < float(thresholds["ood_p33"])
    high_ood = ood_score >= float(thresholds["ood_p67"])
    mid_ood = ~(low_ood | high_ood)

    high_disagreement = raw_std >= float(thresholds["raw_std_q90"])
    summer = np.isin(month, [6, 7, 8])

    target_low = float(thresholds.get("training_target_lower_ppm", 0.0))
    target_high = float(thresholds.get("training_target_upper_ppm", 10.0))
    outside_target_range = (pred < target_low) | (pred > target_high)

    qa_bits = np.zeros(n, dtype=np.uint16)
    qa_bits[weak] |= np.uint16(1 << 0)
    qa_bits[very_weak] |= np.uint16(1 << 1)
    qa_bits[extreme] |= np.uint16(1 << 2)
    qa_bits[low_ood] |= np.uint16(1 << 3)
    qa_bits[high_ood] |= np.uint16(1 << 4)
    qa_bits[high_disagreement] |= np.uint16(1 << 5)
    qa_bits[summer] |= np.uint16(1 << 6)
    qa_bits[outside_target_range] |= np.uint16(1 << 7)
    qa_bits[~finite] |= np.uint16(1 << 8)

    qa_flag = np.full(n, 255, dtype=np.uint8)

    severe = (
    very_weak
    | outside_target_range
    | (high_ood & weak)
    | (high_ood & extreme)
    | (high_ood & high_disagreement)
    | (extreme & high_disagreement)
    | (weak & high_disagreement)
)

    recommended = core & mid_ood & (~high_disagreement) & (~outside_target_range)

    qa_flag[finite & severe] = 2
    qa_flag[finite & (~severe)] = 1
    qa_flag[finite & recommended] = 0

    ood_group = np.full(n, 255, dtype=np.uint8)
    ood_group[finite & low_ood] = 0
    ood_group[finite & mid_ood] = 1
    ood_group[finite & high_ood] = 2

    qa_label = np.array(
        [QA_LABELS[int(flag)] for flag in qa_flag],
        dtype=object,
    )
    return qa_flag, qa_bits, ood_group, qa_label


def decode_qa_bits(bit_value: int) -> List[str]:
    return [
        name
        for bit, name in QA_BIT_DEFINITIONS.items()
        if int(bit_value) & (1 << bit)
    ]


# =============================================================================
# 5. 验证统计
# =============================================================================
def calibration_curve(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma_cal: np.ndarray,
    levels: Sequence[float] | None = None,
) -> pd.DataFrame:
    if levels is None:
        levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    sigma_cal = np.asarray(sigma_cal, dtype=float)

    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
        & np.isfinite(sigma_cal)
        & (sigma_cal >= 0)
    )
    observed = observed[valid]
    predicted = predicted[valid]
    sigma_cal = sigma_cal[valid]

    if len(observed) == 0:
        return pd.DataFrame(
            columns=["nominal_level", "observed_coverage", "absolute_error"]
        )

    abs_error = np.abs(predicted - observed)
    records = []
    for level in levels:
        z_value = norm.ppf((1.0 + float(level)) / 2.0)
        coverage = float(np.mean(abs_error <= z_value * sigma_cal))
        records.append(
            {
                "nominal_level": float(level),
                "observed_coverage": coverage,
                "absolute_error": abs(coverage - float(level)),
            }
        )
    return pd.DataFrame(records)


def summarize_prediction_subset(
    df: pd.DataFrame,
    observed_col: str = "observed_xco2_enhanced",
    predicted_col: str = "pred_xco2_enhanced",
    sigma_col: str = "sigma_cal",
) -> Dict[str, float]:
    valid = (
        np.isfinite(df[observed_col])
        & np.isfinite(df[predicted_col])
        & np.isfinite(df[sigma_col])
    )
    sub = df.loc[valid]
    n = len(sub)

    empty = {
        "n": 0,
        "rmse_ppm": np.nan,
        "mae_ppm": np.nan,
        "bias_ppm": np.nan,
        "picp95_percent": np.nan,
        "ce_percent": np.nan,
        "mean_sigma_cal_ppm": np.nan,
        "mean_raw_std_ppm": np.nan,
        "mean_ood_score": np.nan,
    }
    if n == 0:
        return empty

    error = (
        sub[predicted_col].to_numpy(dtype=float)
        - sub[observed_col].to_numpy(dtype=float)
    )
    sigma = sub[sigma_col].to_numpy(dtype=float)
    curve = calibration_curve(
        sub[observed_col].to_numpy(dtype=float),
        sub[predicted_col].to_numpy(dtype=float),
        sigma,
    )

    return {
        "n": int(n),
        "rmse_ppm": float(np.sqrt(np.mean(error ** 2))),
        "mae_ppm": float(np.mean(np.abs(error))),
        # Bias 定义为 predicted - observed
        "bias_ppm": float(np.mean(error)),
        "picp95_percent": float(
            100.0 * np.mean(np.abs(error) <= 1.96 * sigma)
        ),
        "ce_percent": float(100.0 * curve["absolute_error"].mean()),
        "mean_sigma_cal_ppm": float(np.mean(sigma)),
        "mean_raw_std_ppm": float(np.mean(sub["raw_std"])),
        "mean_ood_score": float(np.mean(sub["ood_score"])),
    }


def assign_season(date_series: pd.Series) -> pd.Series:
    month = pd.to_datetime(date_series, errors="coerce").dt.month
    season = np.select(
        [
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
            month.isin([12, 1, 2]),
        ],
        ["MAM", "JJA", "SON", "DJF"],
        default="Unknown",
    )
    return pd.Series(season, index=date_series.index)


def assign_region(df: pd.DataFrame) -> pd.Series:
    """
    优先使用已有 region 列；否则使用 REGION_BOXES。
    正式论文请务必使这里的边界与区域验证定义完全一致。
    """
    if "region" in df.columns and df["region"].notna().any():
        return df["region"].fillna("Other").astype(str)

    if not {"grid_lon", "grid_lat"}.issubset(df.columns):
        return pd.Series("Unknown", index=df.index)

    lon = pd.to_numeric(df["grid_lon"], errors="coerce")
    lat = pd.to_numeric(df["grid_lat"], errors="coerce")
    region = pd.Series("Other", index=df.index, dtype=object)

    # 反向赋值，保证前面列出的区域优先级更高
    for name, lon_min, lon_max, lat_min, lat_max in reversed(REGION_BOXES):
        mask = (
            lon.between(lon_min, lon_max, inclusive="both")
            & lat.between(lat_min, lat_max, inclusive="both")
        )
        region.loc[mask] = name
    return region


def qa_summary_table(validation_df: pd.DataFrame) -> pd.DataFrame:
    valid_total = int(validation_df["qa_flag"].isin([0, 1, 2]).sum())
    records = []

    for qa_flag in [0, 1, 2]:
        sub = validation_df[validation_df["qa_flag"] == qa_flag]
        metrics = summarize_prediction_subset(sub)
        metrics.update(
            {
                "qa_flag": qa_flag,
                "qa_label": QA_LABELS[qa_flag],
                "proportion_percent": (
                    100.0 * len(sub) / valid_total if valid_total else np.nan
                ),
            }
        )
        records.append(metrics)

    cols = [
        "qa_flag",
        "qa_label",
        "n",
        "proportion_percent",
        "rmse_ppm",
        "mae_ppm",
        "bias_ppm",
        "picp95_percent",
        "ce_percent",
        "mean_sigma_cal_ppm",
        "mean_raw_std_ppm",
        "mean_ood_score",
    ]
    return pd.DataFrame(records)[cols]


def qa_composition_table(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """
    同时报告：
    - pct_within_category: 一个地区/季节内部 QA0/1/2 的比例；
    - pct_within_qa: 一个 QA 等级内部各地区/季节的组成比例。
    """
    valid = df[df["qa_flag"].isin([0, 1, 2])].copy()
    counts = (
        valid.groupby([dimension, "qa_flag"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )

    category_total = valid.groupby(dimension, dropna=False).size()
    qa_total = valid.groupby("qa_flag").size()

    counts["pct_within_category"] = counts.apply(
        lambda row: 100.0
        * row["n"]
        / category_total.loc[row[dimension]],
        axis=1,
    )
    counts["pct_within_qa"] = counts.apply(
        lambda row: 100.0 * row["n"] / qa_total.loc[row["qa_flag"]],
        axis=1,
    )
    counts["qa_label"] = counts["qa_flag"].map(QA_LABELS)
    return counts[
        [
            dimension,
            "qa_flag",
            "qa_label",
            "n",
            "pct_within_category",
            "pct_within_qa",
        ]
    ].sort_values([dimension, "qa_flag"])


def diagnostic_subset_table(df: pd.DataFrame) -> pd.DataFrame:
    """用于定位 QA 排序失败的具体风险条件。"""
    records = []
    for bit, name in QA_BIT_DEFINITIONS.items():
        mask = (df["qa_bits"].astype(np.uint16) & (1 << bit)) != 0
        sub = df.loc[mask]
        metrics = summarize_prediction_subset(sub)
        metrics.update(
            {
                "diagnostic_bit": bit,
                "diagnostic_name": name,
                "proportion_percent": (
                    100.0 * len(sub) / len(df) if len(df) else np.nan
                ),
            }
        )
        records.append(metrics)
    return pd.DataFrame(records)


def reliability_curve_by_qa(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for qa_flag in [0, 1, 2]:
        sub = df[df["qa_flag"] == qa_flag]
        curve = calibration_curve(
            sub["observed_xco2_enhanced"].to_numpy(dtype=float),
            sub["pred_xco2_enhanced"].to_numpy(dtype=float),
            sub["sigma_cal"].to_numpy(dtype=float),
        )
        curve.insert(0, "qa_flag", qa_flag)
        curve.insert(1, "qa_label", QA_LABELS[qa_flag])
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def evaluate_quality_order(
    summary: pd.DataFrame,
    tolerance_pp: float = CE_NEAR_BEST_TOLERANCE_PP,
) -> Dict[str, object]:
    indexed = summary.set_index("qa_flag")

    required = [0, 1, 2]
    enough_data = all(
        flag in indexed.index
        and int(indexed.loc[flag, "n"]) > 0
        and np.isfinite(indexed.loc[flag, "rmse_ppm"])
        and np.isfinite(indexed.loc[flag, "ce_percent"])
        for flag in required
    )

    result: Dict[str, object] = {
        "all_qa_levels_present": bool(enough_data),
        "rmse_order_expected": "RMSE_QA0 < RMSE_QA1 < RMSE_QA2",
        "ce_criterion": (
            f"CE_QA0 <= minimum CE across QA levels + {tolerance_pp:.2f} percentage points"
        ),
    }

    if not enough_data:
        result.update(
            {
                "rmse_monotonic_increase": False,
                "qa0_ce_is_lowest": False,
                "qa0_ce_is_near_lowest": False,
                "interpretation": "至少一个 QA 等级没有有效样本，无法完整检验。",
            }
        )
        return result

    rmse0 = float(indexed.loc[0, "rmse_ppm"])
    rmse1 = float(indexed.loc[1, "rmse_ppm"])
    rmse2 = float(indexed.loc[2, "rmse_ppm"])
    ce0 = float(indexed.loc[0, "ce_percent"])
    ce_values = indexed.loc[required, "ce_percent"].astype(float)
    minimum_ce = float(ce_values.min())

    rmse_ok = rmse0 < rmse1 < rmse2
    ce_lowest = bool(np.isclose(ce0, minimum_ce, atol=1e-12))
    ce_near = ce0 <= minimum_ce + tolerance_pp

    result.update(
        {
            "rmse_qa0": rmse0,
            "rmse_qa1": rmse1,
            "rmse_qa2": rmse2,
            "rmse_monotonic_increase": bool(rmse_ok),
            "ce_qa0_percent": ce0,
            "minimum_ce_percent": minimum_ce,
            "qa0_ce_is_lowest": ce_lowest,
            "qa0_ce_is_near_lowest": bool(ce_near),
            "overall_internal_qa_check_passed": bool(rmse_ok and ce_near),
        }
    )

    messages = []
    if rmse_ok:
        messages.append("RMSE 随 QA 等级恶化而严格增加。")
    else:
        messages.append(
            "RMSE 未满足严格递增；应查看 diagnostic_subset_summary.csv，"
            "判断某个严重条件是否被过度降级，或某个高误差条件仍留在 QA=0/1。"
        )

    if ce_lowest:
        messages.append("QA=0 的 CE 为三个等级中最低。")
    elif ce_near:
        messages.append(
            f"QA=0 的 CE 距最低值不超过 {tolerance_pp:.2f} 个百分点，视为接近最低。"
        )
    else:
        messages.append(
            "QA=0 的 CE 明显高于其他等级；应优先检查 Mid OOD、2–5 ppm "
            "和 raw_std P90 的组合是否足以定义概率可靠性。"
        )
    result["interpretation"] = " ".join(messages)
    return result


# =============================================================================
# 6. 对留出测试样本应用 QA 并输出报告
# =============================================================================
def build_validation_dataframe(
    version: str = ACTIVE_VERSION,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    features, scaler, models, k_factor = load_version_pipeline(version)
    thresholds, nn_model = load_quality_artifacts(version, force_rebuild)
    dc = load_modeling_dataset(features)
    _, test_df = reproduce_holdout_split(dc)

    X_test_raw = test_df[list(features)].to_numpy(dtype=float)
    X_test_scaled = scaler.transform(X_test_raw)

    pred_mean, raw_std, _ = ensemble_predict(models, X_test_scaled)
    ood_score, ood_distance = compute_ood_score(
        nn_model,
        X_test_scaled,
        float(thresholds["ood_reference_distance_mean"]),
        float(thresholds["ood_reference_distance_std"]),
        int(thresholds["k_neighbors"]),
    )
    sigma_cal = raw_std * k_factor

    if "date" in test_df.columns:
        month = pd.to_datetime(test_df["date"], errors="coerce").dt.month.to_numpy()
    else:
        month = np.full(len(test_df), np.nan)

    qa_flag, qa_bits, ood_group, qa_label = assign_qa(
        pred_mean,
        raw_std,
        ood_score,
        month,
        thresholds,
    )

    out = test_df.copy()
    out["observed_xco2_enhanced"] = pd.to_numeric(out[TARGET], errors="coerce")
    out["pred_xco2_enhanced"] = pred_mean
    out["raw_std"] = raw_std
    out["sigma_cal"] = sigma_cal
    out["ood_mean_distance"] = ood_distance
    out["ood_score"] = ood_score
    out["ood_group"] = ood_group
    out["qa_flag"] = qa_flag
    out["qa_label"] = qa_label
    out["qa_bits"] = qa_bits
    out["pred_lower_95ci"] = pred_mean - 1.96 * sigma_cal
    out["pred_upper_95ci"] = pred_mean + 1.96 * sigma_cal
    out["season"] = assign_season(out["date"]) if "date" in out else "Unknown"
    out["region"] = assign_region(out)
    return out


def validate_qa(
    version: str = ACTIVE_VERSION,
    force_rebuild: bool = False,
) -> None:
    QA_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    validation_df = build_validation_dataframe(version, force_rebuild)

    summary = qa_summary_table(validation_df)
    region_composition = qa_composition_table(validation_df, "region")
    season_composition = qa_composition_table(validation_df, "season")
    diagnostic_summary = diagnostic_subset_table(validation_df)
    reliability = reliability_curve_by_qa(validation_df)
    checks = evaluate_quality_order(summary)

    validation_df.to_pickle(
        QA_VALIDATION_DIR / f"qa_validation_predictions_{version}.pkl"
    )
    summary.to_csv(
        QA_VALIDATION_DIR / f"qa_level_summary_{version}.csv",
        index=False,
        float_format="%.6f",
    )
    region_composition.to_csv(
        QA_VALIDATION_DIR / f"qa_region_composition_{version}.csv",
        index=False,
        float_format="%.6f",
    )
    season_composition.to_csv(
        QA_VALIDATION_DIR / f"qa_season_composition_{version}.csv",
        index=False,
        float_format="%.6f",
    )
    diagnostic_summary.to_csv(
        QA_VALIDATION_DIR / f"diagnostic_subset_summary_{version}.csv",
        index=False,
        float_format="%.6f",
    )
    reliability.to_csv(
        QA_VALIDATION_DIR / f"qa_reliability_curves_{version}.csv",
        index=False,
        float_format="%.6f",
    )
    with (QA_VALIDATION_DIR / f"qa_validation_checks_{version}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("QA 分级统计（Bias = predicted - observed）")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 100)
    print("内部 QA 顺序检查")
    print("=" * 100)
    print(json.dumps(checks, indent=2, ensure_ascii=False))

    print(f"\n所有验证文件已写入: {QA_VALIDATION_DIR}")


# =============================================================================
# 7. 逐日预测产品生成
# =============================================================================
def initialize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    float_cols = [
        "pred_xco2_enhanced",
        "raw_std",
        "sigma_cal",
        "ood_mean_distance",
        "ood_score",
        "pred_lower_95ci",
        "pred_upper_95ci",
    ]
    for col in float_cols:
        out[col] = np.full(n, np.nan, dtype=np.float64)

    out["ood_group"] = np.full(n, 255, dtype=np.uint8)
    out["qa_flag"] = np.full(n, 255, dtype=np.uint8)
    out["qa_label"] = np.full(n, QA_LABELS[255], dtype=object)
    out["qa_bits"] = np.full(n, np.uint16(1 << 8), dtype=np.uint16)
    return out


def predict_one_dataframe(
    df_input: pd.DataFrame,
    features: Sequence[str],
    scaler,
    models: Sequence,
    k_factor: float,
    thresholds: Mapping[str, float],
    nn_model: NearestNeighbors,
) -> pd.DataFrame:
    # 重置索引，确保后续按位置写回预测结果时不会因输入索引不连续而错位。
    df = universal_feature_factory(df_input).reset_index(drop=True)
    out = initialize_output_columns(df)

    missing = [name for name in features if name not in df.columns]
    if missing:
        raise KeyError(f"输入数据缺少模型特征: {missing}")

    X_all = df[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid_feature_mask = np.isfinite(X_all).all(axis=1)

    if not valid_feature_mask.any():
        return out

    valid_idx = np.where(valid_feature_mask)[0]
    X_valid_scaled = scaler.transform(X_all[valid_feature_mask])

    pred_mean, raw_std, _ = ensemble_predict(models, X_valid_scaled)
    ood_score, ood_distance = compute_ood_score(
        nn_model,
        X_valid_scaled,
        float(thresholds["ood_reference_distance_mean"]),
        float(thresholds["ood_reference_distance_std"]),
        int(thresholds["k_neighbors"]),
    )
    sigma_cal = raw_std * k_factor

    if "date" in df.columns:
        month = (
            pd.to_datetime(df.loc[valid_feature_mask, "date"], errors="coerce")
            .dt.month
            .to_numpy()
        )
    else:
        month = np.full(len(valid_idx), np.nan)

    qa_flag, qa_bits, ood_group, qa_label = assign_qa(
        pred_mean,
        raw_std,
        ood_score,
        month,
        thresholds,
    )

    out.loc[valid_idx, "pred_xco2_enhanced"] = pred_mean
    out.loc[valid_idx, "raw_std"] = raw_std
    out.loc[valid_idx, "sigma_cal"] = sigma_cal
    out.loc[valid_idx, "ood_mean_distance"] = ood_distance
    out.loc[valid_idx, "ood_score"] = ood_score
    out.loc[valid_idx, "ood_group"] = ood_group
    out.loc[valid_idx, "qa_flag"] = qa_flag
    out.loc[valid_idx, "qa_label"] = qa_label
    out.loc[valid_idx, "qa_bits"] = qa_bits
    out.loc[valid_idx, "pred_lower_95ci"] = pred_mean - 1.96 * sigma_cal
    out.loc[valid_idx, "pred_upper_95ci"] = pred_mean + 1.96 * sigma_cal

    return out


def run_batch_inference(
    version: str = ACTIVE_VERSION,
    force_rebuild: bool = False,
) -> None:
    """逐文件生成预测产品；屏幕仅显示当前文件及成功/失败状态。"""
    features, scaler, models, k_factor = load_version_pipeline(
        version,
        verbose=False,
    )
    thresholds, nn_model = load_quality_artifacts(
        version,
        force_rebuild,
        verbose=False,
    )

    version_output_dir = OUTPUT_PRED_DIR / version
    version_output_dir.mkdir(parents=True, exist_ok=True)

    # input_files = sorted(INPUT_PKL_DIR.glob("post_data_*.pkl"))[:1]
    input_files = sorted(INPUT_PKL_DIR.glob("post_data_*.pkl"))
    if not input_files:
        raise FileNotFoundError(f"输入目录没有 post_data_*.pkl: {INPUT_PKL_DIR}")

    for file_index, input_path in enumerate(input_files, start=1):
        print(
            f"[{file_index}/{len(input_files)}] {input_path.name} ... ",
            end="",
            flush=True,
        )
        try:
            df_input = pd.read_pickle(input_path)
            if len(df_input) == 0:
                print("跳过（空文件）")
                continue

            result = predict_one_dataframe(
                df_input,
                features,
                scaler,
                models,
                k_factor,
                thresholds,
                nn_model,
            )

            output_name = input_path.name.replace(
                "post_data_", "pred_0.1deg_"
            )
            output_path = version_output_dir / output_name
            result.to_pickle(output_path)
            print("成功")

        except Exception as exc:
            print(f"失败：{type(exc).__name__}: {exc}")


# =============================================================================
# 8. 命令行
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate QA-labelled XCO2 enhancement predictions."
    )
    parser.add_argument(
        "command",
        choices=["build", "validate", "predict", "all"],
        nargs="?",
        default="all",
        help="build=建立阈值；validate=测试集验证；predict=逐日预测；all=全部执行",
    )
    parser.add_argument(
        "--version",
        default=ACTIVE_VERSION,
        choices=sorted(MODEL_REGISTRY),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="忽略已有 QA 阈值并重新构建。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command in {"build", "all"}:
        build_reference_artifacts(
            args.version,
            force=args.force_rebuild,
        )

    if args.command in {"validate", "all"}:
        validate_qa(
            args.version,
            force_rebuild=args.force_rebuild,
        )

    if args.command in {"predict", "all"}:
        run_batch_inference(
            args.version,
            force_rebuild=args.force_rebuild,
        )


if __name__ == "__main__":
    main()