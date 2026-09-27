# -*- coding: utf-8 -*-
"""
generate_predict_pkl_to_nc_QA01.py

将预测 PKL 转换为 NetCDF，并在任何统计或写出之前实施 QA 筛选。

QA 定义：
    qa_flag = 0   Recommended
    qa_flag = 1   Use with caution
    qa_flag = 2   Not recommended
    qa_flag = 255 Invalid/fill value

本文结果固定保留：QA = (0, 1)

有效记录还必须同时满足：
    1. qa_flag 属于 KEEP_QA；
    2. pred_xco2_enhanced 为有限值；
    3. 不确定性列为有限值；
    4. 经纬度为有限值；
    5. 日期/时间能够解析。

脚本提供两种输出：
    A. convert_pkl_to_daily_nc()
       每个逐日 PKL 输出一个经过 QA01 筛选的逐日 NC。

    B. generate_qa_screened_stats_nc()
       将指定日期范围内的 QA01 记录聚合到规则网格，输出均值、方差、
       有效记录数、有效天数和有效天数比例等统计量。

默认 main() 运行 B，即生成 2023 年 QA01 聚合统计 NC。
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    nc = None
    NETCDF4_AVAILABLE = False
import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================
# 1. 全局配置
# =============================================================================
INPUT_PKL_DIR = Path("/home/whdong/dl/ML-prediction-output_result/A01")
OUTPUT_NC_DIR = INPUT_PKL_DIR / "nc_files_QA01"
OUTPUT_NC_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATTERN = "pred_0.1deg_0.1deg_*.pkl"

BG_MEDIAN_PRIMARY = Path("/home/whdong/dl/data/daily_bg_median.pkl")
BG_MEDIAN_FALLBACK = Path("/home/whdong/dl/data/daily_bg_median_oco3.pkl")

EPOCH_START = pd.Timestamp("1993-01-01 00:00:00")

# QA 配置
QA_COL = "qa_flag"
KEEP_QA: Tuple[int, ...] = (0, 1)
QA_SELECTION_TAG = "QA" + "".join(str(flag) for flag in KEEP_QA)
QA_DEFINITION = (
    "0=Recommended; 1=Use with caution; "
    "2=Not recommended; 255=Invalid/fill value"
)

PRED_COL = "pred_xco2_enhanced"
LAT_CANDIDATES = ("grid_lat", "lat", "latitude")
LON_CANDIDATES = ("grid_lon", "lon", "longitude")

# 前面 QA 制图流程使用 sigma_cal；旧转换代码使用 pred_uncertainty_1sigma。
# 这里优先使用 sigma_cal，不存在时再兼容旧列名。
UNCERTAINTY_CANDIDATES = (
    "sigma_cal",
    "pred_uncertainty_1sigma",
)

# 年度统计 NC 范围
DEFAULT_START_DATE = "20230101"
DEFAULT_END_DATE = "20231231"
DEFAULT_TARGET_RES = 0.1
DEFAULT_EXTENT = (70.0, 140.0, 15.0, 55.0)  # lon_min, lon_max, lat_min, lat_max

# =============================================================================
# 2. 通用工具
# =============================================================================
def first_existing_column(df: pd.DataFrame, candidates: Iterable[str], role: str) -> str:
    """返回第一个存在的候选列名。"""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"缺少 {role} 列；候选列为: {list(candidates)}")


def load_single_bg_dict(filepath: Path, name: str) -> Dict[str, float]:
    """读取背景场 PKL，转换为 {YYYYMMDD: Background_Median}。"""
    if not filepath.exists():
        print(f"警告: 找不到{name}背景场文件: {filepath}")
        return {}

    bg_df = pd.read_pickle(filepath)
    required = {"Date", "Background_Median"}
    missing = sorted(required.difference(bg_df.columns))
    if missing:
        raise KeyError(f"{filepath.name} 缺少列: {missing}")

    bg_df = bg_df.copy()
    bg_df["Date"] = pd.to_datetime(bg_df["Date"], errors="coerce")
    bg_df["Background_Median"] = pd.to_numeric(
        bg_df["Background_Median"], errors="coerce"
    )
    bg_df = bg_df[
        bg_df["Date"].notna()
        & np.isfinite(bg_df["Background_Median"])
    ].copy()
    bg_df["Date_str"] = bg_df["Date"].dt.strftime("%Y%m%d")

    bg_dict = dict(zip(bg_df["Date_str"], bg_df["Background_Median"]))
    print(f"成功加载{name}背景场，共 {len(bg_dict)} 天。")
    return bg_dict


def parse_time_series(df: pd.DataFrame, file_path: Path) -> pd.Series:
    """解析逐记录真实时间；优先 no2_time，其次 date。"""
    if "no2_time" in df.columns:
        seconds = pd.to_numeric(df["no2_time"], errors="coerce")
        return EPOCH_START + pd.to_timedelta(seconds, unit="s")

    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce")

    # 最后尝试从文件名提取 8 位日期
    digits = "".join(ch if ch.isdigit() else " " for ch in file_path.stem).split()
    candidates = [token for token in digits if len(token) >= 8]
    if candidates:
        date_str = candidates[-1][-8:]
        parsed = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        return pd.Series(parsed, index=df.index, dtype="datetime64[ns]")

    raise KeyError(f"{file_path.name} 中没有 no2_time/date，文件名也无法解析日期。")


def qa_filter_dataframe(
    df: pd.DataFrame,
    file_path: Path,
    keep_qa: Sequence[int] = KEEP_QA,
) -> Tuple[pd.DataFrame, dict]:
    """
    按前述 QA01 流程筛选单个 DataFrame。

    返回：
        filtered_df：包含规范化辅助列的有效记录；
        stats：筛选前后数量及 QA 数量。
    """
    if df.empty:
        return df.copy(), {
            "file": file_path.name,
            "rows_before": 0,
            "rows_after": 0,
            "qa0": 0,
            "qa1": 0,
            "qa2": 0,
            "qa255": 0,
        }

    required = {QA_COL, PRED_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{file_path.name} 缺少 QA 筛选所需列: {missing}")

    lat_col = first_existing_column(df, LAT_CANDIDATES, "纬度")
    lon_col = first_existing_column(df, LON_CANDIDATES, "经度")
    unc_col = first_existing_column(df, UNCERTAINTY_CANDIDATES, "不确定性")

    work = df.copy()
    work["_qa_numeric"] = pd.to_numeric(work[QA_COL], errors="coerce")
    work["_pred_numeric"] = pd.to_numeric(work[PRED_COL], errors="coerce")
    work["_unc_numeric"] = pd.to_numeric(work[unc_col], errors="coerce")
    work["_lat_numeric"] = pd.to_numeric(work[lat_col], errors="coerce")
    work["_lon_numeric"] = pd.to_numeric(work[lon_col], errors="coerce")
    work["_real_time"] = parse_time_series(work, file_path)

    qa_counts = work["_qa_numeric"].value_counts(dropna=False)

    retain = (
        work["_qa_numeric"].isin(list(keep_qa))
        & np.isfinite(work["_pred_numeric"])
        & np.isfinite(work["_unc_numeric"])
        & np.isfinite(work["_lat_numeric"])
        & np.isfinite(work["_lon_numeric"])
        & work["_real_time"].notna()
    )

    filtered = work.loc[retain].copy()
    filtered[QA_COL] = filtered["_qa_numeric"].astype(np.int16)
    filtered[PRED_COL] = filtered["_pred_numeric"].astype(np.float64)
    filtered["uncertainty_selected"] = filtered["_unc_numeric"].astype(np.float64)
    filtered["latitude_selected"] = filtered["_lat_numeric"].astype(np.float64)
    filtered["longitude_selected"] = filtered["_lon_numeric"].astype(np.float64)
    filtered["date"] = pd.to_datetime(filtered["_real_time"]).dt.normalize()
    filtered["date_str"] = filtered["date"].dt.strftime("%Y%m%d")

    stats = {
        "file": file_path.name,
        "uncertainty_column": unc_col,
        "rows_before": int(len(work)),
        "rows_after": int(len(filtered)),
        "removed_rows": int(len(work) - len(filtered)),
        "retained_percent": (
            100.0 * len(filtered) / len(work) if len(work) else np.nan
        ),
        "qa0": int(qa_counts.get(0, 0)),
        "qa1": int(qa_counts.get(1, 0)),
        "qa2": int(qa_counts.get(2, 0)),
        "qa255": int(qa_counts.get(255, 0)),
    }
    return filtered, stats


def resample_grid_centers(
    lat: pd.Series,
    lon: pd.Series,
    target_res: float,
) -> Tuple[pd.Series, pd.Series]:
    """与制图代码一致：floor(coord/res)*res + res/2。"""
    decimals = max(2, int(np.ceil(-np.log10(target_res))) + 2)
    lat_r = (
        np.floor(pd.to_numeric(lat, errors="coerce") / target_res)
        * target_res + target_res / 2.0
    ).round(decimals)
    lon_r = (
        np.floor(pd.to_numeric(lon, errors="coerce") / target_res)
        * target_res + target_res / 2.0
    ).round(decimals)
    return lat_r, lon_r


# =============================================================================
# 3. 逐日 QA01 PKL -> 逐日 NetCDF
# =============================================================================
def convert_pkl_to_daily_nc(
    input_dir: Path = INPUT_PKL_DIR,
    output_dir: Path = OUTPUT_NC_DIR / "daily",
    keep_qa: Sequence[int] = KEEP_QA,
) -> None:
    """将每个逐日 PKL 筛选为 QA01 后写成一个逐日 NC。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not NETCDF4_AVAILABLE:
        raise ImportError("逐日 NC 输出需要 netCDF4；请先安装 netCDF4。")

    print("正在初始化背景场字典...")
    bg_primary = load_single_bg_dict(BG_MEDIAN_PRIMARY, "主选")
    bg_fallback = load_single_bg_dict(BG_MEDIAN_FALLBACK, "备用(OCO-3)")

    pkl_files = sorted(input_dir.glob(FILE_PATTERN))
    if not pkl_files:
        raise FileNotFoundError(f"{input_dir} 中未找到 {FILE_PATTERN}")

    log_rows = []
    print(f"共找到 {len(pkl_files)} 个预测文件。")

    for index, pkl_file in enumerate(pkl_files, start=1):
        print(f"[{index:03d}/{len(pkl_files):03d}] {pkl_file.name}", end=" ... ")
        try:
            raw = pd.read_pickle(pkl_file)
            filtered, stats = qa_filter_dataframe(raw, pkl_file, keep_qa)
            log_rows.append(stats)

            if filtered.empty:
                print("无 QA01 有效记录，跳过")
                continue

            unique_dates = filtered["date_str"].dropna().unique()
            if len(unique_dates) != 1:
                raise ValueError(
                    f"筛选后应只有一个日期，实际为: {unique_dates.tolist()}"
                )
            date_str = str(unique_dates[0])

            if date_str in bg_primary:
                daily_bg_val = float(bg_primary[date_str])
                bg_source = "Primary File"
            elif date_str in bg_fallback:
                daily_bg_val = float(bg_fallback[date_str])
                bg_source = "Fallback File (OCO-3)"
            else:
                print(f"日期 {date_str} 无背景场，跳过")
                continue

            n_samples = len(filtered)
            real_time = pd.to_datetime(filtered["_real_time"])
            time_data = real_time.dt.strftime("%H%M%S").astype(np.float64).to_numpy()

            nc_path = output_dir / f"ml-xco2en-{date_str}-{QA_SELECTION_TAG}.nc"
            with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
                ds.createDimension("nSamples", n_samples)

                def create_f8(name: str):
                    return ds.createVariable(
                        name,
                        "f8",
                        ("nSamples",),
                        zlib=True,
                        complevel=4,
                        fill_value=np.nan,
                    )

                var_lat = create_f8("latitude")
                var_lon = create_f8("longitude")
                var_xco2 = create_f8("xco2_enhancement")
                var_bg = create_f8("background_median")
                var_time = create_f8("time")
                var_unc = create_f8("uncertainty")
                var_qa = ds.createVariable(
                    "qa_flag", "i2", ("nSamples",), zlib=True, complevel=4,
                    fill_value=np.int16(-32768),
                )

                var_lat[:] = filtered["latitude_selected"].to_numpy(np.float64)
                var_lon[:] = filtered["longitude_selected"].to_numpy(np.float64)
                var_xco2[:] = filtered[PRED_COL].to_numpy(np.float64)
                var_bg[:] = np.full(n_samples, daily_bg_val, dtype=np.float64)
                var_time[:] = time_data
                var_unc[:] = filtered["uncertainty_selected"].to_numpy(np.float64)
                var_qa[:] = filtered[QA_COL].to_numpy(np.int16)

                var_lat.units = "degrees_north"
                var_lon.units = "degrees_east"
                var_xco2.units = "ppm"
                var_bg.units = "ppm"
                var_time.units = "HHMMSS.0"
                var_unc.units = "ppm"
                var_qa.long_name = "quality assurance flag"
                var_qa.flag_values = np.array([0, 1, 2, 255], dtype=np.int16)
                var_qa.flag_meanings = (
                    "recommended use_with_caution not_recommended invalid_fill"
                )

                ds.Description = (
                    "Machine-learning reconstructed XCO2 enhancement, "
                    "screened using QA=(0,1) before NetCDF export"
                )
                ds.QA_Definition = QA_DEFINITION
                ds.QA_Kept = ",".join(map(str, keep_qa))
                ds.QA_Filter = (
                    "qa_flag in KEEP_QA and finite prediction, uncertainty, "
                    "latitude, longitude, and valid time"
                )
                ds.Records_Before_QA = int(stats["rows_before"])
                ds.Records_After_QA = int(stats["rows_after"])
                ds.Uncertainty_Source_Column = str(stats["uncertainty_column"])
                ds.Background_Source = f"Matched from {bg_source}"

            print(f"成功: {nc_path.name}，保留 {n_samples} 条")
        except Exception as exc:
            print(f"失败: {type(exc).__name__}: {exc}")

    if log_rows:
        log_path = output_dir / f"Daily_NC_QA_Filter_Log_{QA_SELECTION_TAG}.csv"
        pd.DataFrame(log_rows).to_csv(
            log_path, index=False, encoding="utf-8-sig", float_format="%.4f"
        )
        print(f"逐文件 QA 日志: {log_path}")


# =============================================================================
# 4. 指定时间范围 QA01 聚合统计 -> 单个 NetCDF
# =============================================================================
def generate_qa_screened_stats_nc(
    input_dir: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    target_res: float = DEFAULT_TARGET_RES,
    extent: Sequence[float] = DEFAULT_EXTENT,
    keep_qa: Sequence[int] = KEEP_QA,
) -> Path:
    """
    将指定日期范围内的 PKL 在 QA01 筛选后聚合为规则网格统计 NC。

    注意：QA 筛选发生在拼接、重采样和分组统计之前。
    """
    all_files = sorted(input_dir.glob(FILE_PATTERN))
    if not all_files:
        raise FileNotFoundError(f"{input_dir} 中未找到 {FILE_PATTERN}")

    start_ts = pd.to_datetime(start_date, format="%Y%m%d")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d")
    if start_ts > end_ts:
        raise ValueError("start_date 不能晚于 end_date")

    retained_frames = []
    qa_log_rows = []
    processed_dates = set()

    print(f"共找到 {len(all_files)} 个 PKL；开始 QA={tuple(keep_qa)} 筛选。")

    for index, file_path in enumerate(all_files, start=1):
        print(f"[{index:03d}/{len(all_files):03d}] {file_path.name}", end=" ... ")
        try:
            raw = pd.read_pickle(file_path)
            if raw.empty:
                print("空文件")
                continue

            # 先解析日期，用于日期范围限制；随后执行 QA01 流程。
            raw = raw.copy()
            raw["_pre_time"] = parse_time_series(raw, file_path)
            in_range = raw["_pre_time"].between(start_ts, end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
            raw = raw.loc[in_range].copy()

            if raw.empty:
                print("不在日期范围")
                continue

            # qa_filter_dataframe 会重新规范化时间，并执行完整有效性判断。
            filtered, stats = qa_filter_dataframe(raw, file_path, keep_qa)
            qa_log_rows.append(stats)

            # 分母使用日期范围内成功读取且可解析的唯一日期数；
            # 即使某天没有 QA01 有效记录，该日仍属于处理时段。
            valid_dates_before_qa = pd.to_datetime(raw["_pre_time"], errors="coerce").dt.normalize().dropna().unique()
            processed_dates.update(pd.Timestamp(d) for d in valid_dates_before_qa)

            if filtered.empty:
                print("无 QA01 有效记录")
                continue

            retained_frames.append(filtered[[
                "date",
                "date_str",
                QA_COL,
                PRED_COL,
                "uncertainty_selected",
                "latitude_selected",
                "longitude_selected",
            ]].copy())
            print(f"保留 {len(filtered)}/{len(raw)}")
        except Exception as exc:
            print(f"失败: {type(exc).__name__}: {exc}")

    if not retained_frames:
        raise RuntimeError("日期范围内没有通过 QA=(0,1) 的有效记录。")

    df_all = pd.concat(retained_frames, ignore_index=True)
    print(f"QA01 筛选后总记录数: {len(df_all)}")

    # 重采样
    lat_r, lon_r = resample_grid_centers(
        df_all["latitude_selected"],
        df_all["longitude_selected"],
        target_res,
    )
    df_all["lat_resampled"] = lat_r
    df_all["lon_resampled"] = lon_r

    # 区域筛选
    lon_min, lon_max, lat_min, lat_max = map(float, extent)
    region_mask = (
        df_all["lon_resampled"].between(lon_min, lon_max)
        & df_all["lat_resampled"].between(lat_min, lat_max)
    )
    df_all = df_all.loc[region_mask].copy()
    if df_all.empty:
        raise RuntimeError("QA01 与区域筛选后没有有效记录。")
    print(f"区域筛选后记录数: {len(df_all)}")

    # 网格统计
    stats = (
        df_all.groupby(
            ["lat_resampled", "lon_resampled"],
            as_index=False,
            sort=True,
            observed=True,
        )
        .agg(
            xco2_mean=(PRED_COL, "mean"),
            xco2_var=(PRED_COL, lambda x: np.var(x, ddof=0)),
            uncertainty_mean=("uncertainty_selected", "mean"),
            uncertainty_var=("uncertainty_selected", lambda x: np.var(x, ddof=0)),
            valid_prediction_record_count=(PRED_COL, "count"),
            valid_day_count=("date", "nunique"),
            qa0_record_count=(QA_COL, lambda x: int((x == 0).sum())),
            qa1_record_count=(QA_COL, lambda x: int((x == 1).sum())),
        )
    )

    period_day_count = len(processed_dates)
    stats["period_day_count"] = int(period_day_count)
    stats["valid_day_percent"] = (
        100.0 * stats["valid_day_count"] / period_day_count
        if period_day_count > 0
        else np.nan
    )

    lats = np.sort(stats["lat_resampled"].unique())
    lons = np.sort(stats["lon_resampled"].unique())

    def matrix(value_col: str) -> np.ndarray:
        return (
            stats.pivot(
                index="lat_resampled",
                columns="lon_resampled",
                values=value_col,
            )
            .reindex(index=lats, columns=lons)
            .to_numpy()
        )

    ds = xr.Dataset(
        data_vars={
            "xco2_enhanced_mean": (("lat", "lon"), matrix("xco2_mean")),
            "xco2_enhanced_var": (("lat", "lon"), matrix("xco2_var")),
            "uncertainty_mean": (("lat", "lon"), matrix("uncertainty_mean")),
            "uncertainty_var": (("lat", "lon"), matrix("uncertainty_var")),
            "valid_prediction_record_count": (
                ("lat", "lon"), matrix("valid_prediction_record_count")
            ),
            "valid_day_count": (("lat", "lon"), matrix("valid_day_count")),
            "valid_day_percent": (("lat", "lon"), matrix("valid_day_percent")),
            "qa0_record_count": (("lat", "lon"), matrix("qa0_record_count")),
            "qa1_record_count": (("lat", "lon"), matrix("qa1_record_count")),
        },
        coords={"lon": lons, "lat": lats},
        attrs={
            "description": (
                "QA-screened gridded statistics of reconstructed XCO2 "
                "enhancement and calibrated predictive uncertainty"
            ),
            "temporal_range": f"{start_date}-{end_date}",
            "spatial_extent": (
                f"lon={lon_min}:{lon_max}, lat={lat_min}:{lat_max}"
            ),
            "spatial_resolution": f"{target_res} degree x {target_res} degree",
            "grid_method": "floor(coord/res)*res + res/2",
            "qa_definition": QA_DEFINITION,
            "qa_kept": ",".join(map(str, keep_qa)),
            "qa_filter": (
                "qa_flag in (0,1) and finite pred_xco2_enhanced, "
                "uncertainty, latitude, longitude, and valid date"
            ),
            "records_after_qa_and_region_filter": int(len(df_all)),
            "processed_day_count": int(period_day_count),
            "missing_value": "NaN",
        },
    )

    # 坐标属性
    ds["lon"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})
    ds["lat"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})

    # 数据变量属性
    ds["xco2_enhanced_mean"].attrs.update({"units": "ppm", "long_name": "mean reconstructed XCO2 enhancement after QA01 screening"})
    ds["xco2_enhanced_var"].attrs.update({"units": "ppm2", "long_name": "population variance of reconstructed XCO2 enhancement after QA01 screening"})
    ds["uncertainty_mean"].attrs.update({"units": "ppm", "long_name": "mean calibrated predictive uncertainty after QA01 screening"})
    ds["uncertainty_var"].attrs.update({"units": "ppm2", "long_name": "population variance of calibrated predictive uncertainty after QA01 screening"})
    ds["valid_prediction_record_count"].attrs["units"] = "1"
    ds["valid_day_count"].attrs["units"] = "day"
    ds["valid_day_percent"].attrs["units"] = "%"
    ds["qa0_record_count"].attrs["units"] = "1"
    ds["qa1_record_count"].attrs["units"] = "1"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"reconstructed_xco2en_stats_{start_date}_{end_date}_{QA_SELECTION_TAG}.nc"
    )

    if NETCDF4_AVAILABLE:
        encoding = {
            name: {"zlib": True, "complevel": 4}
            for name in ds.data_vars
        }
        ds.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
    else:
        # scipy 引擎不支持 zlib 压缩，但仍可生成标准 NetCDF3 文件。
        ds.to_netcdf(output_path, engine="scipy")

    # 同时输出 QA 筛选日志，便于核查。
    if qa_log_rows:
        log_path = output_dir / (
            f"PKL_QA_Filter_Log_{start_date}_{end_date}_{QA_SELECTION_TAG}.csv"
        )
        pd.DataFrame(qa_log_rows).to_csv(
            log_path,
            index=False,
            encoding="utf-8-sig",
            float_format="%.4f",
        )
        print(f"QA 筛选日志: {log_path}")

    print(f"\n成功生成 QA01 NetCDF:\n{output_path}")
    print(f"纬度格点数: {len(lats)}；经度格点数: {len(lons)}")
    print(f"处理日期数: {period_day_count}")
    print(
        "XCO2 均值范围: "
        f"[{np.nanmin(ds['xco2_enhanced_mean'].values):.3f}, "
        f"{np.nanmax(ds['xco2_enhanced_mean'].values):.3f}] ppm"
    )
    return output_path


# =============================================================================
# 5. 主程序
# =============================================================================
if __name__ == "__main__":
    # 方案 A：逐日 PKL -> QA01 逐日 NC
    convert_pkl_to_daily_nc()

    # 方案 B：指定日期范围 -> 一个 QA01 聚合统计 NC（默认运行）
    # generate_qa_screened_stats_nc(
    #     input_dir=INPUT_PKL_DIR,
    #     output_dir=OUTPUT_NC_DIR,
    #     start_date=DEFAULT_START_DATE,
    #     end_date=DEFAULT_END_DATE,
    #     target_res=DEFAULT_TARGET_RES,
    #     extent=DEFAULT_EXTENT,
    #     keep_qa=KEEP_QA,
    # )