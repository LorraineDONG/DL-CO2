# -*- coding: utf-8 -*-
"""
fig5_city_case_study_QA012_monthly_weekly_weekday.py

面向 case study 的四城市时间序列生产线，输出月尺度、连续周尺度和周内周期
（Monday–Sunday）结果。

质量控制原则与 fig3and5_flag012_mask.py 一致：

1. 新 QA 体系：
       qa_flag = 0   Recommended
       qa_flag = 1   Use with caution
       qa_flag = 2   Not recommended
       qa_flag = 255 Invalid/fill value

2. 默认采用 QA=0+1。仅保留同时满足以下条件的预测：
       - qa_flag 属于 FIXED_KEEP_QA；
       - pred_xco2_enhanced、sigma_cal 和经纬度均为有限值；
       - 城市固定网格位于中国省级行政区面内；
       - 城市固定网格不位于启用的 MANUAL_MASK_BOXES 中。

3. 使用固定城市空间窗口提取北京、上海、武汉和广州的逐日序列。默认使用
   城市中心 30 km 半径内的 0.1° 网格，并以 cos(latitude) 近似面积加权；
   也可切换为最近单格点模式。

4. 输出 CSV：
       - 城市固定网格定义；
       - 城市逐日统计；
       - 城市月尺度统计；
       - 城市连续周尺度统计；
       - 城市周内 Monday–Sunday 统计。

5. 输出三张图，每张图只保留两个子图：
       (a) 对流层 NO2 VCD；
       (b) QA 筛选后的 ΔXCO2。

6. 月尺度与连续周尺度图中的浅色带表示逐日城市值的 25%–75% 分位范围；
   ΔXCO2 竖向误差棒表示该时间段内平均 calibrated predictive uncertainty，
   不是聚合均值的标准误。周内图采用同样的表达方式。

7. RUN_MODE：
       "full"      读取逐日 PKL、重新质量控制与聚合，并输出全部 CSV 和图；
       "plot_only" 直接读取已有 CSV 重新绘图，适合微调配色、字号和布局。
   若 plot_only 模式下缺少周内 CSV，但已有逐日 CSV，脚本会快速从逐日 CSV
   重新生成周内统计，无需读取原始 PKL。
"""


from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shapefile
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

try:
    # Shapely >= 2.0
    from shapely import intersects_xy
except ImportError:
    intersects_xy = None


# =============================================================================
# 1. 路径和质量控制配置
# =============================================================================
TARGET_YEAR = 2023
ACTIVE_VERSION = "A01"

# "full"：完整读取逐日 PKL；"plot_only"：仅从已有 CSV 绘图。
RUN_MODE = "plot_only"

PREDICTION_DIR = Path(
    f"/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}"
)
PREDICTION_PATTERN = "pred_0.1deg_0.1deg*.pkl"

INPUT_FEATURES_DIR = Path("/home/whdong/dl/ML-prediction_input_data")
INPUT_FEATURE_PATTERN = "post_data_*.pkl"

OUTPUT_DIR = PREDICTION_DIR / "figures_case_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORLD_SHP = Path("/home/whdong/shapefile/world/world.shp")
CHINA_PROV_SHP = Path("/home/whdong/shapefile/china/province.shp")

PRED_COL = "pred_xco2_enhanced"
SIGMA_COL = "sigma_cal"
QA_COL = "qa_flag"
NO2_COL = "no2_trop"

# 正文主产品推荐使用 QA=0+1；若要与只保留 QA=0 的产品一致，改为 (0,)。
FIXED_KEEP_QA = (0, 1)
QA_SELECTION_TAG = "QA" + "".join(str(flag) for flag in FIXED_KEEP_QA)

APPLY_CHINA_SHAPE_CLIP = True
APPLY_MANUAL_MASK = True

# 与 fig3and5_flag012_mask.py 保持一致。
MANUAL_MASK_BOXES = [
    {
        "name": "entire_1_strip",
        "enabled": True,
        "lon_min": 90.1,
        "lon_max": 94.7,
        "lat_min": None,
        "lat_max": None,
    },
    {
        "name": "entire_2_strip",
        "enabled": True,
        "lon_min": 95.3,
        "lon_max": 102.0,
        "lat_min": None,
        "lat_max": None,
    },
]

TARGET_RESOLUTION = 0.1


# =============================================================================
# 2. 城市空间窗口配置
# =============================================================================
TARGET_CITIES: Dict[str, Dict[str, object]] = {
    "Beijing": {
        "lat": 39.90,
        "lon": 116.40,
        "color": "#9B59B6",
    },
    "Shanghai": {
        "lat": 31.20,
        "lon": 121.50,
        "color": "#34495E",
    },
    "Wuhan": {
        "lat": 30.60,
        "lon": 114.30,
        "color": "#E67E22",
    },
    "Guangzhou": {
        "lat": 23.10,
        "lon": 113.30,
        "color": "#A2B836",
    },
}

# "radius"：使用城市中心一定半径内的全部网格；
# "nearest"：每个城市仅使用最近的一个网格。
CITY_SELECTION_MODE = "radius"
CITY_RADIUS_KM = 30.0

# 城市窗口内对各网格进行面积近似加权，权重为 cos(latitude)。
USE_COSLAT_WEIGHT = True


# =============================================================================
# 3. 输出和绘图配置
# =============================================================================
DAILY_CSV = OUTPUT_DIR / f"FIG5-City_Daily_{QA_SELECTION_TAG}.csv"
MONTHLY_CSV = OUTPUT_DIR / f"FIG5-City_Monthly_{QA_SELECTION_TAG}.csv"
WEEKLY_CSV = OUTPUT_DIR / f"FIG5-City_Weekly_{QA_SELECTION_TAG}.csv"
WEEKDAY_CSV = OUTPUT_DIR / f"FIG5-City_Weekday_{QA_SELECTION_TAG}.csv"
CITY_GRID_CSV = OUTPUT_DIR / f"FIG5-City_Grid_Definition_{QA_SELECTION_TAG}.csv"

MONTHLY_FIGURE = OUTPUT_DIR / f"FIG5-City_Monthly_TimeSeries_{QA_SELECTION_TAG}.png"
WEEKLY_FIGURE = OUTPUT_DIR / f"FIG5-City_Weekly_TimeSeries_{QA_SELECTION_TAG}.png"
WEEKDAY_FIGURE = OUTPUT_DIR / f"FIG5-City_Weekday_Cycle_{QA_SELECTION_TAG}.png"

NO2_YLABEL = r"Tropospheric NO$_2$ VCD"
PRED_YLABEL = r"Predicted $\Delta$XCO$_2$ [ppm]"

WEEKDAY_LABELS = [
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
]

FONT_BASE = 11
FONT_PANEL = 13
FONT_AXIS = 12
FONT_TICK = 10
FONT_LEGEND = 10

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": FONT_BASE,
    "axes.titlesize": FONT_PANEL,
    "axes.titleweight": "bold",
    "axes.labelsize": FONT_AXIS,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "axes.linewidth": 1.3,
    "lines.linewidth": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# =============================================================================
# 4. 基础工具
# =============================================================================
def resample_grid_centers(
    lat: pd.Series,
    lon: pd.Series,
    target_res: float,
) -> Tuple[pd.Series, pd.Series]:
    """将经纬度映射至固定分辨率网格中心。"""
    decimals = max(2, int(np.ceil(-np.log10(target_res))) + 2)

    lat_r = (
        np.floor(pd.to_numeric(lat, errors="coerce") / target_res)
        * target_res
        + target_res / 2.0
    ).round(decimals)

    lon_r = (
        np.floor(pd.to_numeric(lon, errors="coerce") / target_res)
        * target_res
        + target_res / 2.0
    ).round(decimals)

    return lat_r, lon_r


def infer_single_date(df: pd.DataFrame, file_path: Path) -> pd.Timestamp:
    """优先从 date 字段读取日期；失败时从文件名中的 YYYYMMDD 解析。"""
    if "date" in df.columns:
        dates = (
            pd.to_datetime(df["date"], errors="coerce")
            .dt.normalize()
            .dropna()
            .unique()
        )
        if len(dates) == 1:
            return pd.Timestamp(dates[0])
        if len(dates) > 1:
            raise ValueError(f"{file_path.name} 包含多个自然日。")

    match = re.search(r"(20\d{6})", file_path.name)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d")

    raise ValueError(f"无法从 {file_path.name} 解析日期。")


def haversine_distance_km(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: float,
    lat2: float,
) -> np.ndarray:
    """计算坐标到城市中心的大圆距离，单位 km。"""
    earth_radius_km = 6371.0088

    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    dlon = lon1_rad - lon2_rad
    dlat = lat1_rad - lat2_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """忽略非有限值的加权均值。"""
    value_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weight_array = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)

    valid = (
        np.isfinite(value_array)
        & np.isfinite(weight_array)
        & (weight_array > 0)
    )
    if not valid.any():
        return np.nan

    return float(np.average(value_array[valid], weights=weight_array[valid]))


def validate_prediction_columns(df: pd.DataFrame, file_path: Path) -> None:
    required = {"grid_lat", "grid_lon", "date", PRED_COL, SIGMA_COL, QA_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{file_path.name} 缺少预测字段: {missing}")


def validate_no2_columns(df: pd.DataFrame, file_path: Path) -> None:
    required = {"grid_lat", "grid_lon", NO2_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{file_path.name} 缺少 NO2 字段: {missing}")


# =============================================================================
# 5. 中国范围裁剪和手动空间掩膜
# =============================================================================
def build_china_union_geometry():
    if not APPLY_CHINA_SHAPE_CLIP:
        return None

    if not CHINA_PROV_SHP.exists():
        raise FileNotFoundError(
            f"中国省级行政区 shapefile 不存在: {CHINA_PROV_SHP}"
        )

    reader = shapefile.Reader(
        str(CHINA_PROV_SHP),
        encoding="gbk",
    )
    geometries = [
        shape(item.__geo_interface__)
        for item in reader.shapes()
        if item is not None
    ]
    geometries = [
        geometry
        for geometry in geometries
        if geometry is not None and not geometry.is_empty
    ]
    if not geometries:
        raise ValueError("中国省级行政区 shapefile 中没有可用几何对象。")

    china_geometry = unary_union(geometries)
    if not china_geometry.is_valid:
        repaired = china_geometry.buffer(0)
        if not repaired.is_empty:
            china_geometry = repaired

    return china_geometry


def inside_china_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    china_geometry,
) -> np.ndarray:
    """判断网格中心是否位于中国省级行政区面内，边界点保留。"""
    finite = np.isfinite(lon) & np.isfinite(lat)
    inside = np.zeros(len(lon), dtype=bool)

    if not APPLY_CHINA_SHAPE_CLIP:
        inside[finite] = True
        return inside

    if china_geometry is None or not finite.any():
        return inside

    if intersects_xy is not None:
        inside[finite] = np.asarray(
            intersects_xy(
                china_geometry,
                lon[finite],
                lat[finite],
            ),
            dtype=bool,
        )
    else:
        prepared_geometry = prep(china_geometry)
        inside[finite] = np.fromiter(
            (
                prepared_geometry.covers(Point(x, y))
                for x, y in zip(lon[finite], lat[finite])
            ),
            dtype=bool,
            count=int(finite.sum()),
        )

    return inside


def validate_manual_mask_boxes(
    mask_boxes: Sequence[Mapping[str, object]],
) -> None:
    for index, box in enumerate(mask_boxes, start=1):
        if not bool(box.get("enabled", True)):
            continue

        name = str(box.get("name", f"mask_{index}"))
        lon_min = box.get("lon_min")
        lon_max = box.get("lon_max")
        lat_min = box.get("lat_min")
        lat_max = box.get("lat_max")

        if all(
            value is None
            for value in [lon_min, lon_max, lat_min, lat_max]
        ):
            raise ValueError(f"手动掩膜 {name!r} 没有设置边界。")

        if lon_min is not None and lon_max is not None:
            if float(lon_min) > float(lon_max):
                raise ValueError(f"手动掩膜 {name!r}: lon_min > lon_max")

        if lat_min is not None and lat_max is not None:
            if float(lat_min) > float(lat_max):
                raise ValueError(f"手动掩膜 {name!r}: lat_min > lat_max")


def manual_mask_for_coordinates(
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """返回落入任一启用 MANUAL_MASK_BOXES 的布尔掩膜。"""
    masked = np.zeros(len(lon), dtype=bool)

    if not APPLY_MANUAL_MASK:
        return masked

    enabled_boxes = [
        box
        for box in MANUAL_MASK_BOXES
        if bool(box.get("enabled", True))
    ]
    if not enabled_boxes:
        return masked

    validate_manual_mask_boxes(enabled_boxes)

    finite = np.isfinite(lon) & np.isfinite(lat)

    for box in enabled_boxes:
        one_box = finite.copy()
        lon_min = box.get("lon_min")
        lon_max = box.get("lon_max")
        lat_min = box.get("lat_min")
        lat_max = box.get("lat_max")

        if lon_min is not None:
            one_box &= lon >= float(lon_min)
        if lon_max is not None:
            one_box &= lon <= float(lon_max)
        if lat_min is not None:
            one_box &= lat >= float(lat_min)
        if lat_max is not None:
            one_box &= lat <= float(lat_max)

        masked |= one_box

    return masked


# =============================================================================
# 6. 构建固定城市网格窗口
# =============================================================================
def find_prediction_files() -> List[Path]:
    files = sorted(PREDICTION_DIR.glob(PREDICTION_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"未找到预测文件：{PREDICTION_DIR / PREDICTION_PATTERN}"
        )
    return files


def find_no2_files() -> List[Path]:
    return sorted(INPUT_FEATURES_DIR.glob(INPUT_FEATURE_PATTERN))


def build_city_grid_definition(
    prediction_files: Sequence[Path],
    china_geometry,
) -> pd.DataFrame:
    """
    使用第一个可读取的预测文件建立固定城市空间窗口。

    返回字段：City、lat_resampled、lon_resampled、distance_km、weight。
    """
    domain_grid = None

    for file_path in prediction_files:
        try:
            df = pd.read_pickle(file_path)
            validate_prediction_columns(df, file_path)

            lat_r, lon_r = resample_grid_centers(
                df["grid_lat"],
                df["grid_lon"],
                TARGET_RESOLUTION,
            )
            domain_grid = (
                pd.DataFrame({
                    "lat_resampled": lat_r,
                    "lon_resampled": lon_r,
                })
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if not domain_grid.empty:
                break
        except Exception as exc:
            print(f"跳过城市网格模板文件 {file_path.name}: {exc}")

    if domain_grid is None or domain_grid.empty:
        raise RuntimeError("无法从预测文件建立城市网格模板。")

    lon = domain_grid["lon_resampled"].to_numpy(dtype=float)
    lat = domain_grid["lat_resampled"].to_numpy(dtype=float)

    spatial_keep = inside_china_mask(lon, lat, china_geometry)
    spatial_keep &= ~manual_mask_for_coordinates(lon, lat)
    domain_grid = domain_grid.loc[spatial_keep].copy()

    if domain_grid.empty:
        raise RuntimeError("中国范围裁剪和手动掩膜后没有剩余网格。")

    records: List[dict] = []

    for city, info in TARGET_CITIES.items():
        center_lon = float(info["lon"])
        center_lat = float(info["lat"])

        distances = haversine_distance_km(
            domain_grid["lon_resampled"].to_numpy(dtype=float),
            domain_grid["lat_resampled"].to_numpy(dtype=float),
            center_lon,
            center_lat,
        )

        if CITY_SELECTION_MODE.lower() == "nearest":
            selected_indices = np.array([int(np.argmin(distances))])
        elif CITY_SELECTION_MODE.lower() == "radius":
            selected_indices = np.flatnonzero(distances <= CITY_RADIUS_KM)
            if len(selected_indices) == 0:
                nearest_index = int(np.argmin(distances))
                selected_indices = np.array([nearest_index])
                print(
                    f"警告：{city} 的 {CITY_RADIUS_KM:.1f} km 半径内没有网格，"
                    "已回退至最近网格。"
                )
        else:
            raise ValueError(
                "CITY_SELECTION_MODE 只能是 'radius' 或 'nearest'。"
            )

        selected = domain_grid.iloc[selected_indices].copy()
        selected_distances = distances[selected_indices]

        if USE_COSLAT_WEIGHT:
            weights = np.cos(
                np.radians(selected["lat_resampled"].to_numpy(dtype=float))
            )
        else:
            weights = np.ones(len(selected), dtype=float)

        for (_, row), distance, weight in zip(
            selected.iterrows(),
            selected_distances,
            weights,
        ):
            records.append({
                "City": city,
                "city_center_lat": center_lat,
                "city_center_lon": center_lon,
                "lat_resampled": float(row["lat_resampled"]),
                "lon_resampled": float(row["lon_resampled"]),
                "distance_km": float(distance),
                "weight": float(weight),
            })

        print(
            f"{city}: 选择 {len(selected)} 个固定网格；"
            f"距离范围 {selected_distances.min():.1f}–"
            f"{selected_distances.max():.1f} km"
        )

    city_grid = pd.DataFrame(records)
    city_grid.to_csv(
        CITY_GRID_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    return city_grid


# =============================================================================
# 7. 逐日城市 ΔXCO2 和 calibrated predictive uncertainty
# =============================================================================
def extract_daily_prediction_series(
    prediction_files: Sequence[Path],
    city_grid: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    records: List[dict] = []
    successful_dates: List[pd.Timestamp] = []

    lookup = city_grid[
        ["City", "lat_resampled", "lon_resampled", "weight"]
    ].copy()

    for index, file_path in enumerate(prediction_files, start=1):
        print(
            f"预测 [{index:03d}/{len(prediction_files):03d}] "
            f"{file_path.name}",
            end=" ... ",
            flush=True,
        )

        try:
            df = pd.read_pickle(file_path)
            if df.empty:
                print("empty")
                continue

            validate_prediction_columns(df, file_path)
            day = infer_single_date(df, file_path)
            if int(day.year) != TARGET_YEAR:
                print("outside target year")
                continue

            successful_dates.append(day)

            lat_r, lon_r = resample_grid_centers(
                df["grid_lat"],
                df["grid_lon"],
                TARGET_RESOLUTION,
            )

            work = pd.DataFrame({
                "lat_resampled": lat_r,
                "lon_resampled": lon_r,
                PRED_COL: pd.to_numeric(df[PRED_COL], errors="coerce"),
                SIGMA_COL: pd.to_numeric(df[SIGMA_COL], errors="coerce"),
                QA_COL: pd.to_numeric(df[QA_COL], errors="coerce"),
            })

            selected = work.merge(
                lookup,
                on=["lat_resampled", "lon_resampled"],
                how="inner",
                validate="many_to_many",
            )

            if selected.empty:
                print("no city grids")
                continue

            retain = (
                selected[QA_COL].isin(FIXED_KEEP_QA)
                & np.isfinite(selected[PRED_COL])
                & np.isfinite(selected[SIGMA_COL])
            )
            selected = selected.loc[retain].copy()

            if selected.empty:
                print("no retained city predictions")
                continue

            # 同一日期、城市、网格若有多条记录，先合并为一个网格值。
            grid_daily = (
                selected.groupby(
                    [
                        "City",
                        "lat_resampled",
                        "lon_resampled",
                        "weight",
                    ],
                    observed=True,
                    sort=False,
                )
                .agg(
                    pred_grid_mean=(PRED_COL, "mean"),
                    sigma_grid_mean=(SIGMA_COL, "mean"),
                    qa0_count=(QA_COL, lambda x: int((x == 0).sum())),
                    qa1_count=(QA_COL, lambda x: int((x == 1).sum())),
                )
                .reset_index()
            )

            for city in TARGET_CITIES:
                city_rows = grid_daily[grid_daily["City"] == city]
                if city_rows.empty:
                    continue

                pred_daily = weighted_mean(
                    city_rows["pred_grid_mean"],
                    city_rows["weight"],
                )
                sigma_daily = weighted_mean(
                    city_rows["sigma_grid_mean"],
                    city_rows["weight"],
                )

                qa0_count = int(city_rows["qa0_count"].sum())
                qa1_count = int(city_rows["qa1_count"].sum())
                qa_retained_count = qa0_count + qa1_count

                records.append({
                    "City": city,
                    "date": day,
                    "pred_daily_mean": pred_daily,
                    "sigma_cal_daily_mean": sigma_daily,
                    "valid_grid_count": int(len(city_rows)),
                    "qa0_record_count": qa0_count,
                    "qa1_record_count": qa1_count,
                    "qa0_fraction_within_retained": (
                        100.0 * qa0_count / qa_retained_count
                        if qa_retained_count > 0
                        else np.nan
                    ),
                })

            print("success")

        except Exception as exc:
            print(f"failed: {type(exc).__name__}: {exc}")

    if not successful_dates:
        raise RuntimeError("没有成功读取目标年份的预测日期。")

    daily_prediction = pd.DataFrame(records)
    if daily_prediction.empty:
        raise RuntimeError("QA 筛选后没有任何城市预测记录。")

    daily_prediction["date"] = pd.to_datetime(
        daily_prediction["date"]
    ).dt.normalize()

    successful_dates = sorted(set(pd.Timestamp(x) for x in successful_dates))
    return daily_prediction, successful_dates


# =============================================================================
# 8. 逐日城市 NO2 序列
# =============================================================================
def extract_daily_no2_series(
    no2_files: Sequence[Path],
    city_grid: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    if not no2_files:
        print(
            f"警告：未找到 NO2 输入文件："
            f"{INPUT_FEATURES_DIR / INPUT_FEATURE_PATTERN}"
        )
        return pd.DataFrame(), []

    records: List[dict] = []
    successful_dates: List[pd.Timestamp] = []

    lookup = city_grid[
        ["City", "lat_resampled", "lon_resampled", "weight"]
    ].copy()

    for index, file_path in enumerate(no2_files, start=1):
        print(
            f"NO2  [{index:03d}/{len(no2_files):03d}] "
            f"{file_path.name}",
            end=" ... ",
            flush=True,
        )

        try:
            df = pd.read_pickle(file_path)
            if df.empty:
                print("empty")
                continue

            validate_no2_columns(df, file_path)
            day = infer_single_date(df, file_path)
            if int(day.year) != TARGET_YEAR:
                print("outside target year")
                continue

            successful_dates.append(day)

            lat_r, lon_r = resample_grid_centers(
                df["grid_lat"],
                df["grid_lon"],
                TARGET_RESOLUTION,
            )

            work = pd.DataFrame({
                "lat_resampled": lat_r,
                "lon_resampled": lon_r,
                NO2_COL: pd.to_numeric(df[NO2_COL], errors="coerce"),
            })

            selected = work.merge(
                lookup,
                on=["lat_resampled", "lon_resampled"],
                how="inner",
                validate="many_to_many",
            )
            selected = selected.loc[np.isfinite(selected[NO2_COL])].copy()

            if selected.empty:
                print("no valid city NO2")
                continue

            grid_daily = (
                selected.groupby(
                    [
                        "City",
                        "lat_resampled",
                        "lon_resampled",
                        "weight",
                    ],
                    observed=True,
                    sort=False,
                )[NO2_COL]
                .mean()
                .reset_index(name="no2_grid_mean")
            )

            for city in TARGET_CITIES:
                city_rows = grid_daily[grid_daily["City"] == city]
                if city_rows.empty:
                    continue

                records.append({
                    "City": city,
                    "date": day,
                    "no2_daily_mean": weighted_mean(
                        city_rows["no2_grid_mean"],
                        city_rows["weight"],
                    ),
                    "no2_valid_grid_count": int(len(city_rows)),
                })

            print("success")

        except Exception as exc:
            print(f"failed: {type(exc).__name__}: {exc}")

    daily_no2 = pd.DataFrame(records)
    if not daily_no2.empty:
        daily_no2["date"] = pd.to_datetime(
            daily_no2["date"]
        ).dt.normalize()

    successful_dates = sorted(set(pd.Timestamp(x) for x in successful_dates))
    return daily_no2, successful_dates


# =============================================================================
# 9. 月尺度、连续周尺度和周内周期聚合
# =============================================================================
def month_key(
    date_like: pd.Series | pd.DatetimeIndex | Sequence[pd.Timestamp],
):
    """返回每个日期所属月份的月初日期，同时兼容 Series 和 Index。"""
    converted = pd.to_datetime(date_like)
    if isinstance(converted, pd.Series):
        return converted.dt.to_period("M").dt.to_timestamp()
    return pd.DatetimeIndex(converted).to_period("M").to_timestamp()


def week_key(
    date_like: pd.Series | pd.DatetimeIndex | Sequence[pd.Timestamp],
):
    """周一至周日分组，并以周日日期作为横轴时间。"""
    converted = pd.to_datetime(date_like)
    if isinstance(converted, pd.Series):
        return (
            converted.dt.to_period("W-SUN")
            .dt.end_time
            .dt.normalize()
        )
    return (
        pd.DatetimeIndex(converted)
        .to_period("W-SUN")
        .end_time
        .normalize()
    )


def make_period_denominator(
    dates: Sequence[pd.Timestamp],
    frequency: str,
    count_name: str,
) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame(columns=["period", count_name])

    calendar = pd.DataFrame({
        "date": pd.to_datetime(sorted(set(dates))).normalize()
    })

    if frequency == "monthly":
        calendar["period"] = month_key(calendar["date"])
    elif frequency == "weekly":
        calendar["period"] = week_key(calendar["date"])
    else:
        raise ValueError("frequency 必须是 monthly 或 weekly。")

    return (
        calendar.groupby("period", observed=True)["date"]
        .nunique()
        .reset_index(name=count_name)
    )


def summarize_numeric_series(
    values: pd.Series,
    prefix: str,
) -> Dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p75": np.nan,
        }

    return {
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_std": float(numeric.std(ddof=1)) if len(numeric) > 1 else np.nan,
        f"{prefix}_p25": float(numeric.quantile(0.25)),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_p75": float(numeric.quantile(0.75)),
    }


def aggregate_city_periods(
    daily: pd.DataFrame,
    prediction_dates: Sequence[pd.Timestamp],
    no2_dates: Sequence[pd.Timestamp],
    frequency: str,
) -> pd.DataFrame:
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()

    if frequency == "monthly":
        work["period"] = month_key(work["date"])
        complete_periods = pd.date_range(
            f"{TARGET_YEAR}-01-01",
            f"{TARGET_YEAR}-12-01",
            freq="MS",
        )
    elif frequency == "weekly":
        work["period"] = week_key(work["date"])
        full_dates = pd.date_range(
            f"{TARGET_YEAR}-01-01",
            f"{TARGET_YEAR}-12-31",
            freq="D",
        )
        complete_periods = pd.DatetimeIndex(sorted(set(week_key(full_dates))))
    else:
        raise ValueError("frequency 必须是 monthly 或 weekly。")

    prediction_denominator = make_period_denominator(
        prediction_dates,
        frequency,
        "prediction_period_day_count",
    )
    no2_denominator = make_period_denominator(
        no2_dates,
        frequency,
        "no2_period_day_count",
    )

    output_records: List[dict] = []

    for city in TARGET_CITIES:
        city_daily = work[work["City"] == city]

        for period in complete_periods:
            subset = city_daily[city_daily["period"] == period]

            pred_valid = subset.loc[
                np.isfinite(subset["pred_daily_mean"])
            ]
            no2_valid = subset.loc[
                np.isfinite(subset["no2_daily_mean"])
            ]

            row: Dict[str, object] = {
                "City": city,
                "period": pd.Timestamp(period),
                "pred_valid_day_count": int(pred_valid["date"].nunique()),
                "no2_valid_day_count": int(no2_valid["date"].nunique()),
                "valid_grid_count_mean": (
                    float(pred_valid["valid_grid_count"].mean())
                    if not pred_valid.empty
                    else np.nan
                ),
                "qa0_fraction_within_retained_mean": (
                    float(pred_valid["qa0_fraction_within_retained"].mean())
                    if not pred_valid.empty
                    else np.nan
                ),
            }

            row.update(
                summarize_numeric_series(
                    pred_valid["pred_daily_mean"],
                    "pred",
                )
            )
            row.update(
                summarize_numeric_series(
                    no2_valid["no2_daily_mean"],
                    "no2",
                )
            )

            row["sigma_cal_mean"] = (
                float(pred_valid["sigma_cal_daily_mean"].mean())
                if not pred_valid.empty
                else np.nan
            )
            row["sigma_cal_median"] = (
                float(pred_valid["sigma_cal_daily_mean"].median())
                if not pred_valid.empty
                else np.nan
            )

            output_records.append(row)

    out = pd.DataFrame(output_records)

    out = out.merge(
        prediction_denominator,
        on="period",
        how="left",
    )
    out = out.merge(
        no2_denominator,
        on="period",
        how="left",
    )

    out["prediction_period_day_count"] = (
        out["prediction_period_day_count"].fillna(0).astype(int)
    )
    out["no2_period_day_count"] = (
        out["no2_period_day_count"].fillna(0).astype(int)
    )

    out["valid_day_percent"] = np.divide(
        out["pred_valid_day_count"] * 100.0,
        out["prediction_period_day_count"],
        out=np.full(len(out), np.nan, dtype=float),
        where=out["prediction_period_day_count"].to_numpy() > 0,
    )

    out["no2_valid_day_percent"] = np.divide(
        out["no2_valid_day_count"] * 100.0,
        out["no2_period_day_count"],
        out=np.full(len(out), np.nan, dtype=float),
        where=out["no2_period_day_count"].to_numpy() > 0,
    )

    if frequency == "monthly":
        out["month"] = out["period"].dt.month
        out["month_label"] = out["period"].dt.strftime("%b")
    else:
        out["week_end"] = out["period"]
        out["iso_week"] = out["period"].dt.isocalendar().week.astype(int)

    return out.sort_values(["City", "period"]).reset_index(drop=True)


def aggregate_city_weekdays(
    daily: pd.DataFrame,
    prediction_dates: Sequence[pd.Timestamp],
    no2_dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """
    按周内日（Monday=0, ..., Sunday=6）汇总四城市逐日序列。

    输出的是 2023 全年所有同一周内日的统计分布，不是某一具体自然周。
    """
    work = daily.copy()
    work["date"] = pd.to_datetime(
        work["date"],
        errors="coerce",
    ).dt.normalize()
    work = work.loc[work["date"].notna()].copy()
    work["weekday"] = work["date"].dt.weekday.astype(int)

    prediction_calendar = pd.DataFrame({
        "date": pd.to_datetime(
            sorted(set(pd.Timestamp(x) for x in prediction_dates)),
            errors="coerce",
        )
    }).dropna()
    prediction_calendar["weekday"] = (
        prediction_calendar["date"].dt.weekday.astype(int)
    )
    prediction_denominator = (
        prediction_calendar.groupby("weekday", observed=True)["date"]
        .nunique()
        .reindex(range(7), fill_value=0)
        .rename("prediction_weekday_day_count")
        .reset_index()
    )

    no2_calendar = pd.DataFrame({
        "date": pd.to_datetime(
            sorted(set(pd.Timestamp(x) for x in no2_dates)),
            errors="coerce",
        )
    }).dropna()
    if no2_calendar.empty:
        no2_denominator = pd.DataFrame({
            "weekday": np.arange(7, dtype=int),
            "no2_weekday_day_count": np.zeros(7, dtype=int),
        })
    else:
        no2_calendar["weekday"] = (
            no2_calendar["date"].dt.weekday.astype(int)
        )
        no2_denominator = (
            no2_calendar.groupby("weekday", observed=True)["date"]
            .nunique()
            .reindex(range(7), fill_value=0)
            .rename("no2_weekday_day_count")
            .reset_index()
        )

    records: List[dict] = []

    for city in TARGET_CITIES:
        city_daily = work[work["City"] == city]

        for weekday in range(7):
            subset = city_daily[city_daily["weekday"] == weekday]

            pred_valid = subset.loc[
                np.isfinite(
                    pd.to_numeric(
                        subset["pred_daily_mean"],
                        errors="coerce",
                    )
                )
            ]
            no2_valid = subset.loc[
                np.isfinite(
                    pd.to_numeric(
                        subset["no2_daily_mean"],
                        errors="coerce",
                    )
                )
            ]

            row: Dict[str, object] = {
                "City": city,
                "weekday": int(weekday),
                "weekday_label": WEEKDAY_LABELS[weekday],
                "pred_valid_day_count": int(
                    pred_valid["date"].nunique()
                ),
                "no2_valid_day_count": int(
                    no2_valid["date"].nunique()
                ),
                "valid_grid_count_mean": (
                    float(pred_valid["valid_grid_count"].mean())
                    if not pred_valid.empty
                    else np.nan
                ),
                "qa0_fraction_within_retained_mean": (
                    float(
                        pred_valid[
                            "qa0_fraction_within_retained"
                        ].mean()
                    )
                    if not pred_valid.empty
                    else np.nan
                ),
            }

            row.update(
                summarize_numeric_series(
                    pred_valid["pred_daily_mean"],
                    "pred",
                )
            )
            row.update(
                summarize_numeric_series(
                    no2_valid["no2_daily_mean"],
                    "no2",
                )
            )

            row["sigma_cal_mean"] = (
                float(
                    pred_valid["sigma_cal_daily_mean"].mean()
                )
                if not pred_valid.empty
                else np.nan
            )
            row["sigma_cal_median"] = (
                float(
                    pred_valid["sigma_cal_daily_mean"].median()
                )
                if not pred_valid.empty
                else np.nan
            )

            records.append(row)

    out = pd.DataFrame(records)
    out = out.merge(
        prediction_denominator,
        on="weekday",
        how="left",
    )
    out = out.merge(
        no2_denominator,
        on="weekday",
        how="left",
    )

    out["prediction_weekday_day_count"] = (
        out["prediction_weekday_day_count"]
        .fillna(0)
        .astype(int)
    )
    out["no2_weekday_day_count"] = (
        out["no2_weekday_day_count"]
        .fillna(0)
        .astype(int)
    )

    out["valid_day_percent"] = np.divide(
        out["pred_valid_day_count"] * 100.0,
        out["prediction_weekday_day_count"],
        out=np.full(len(out), np.nan, dtype=float),
        where=(
            out["prediction_weekday_day_count"].to_numpy()
            > 0
        ),
    )
    out["no2_valid_day_percent"] = np.divide(
        out["no2_valid_day_count"] * 100.0,
        out["no2_weekday_day_count"],
        out=np.full(len(out), np.nan, dtype=float),
        where=(
            out["no2_weekday_day_count"].to_numpy()
            > 0
        ),
    )

    return (
        out.sort_values(["City", "weekday"])
        .reset_index(drop=True)
    )


# =============================================================================
# 10. 图形绘制
# =============================================================================
def style_time_axis(ax) -> None:
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35, color="#B0B0B0")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def draw_panel_label(ax, text: str) -> None:
    ax.text(
        0.015,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=FONT_PANEL,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "pad": 2,
        },
        zorder=10,
    )


def plot_city_period_figure(
    summary: pd.DataFrame,
    frequency: str,
    output_path: Path,
) -> None:
    """
    绘制月尺度、连续周尺度或周内周期四城市变化。

    每张图仅包含：
        (a) Tropospheric NO2 VCD
        (b) QA-screened ΔXCO2
    """
    if summary.empty:
        raise ValueError(
            f"{frequency} 汇总表为空，无法绘图。"
        )

    if frequency == "monthly":
        x_column = "month"
        figure_size = (10.5, 7.2)
        x_label = "Month"
    elif frequency == "weekly":
        x_column = "period"
        figure_size = (14.0, 7.2)
        x_label = "Week ending"
    elif frequency == "weekday":
        x_column = "weekday"
        figure_size = (10.5, 7.2)
        x_label = "Day of week"
    else:
        raise ValueError(
            "frequency 必须是 monthly、weekly 或 weekday。"
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size,
        sharex=True,
    )

    for city, info in TARGET_CITIES.items():
        if frequency == "weekday":
            city_data = (
                summary[summary["City"] == city]
                .sort_values("weekday")
            )
        else:
            city_data = (
                summary[summary["City"] == city]
                .sort_values("period")
            )

        if city_data.empty:
            print(f"警告：{frequency} 图中 {city} 没有数据。")
            continue

        color = str(info["color"])

        if frequency == "weekly":
            x = pd.to_datetime(
                city_data[x_column],
                errors="coerce",
            )
        else:
            x = pd.to_numeric(
                city_data[x_column],
                errors="coerce",
            ).to_numpy(dtype=float)

        no2_mean = pd.to_numeric(
            city_data["no2_mean"],
            errors="coerce",
        ).to_numpy(dtype=float)
        no2_p25 = pd.to_numeric(
            city_data["no2_p25"],
            errors="coerce",
        ).to_numpy(dtype=float)
        no2_p75 = pd.to_numeric(
            city_data["no2_p75"],
            errors="coerce",
        ).to_numpy(dtype=float)

        pred_mean = pd.to_numeric(
            city_data["pred_mean"],
            errors="coerce",
        ).to_numpy(dtype=float)
        pred_p25 = pd.to_numeric(
            city_data["pred_p25"],
            errors="coerce",
        ).to_numpy(dtype=float)
        pred_p75 = pd.to_numeric(
            city_data["pred_p75"],
            errors="coerce",
        ).to_numpy(dtype=float)
        sigma_mean = pd.to_numeric(
            city_data["sigma_cal_mean"],
            errors="coerce",
        ).to_numpy(dtype=float)

        # Panel (a): NO2
        axes[0].plot(
            x,
            no2_mean,
            color=color,
            marker="o",
            markersize=4.4,
            linewidth=1.9,
            label=city,
            zorder=4,
        )
        axes[0].fill_between(
            x,
            no2_p25,
            no2_p75,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

        # Panel (b): ΔXCO2 + calibrated predictive uncertainty
        axes[1].plot(
            x,
            pred_mean,
            color=color,
            marker="o",
            markersize=4.4,
            linewidth=1.9,
            zorder=4,
        )
        axes[1].fill_between(
            x,
            pred_p25,
            pred_p75,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )
        axes[1].errorbar(
            x,
            pred_mean,
            yerr=sigma_mean,
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=1.8,
            alpha=0.38,
            zorder=3,
        )

    axes[0].set_ylabel(
        NO2_YLABEL,
        fontsize=FONT_AXIS,
    )
    axes[1].set_ylabel(
        PRED_YLABEL,
        fontsize=FONT_AXIS,
    )
    axes[1].set_xlabel(
        x_label,
        fontsize=FONT_AXIS,
    )

    draw_panel_label(axes[0], "(a)")
    draw_panel_label(axes[1], "(b)")

    for ax in axes:
        style_time_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=FONT_LEGEND,
        columnspacing=1.8,
        handlelength=2.4,
    )

    if frequency == "monthly":
        axes[1].set_xticks(np.arange(1, 13))
        axes[1].set_xticklabels(
            [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ],
            fontsize=FONT_TICK,
        )
        axes[1].set_xlim(0.7, 12.3)

    elif frequency == "weekly":
        axes[1].xaxis.set_major_locator(
            mdates.MonthLocator()
        )
        axes[1].xaxis.set_major_formatter(
            mdates.DateFormatter("%b")
        )
        axes[1].xaxis.set_minor_locator(
            mdates.WeekdayLocator(byweekday=mdates.MO)
        )
        axes[1].set_xlim(
            pd.Timestamp(f"{TARGET_YEAR}-01-01"),
            pd.Timestamp(f"{TARGET_YEAR}-12-31"),
        )

    else:
        axes[1].set_xticks(np.arange(7))
        axes[1].set_xticklabels(
            WEEKDAY_LABELS,
            fontsize=FONT_TICK,
        )
        axes[1].set_xlim(-0.25, 6.25)

    # 误差棒表示 calibrated predictive uncertainty，而非均值标准误。
    axes[1].text(
        0.995,
        0.03,
        "Error bars: mean calibrated predictive uncertainty",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.935,
        hspace=0.10,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"图像已保存：{output_path}")


# =============================================================================
# 11. 主程序
# =============================================================================
def read_summary_csv(
    path: Path,
    frequency: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"汇总 CSV 不存在：{path}")

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    if frequency in {"monthly", "weekly"}:
        if "period" not in df.columns:
            raise KeyError(f"{path.name} 缺少 period 字段。")
        df["period"] = pd.to_datetime(
            df["period"],
            errors="coerce",
        )

    if frequency == "monthly":
        if "month" not in df.columns:
            df["month"] = df["period"].dt.month
    elif frequency == "weekday":
        if "weekday" not in df.columns:
            raise KeyError(f"{path.name} 缺少 weekday 字段。")
        df["weekday"] = pd.to_numeric(
            df["weekday"],
            errors="coerce",
        ).astype("Int64")

    return df


def infer_dates_from_daily_csv(
    daily: pd.DataFrame,
) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(
        daily["date"],
        errors="coerce",
    ).dt.normalize()
    daily = daily.loc[daily["date"].notna()]

    prediction_dates = sorted(
        set(pd.Timestamp(x) for x in daily["date"].unique())
    )

    no2_dates: List[pd.Timestamp] = []
    if "no2_daily_mean" in daily.columns:
        no2_valid = daily.loc[
            np.isfinite(
                pd.to_numeric(
                    daily["no2_daily_mean"],
                    errors="coerce",
                )
            ),
            "date",
        ]
        no2_dates = sorted(
            set(pd.Timestamp(x) for x in no2_valid.unique())
        )

    return prediction_dates, no2_dates


def run_plot_only_mode() -> None:
    print("运行 plot_only 模式：直接读取已有 CSV。")

    monthly = read_summary_csv(
        MONTHLY_CSV,
        frequency="monthly",
    )
    weekly = read_summary_csv(
        WEEKLY_CSV,
        frequency="weekly",
    )

    if WEEKDAY_CSV.exists():
        weekday = read_summary_csv(
            WEEKDAY_CSV,
            frequency="weekday",
        )
    else:
        if not DAILY_CSV.exists():
            raise FileNotFoundError(
                "缺少周内 CSV，且没有逐日 CSV 可用于快速重建："
                f"\n  {WEEKDAY_CSV}\n  {DAILY_CSV}"
            )

        print(
            "周内 CSV 不存在；从逐日 CSV 快速聚合，"
            "无需重新读取 PKL。"
        )
        daily = pd.read_csv(
            DAILY_CSV,
            encoding="utf-8-sig",
        )
        daily["date"] = pd.to_datetime(
            daily["date"],
            errors="coerce",
        )
        prediction_dates, no2_dates = (
            infer_dates_from_daily_csv(daily)
        )
        weekday = aggregate_city_weekdays(
            daily,
            prediction_dates,
            no2_dates,
        )
        weekday.to_csv(
            WEEKDAY_CSV,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
        print(f"周内统计表已保存：{WEEKDAY_CSV}")

    plot_city_period_figure(
        monthly,
        frequency="monthly",
        output_path=MONTHLY_FIGURE,
    )
    plot_city_period_figure(
        weekly,
        frequency="weekly",
        output_path=WEEKLY_FIGURE,
    )
    plot_city_period_figure(
        weekday,
        frequency="weekday",
        output_path=WEEKDAY_FIGURE,
    )


def run_full_mode() -> None:
    prediction_files = find_prediction_files()
    no2_files = find_no2_files()

    china_geometry = build_china_union_geometry()
    city_grid = build_city_grid_definition(
        prediction_files,
        china_geometry,
    )

    daily_prediction, successful_prediction_dates = (
        extract_daily_prediction_series(
            prediction_files,
            city_grid,
        )
    )

    daily_no2, successful_no2_dates = extract_daily_no2_series(
        no2_files,
        city_grid,
    )

    if daily_no2.empty:
        daily = daily_prediction.copy()
        daily["no2_daily_mean"] = np.nan
        daily["no2_valid_grid_count"] = 0
    else:
        daily = daily_prediction.merge(
            daily_no2,
            on=["City", "date"],
            how="outer",
            validate="one_to_one",
        )

    # 补齐四城市 × 全部成功预测日期，明确保留缺测日。
    complete_index = pd.MultiIndex.from_product(
        [
            list(TARGET_CITIES.keys()),
            pd.DatetimeIndex(successful_prediction_dates),
        ],
        names=["City", "date"],
    )
    daily = (
        daily.set_index(["City", "date"])
        .reindex(complete_index)
        .reset_index()
        .sort_values(["City", "date"])
    )

    count_columns = [
        "valid_grid_count",
        "qa0_record_count",
        "qa1_record_count",
        "no2_valid_grid_count",
    ]
    for column in count_columns:
        if column not in daily.columns:
            daily[column] = 0
        daily[column] = (
            daily[column]
            .fillna(0)
            .astype(int)
        )

    daily["qa_selection"] = QA_SELECTION_TAG
    daily["city_selection_mode"] = CITY_SELECTION_MODE
    daily["city_radius_km"] = (
        CITY_RADIUS_KM
        if CITY_SELECTION_MODE.lower() == "radius"
        else np.nan
    )

    monthly = aggregate_city_periods(
        daily,
        successful_prediction_dates,
        successful_no2_dates,
        frequency="monthly",
    )
    weekly = aggregate_city_periods(
        daily,
        successful_prediction_dates,
        successful_no2_dates,
        frequency="weekly",
    )
    weekday = aggregate_city_weekdays(
        daily,
        successful_prediction_dates,
        successful_no2_dates,
    )

    daily.to_csv(
        DAILY_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    monthly.to_csv(
        MONTHLY_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    weekly.to_csv(
        WEEKLY_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    weekday.to_csv(
        WEEKDAY_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    plot_city_period_figure(
        monthly,
        frequency="monthly",
        output_path=MONTHLY_FIGURE,
    )
    plot_city_period_figure(
        weekly,
        frequency="weekly",
        output_path=WEEKLY_FIGURE,
    )
    plot_city_period_figure(
        weekday,
        frequency="weekday",
        output_path=WEEKDAY_FIGURE,
    )

    print("\n输出完成：")
    print(f"  City grid definition: {CITY_GRID_CSV}")
    print(f"  Daily statistics:     {DAILY_CSV}")
    print(f"  Monthly statistics:   {MONTHLY_CSV}")
    print(f"  Weekly statistics:    {WEEKLY_CSV}")
    print(f"  Weekday statistics:   {WEEKDAY_CSV}")
    print(f"  Monthly figure:       {MONTHLY_FIGURE}")
    print(f"  Weekly figure:        {WEEKLY_FIGURE}")
    print(f"  Weekday figure:       {WEEKDAY_FIGURE}")


def main() -> None:
    print("=" * 78)
    print("FIG5 city case-study pipeline")
    print(f"Target year: {TARGET_YEAR}")
    print(f"QA selection: {FIXED_KEEP_QA} ({QA_SELECTION_TAG})")
    print(
        f"City selection: {CITY_SELECTION_MODE}; "
        f"radius={CITY_RADIUS_KM:.1f} km"
    )
    print(f"Run mode: {RUN_MODE}")
    print("=" * 78)

    mode = RUN_MODE.strip().lower()
    if mode == "full":
        run_full_mode()
    elif mode == "plot_only":
        run_plot_only_mode()
    else:
        raise ValueError(
            "RUN_MODE 只能设置为 'full' 或 'plot_only'。"
        )


if __name__ == "__main__":
    main()