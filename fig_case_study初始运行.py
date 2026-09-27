# -*- coding: utf-8 -*-
"""
fig5_city_qa01_qa012_mask_smooth.py

城市尺度月变化与星期变化双轴图：
    - 左轴：Predicted Delta XCO2
    - 右轴：Tropospheric NO2 VCD
    - 圆点保留为 Delta XCO2 聚合均值
    - 方块保留为 NO2 聚合均值
    - Delta XCO2 中心曲线、上下阴影边界均平滑绘制
    - NO2 上下阴影边界平滑绘制；可选绘制 NO2 平滑虚线

质量控制与空间筛选参照 fig3and5_flag012_mask.py：
    1. qa_flag 属于指定 QA 集合；
    2. pred_xco2_enhanced、sigma_cal、经纬度均为有限值；
    3. 可选：中国省级行政区面裁剪；
    4. 可选：手动经纬度矩形 MASK。

固定输出两套预测 CSV：
    - QA01：qa_flag in (0, 1)，用于正文制图；
    - QA012：qa_flag in (0, 1, 2)，用于质量控制前对照。

运行模式：
    PLOT_ONLY = False
        逐个读取 PKL，仅提取四个城市对应网格，输出全部 CSV，再绘图。

    PLOT_ONLY = True
        不读取原始 PKL，直接读取 QA01 聚合 CSV 和 NO2 聚合 CSV 绘图。

说明：
    - QA 筛选只针对预测产品；NO2 本身没有 qa_flag。
    - 若某城市对应的预测网格位于中国行政区面外，或落入启用的手动 MASK，
      该城市的聚合均值与阴影会设为 NaN，不参与正文绘图。
    - 默认 APPLY_MASK_TO_NO2 = True，使同一城市的 NO2 也同步不显示，避免
      一个城市只剩右轴方块而左轴预测已被 MASK 的不一致情况。
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# =============================================================================
# 1. 路径与运行配置
# =============================================================================
ACTIVE_VERSION = "A01"
PKL_DIR = Path(
    "/home/whdong/dl/ML-prediction-output_result"
) / ACTIVE_VERSION
FILE_PATTERN = "pred_0.1deg_0.1deg*.pkl"

INPUT_FEATURES_DIR = Path(
    "/home/whdong/dl/ML-prediction_input_data"
)
NO2_FILE_PATTERN = "post_data_*.pkl"

OUTPUT_DIR = PKL_DIR / "figures_case_study_qa_mask"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# False：从 PKL 重新处理并输出所有 CSV；True：直接使用已有 CSV 绘图。
PLOT_ONLY = False

TARGET_RES = 0.1
PRED_COL = "pred_xco2_enhanced"
SIGMA_COL = "sigma_cal"
QA_COL = "qa_flag"
NO2_COL = "no2_trop"

# 正文图固定使用 QA=0,1。
PLOT_KEEP_QA = (0, 1)
PLOT_QA_TAG = "QA01"

# FULL 模式固定同时输出正文 QA01 和质量控制前对照 QA012。
EXPORT_QA_SCHEMES: Mapping[str, Tuple[int, ...]] = {
    "QA01": (0, 1),
    "QA012": (0, 1, 2),
}

# 输出图。
OUTPUT_MONTHLY = (
    OUTPUT_DIR
    / "FIG5-City_Monthly_TimeSeries_QA01_masked_smooth.png"
)
OUTPUT_WEEKDAY = (
    OUTPUT_DIR
    / "FIG5-City_Weekday_Cycle_QA01_masked_smooth.png"
)


# =============================================================================
# 2. QA、China shape 和手动 MASK 配置
# =============================================================================
CHINA_PROV_SHP = Path(
    "/home/whdong/shapefile/china/province.shp"
)

# True：仅保留城市预测网格中心位于中国省级行政区面并集内的城市。
APPLY_CHINA_SHAPE_CLIP = True

# True：应用下面启用的经纬度矩形 MASK。
APPLY_MANUAL_MASK = True

# True：若某城市预测网格被 China shape/MASK 排除，则同步隐藏该城市 NO2。
APPLY_MASK_TO_NO2 = True

# 边界为闭区间；None 表示该方向无限制。
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


# =============================================================================
# 3. 平滑绘图配置
# =============================================================================
# "poly"：保持原代码的多项式拟合思路；
# "pchip"：若安装 scipy，则采用形状保持三次插值，过冲通常更小。
SMOOTH_METHOD = "poly"
POLY_DEGREE = 3
SMOOTH_POINT_COUNT = 300

# NO2 原代码只有方块和阴影。设为 True 可额外增加平滑虚线。
DRAW_NO2_SMOOTH_LINE = False

# 变量形状说明图例。
SHOW_MARKER_TYPE_LEGEND = True


# =============================================================================
# 4. 城市和绘图格式
# =============================================================================
TARGET_CITIES = {
    "Beijing": {
        "lat": 39.9,
        "lon": 116.4,
        "color": "#9B59B6",
    },
    "Shanghai": {
        "lat": 31.2,
        "lon": 121.5,
        "color": "#34495E",
    },
    "Wuhan": {
        "lat": 30.6,
        "lon": 114.3,
        "color": "#E67E22",
    },
    "Guangzhou": {
        "lat": 23.1,
        "lon": 113.3,
        "color": "#A2B836",
    },
}

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
GLOBAL_FONT_SIZE = 13

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": GLOBAL_FONT_SIZE,
    "axes.titlesize": GLOBAL_FONT_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": GLOBAL_FONT_SIZE,
    "xtick.labelsize": GLOBAL_FONT_SIZE - 1,
    "ytick.labelsize": GLOBAL_FONT_SIZE - 1,
    "legend.fontsize": GLOBAL_FONT_SIZE - 1,
    "axes.linewidth": 1.5,
    "lines.linewidth": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.35,
    "grid.color": "#B0B0B0",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# =============================================================================
# 5. CSV 路径
# =============================================================================
def csv_pred_records(tag: str) -> Path:
    return OUTPUT_DIR / f"FIG5_city_prediction_records_{tag}.csv"


def csv_pred_month(tag: str) -> Path:
    return OUTPUT_DIR / f"FIG5_intermediate_pred_month_{tag}.csv"


def csv_pred_weekday(tag: str) -> Path:
    return OUTPUT_DIR / f"FIG5_intermediate_pred_weekday_{tag}.csv"


def csv_no2_records() -> Path:
    return OUTPUT_DIR / "FIG5_city_no2_records.csv"


def csv_no2_month() -> Path:
    return OUTPUT_DIR / "FIG5_intermediate_no2_month.csv"


def csv_no2_weekday() -> Path:
    return OUTPUT_DIR / "FIG5_intermediate_no2_weekday.csv"


CSV_CITY_GRID_SELECTION = OUTPUT_DIR / "FIG5_city_grid_selection.csv"
CSV_QA_STATS_CITY = OUTPUT_DIR / "FIG5_QA_Statistics_ByCity.csv"
CSV_QA_STATS_MONTH = OUTPUT_DIR / "FIG5_QA_Statistics_ByCityMonth.csv"
CSV_MASK_STATS = OUTPUT_DIR / "FIG5_Manual_Spatial_Mask_Statistics.csv"
CSV_PROCESSING_LOG = OUTPUT_DIR / "FIG5_Prediction_File_Processing_Log.csv"


# =============================================================================
# 6. 通用工具
# =============================================================================
def save_csv(df: pd.DataFrame, path: Path, float_format: str = "%.6f") -> None:
    """即使 DataFrame 为空，也输出带表头 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format=float_format,
    )
    print(f"  Saved CSV: {path}")


def resample_grid_centers(
    lat: pd.Series,
    lon: pd.Series,
    target_res: float,
) -> Tuple[pd.Series, pd.Series]:
    """将经纬度统一映射到 target_res 网格中心，并适当舍入。"""
    decimals = max(2, int(np.ceil(-np.log10(target_res))) + 2)

    lat_numeric = pd.to_numeric(lat, errors="coerce")
    lon_numeric = pd.to_numeric(lon, errors="coerce")

    lat_r = (
        np.floor(lat_numeric / target_res) * target_res
        + target_res / 2.0
    ).round(decimals)
    lon_r = (
        np.floor(lon_numeric / target_res) * target_res
        + target_res / 2.0
    ).round(decimals)
    return lat_r, lon_r


def validate_prediction_columns(df: pd.DataFrame, file_path: Path) -> None:
    required = {
        "grid_lat", "grid_lon", "date",
        PRED_COL, SIGMA_COL, QA_COL,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{file_path.name} 缺少列: {missing}")


def validate_no2_columns(df: pd.DataFrame, file_path: Path) -> None:
    required = {"grid_lat", "grid_lon", "date", NO2_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{file_path.name} 缺少列: {missing}")


def validate_manual_mask_boxes(
    mask_boxes: Sequence[Mapping[str, object]],
) -> None:
    """检查启用的矩形 MASK 配置是否合法。"""
    for index, box in enumerate(mask_boxes, start=1):
        name = str(box.get("name", f"mask_{index}"))
        lon_min = box.get("lon_min")
        lon_max = box.get("lon_max")
        lat_min = box.get("lat_min")
        lat_max = box.get("lat_max")

        for field_name, value in [
            ("lon_min", lon_min),
            ("lon_max", lon_max),
            ("lat_min", lat_min),
            ("lat_max", lat_max),
        ]:
            if value is not None:
                try:
                    float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"MASK {name!r} 的 {field_name} "
                        f"不是数字或 None: {value!r}"
                    ) from exc

        if lon_min is not None and lon_max is not None:
            if float(lon_min) > float(lon_max):
                raise ValueError(
                    f"MASK {name!r}: lon_min 不能大于 lon_max"
                )

        if lat_min is not None and lat_max is not None:
            if float(lat_min) > float(lat_max):
                raise ValueError(
                    f"MASK {name!r}: lat_min 不能大于 lat_max"
                )

        if all(
            value is None
            for value in [lon_min, lon_max, lat_min, lat_max]
        ):
            raise ValueError(
                f"MASK {name!r} 未设置任何边界，会掩膜全部网格。"
            )


def evaluate_manual_masks(
    lat: np.ndarray,
    lon: np.ndarray,
    mask_boxes: Sequence[Mapping[str, object]],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """对给定网格中心判断多个手动矩形 MASK。"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if len(lat) != len(lon):
        raise ValueError("lat 和 lon 长度不一致。")

    if not APPLY_MANUAL_MASK:
        return (
            np.zeros(len(lat), dtype=bool),
            [""] * len(lat),
            pd.DataFrame([{
                "mask_name": "ALL_MASKS_DISABLED",
                "masked_grid_count": 0,
            }]),
        )

    enabled_boxes = [
        box for box in mask_boxes
        if bool(box.get("enabled", True))
    ]

    if not enabled_boxes:
        return (
            np.zeros(len(lat), dtype=bool),
            [""] * len(lat),
            pd.DataFrame([{
                "mask_name": "NO_ENABLED_MASK_BOX",
                "masked_grid_count": 0,
            }]),
        )

    validate_manual_mask_boxes(enabled_boxes)

    finite = np.isfinite(lat) & np.isfinite(lon)
    total_mask = np.zeros(len(lat), dtype=bool)
    names: List[List[str]] = [[] for _ in range(len(lat))]
    stats: List[dict] = []

    for index, box in enumerate(enabled_boxes, start=1):
        name = str(box.get("name", f"mask_{index}"))
        lon_min = box.get("lon_min")
        lon_max = box.get("lon_max")
        lat_min = box.get("lat_min")
        lat_max = box.get("lat_max")

        box_mask = finite.copy()
        if lon_min is not None:
            box_mask &= lon >= float(lon_min)
        if lon_max is not None:
            box_mask &= lon <= float(lon_max)
        if lat_min is not None:
            box_mask &= lat >= float(lat_min)
        if lat_max is not None:
            box_mask &= lat <= float(lat_max)

        for row_index in np.flatnonzero(box_mask):
            names[int(row_index)].append(name)

        total_mask |= box_mask
        stats.append({
            "mask_name": name,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "masked_grid_count": int(box_mask.sum()),
        })

    stats.append({
        "mask_name": "UNION_OF_ENABLED_MASKS",
        "lon_min": np.nan,
        "lon_max": np.nan,
        "lat_min": np.nan,
        "lat_max": np.nan,
        "masked_grid_count": int(total_mask.sum()),
    })

    return (
        total_mask,
        [";".join(item) for item in names],
        pd.DataFrame(stats),
    )


def build_china_union_geometry():
    """读取中国省级面要素并合并；实现方式与参考脚本一致。"""
    if not CHINA_PROV_SHP.exists():
        raise FileNotFoundError(
            f"中国省级行政区 shapefile 不存在: {CHINA_PROV_SHP}"
        )

    try:
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError(
            "APPLY_CHINA_SHAPE_CLIP=True 时需要 cartopy 和 shapely。"
        ) from exc

    try:
        reader = shpreader.Reader(str(CHINA_PROV_SHP), encoding="gbk")
    except TypeError:
        reader = shpreader.Reader(str(CHINA_PROV_SHP))

    geometries = [
        geom for geom in reader.geometries()
        if geom is not None and not geom.is_empty
    ]
    if not geometries:
        raise ValueError(
            f"shapefile 中没有可用几何对象: {CHINA_PROV_SHP}"
        )

    china_geometry = unary_union(geometries)
    if china_geometry.is_empty:
        raise ValueError("合并后的中国范围几何为空。")

    if not china_geometry.is_valid:
        repaired = china_geometry.buffer(0)
        if not repaired.is_empty:
            china_geometry = repaired

    return china_geometry


def points_inside_china(
    lat: np.ndarray,
    lon: np.ndarray,
    china_geometry,
) -> np.ndarray:
    """判断网格中心是否位于中国省级面要素并集内，包含边界点。"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    inside = np.zeros(len(lat), dtype=bool)

    if not finite.any():
        return inside

    try:
        from shapely import intersects_xy
    except ImportError:
        intersects_xy = None

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
        from shapely.geometry import Point
        from shapely.prepared import prep

        prepared = prep(china_geometry)
        inside[finite] = np.fromiter(
            (
                prepared.covers(Point(x, y))
                for x, y in zip(lon[finite], lat[finite])
            ),
            dtype=bool,
            count=int(finite.sum()),
        )

    return inside


# =============================================================================
# 7. 城市预测网格选择、QA 统计和聚合
# =============================================================================
def find_reference_prediction_grid(
    files: Sequence[Path],
) -> Tuple[pd.DataFrame, Path]:
    """从首个可用预测文件获得稳定网格域。"""
    for file_path in files:
        try:
            df = pd.read_pickle(file_path)
            if df.empty:
                continue
            validate_prediction_columns(df, file_path)

            lat_r, lon_r = resample_grid_centers(
                df["grid_lat"],
                df["grid_lon"],
                TARGET_RES,
            )
            grid_lat = pd.to_numeric(df["grid_lat"], errors="coerce")
            grid_lon = pd.to_numeric(df["grid_lon"], errors="coerce")
            finite = (
                np.isfinite(lat_r)
                & np.isfinite(lon_r)
                & np.isfinite(grid_lat)
                & np.isfinite(grid_lon)
            )

            grid = pd.DataFrame({
                "lat_r": lat_r.loc[finite].to_numpy(),
                "lon_r": lon_r.loc[finite].to_numpy(),
                "grid_lat": grid_lat.loc[finite].to_numpy(),
                "grid_lon": grid_lon.loc[finite].to_numpy(),
            })
            grid = (
                grid.groupby(["lat_r", "lon_r"], as_index=False)
                [["grid_lat", "grid_lon"]]
                .mean()
            )
            if not grid.empty:
                return grid, file_path
        except Exception as exc:
            print(
                f"  Reference grid skip {file_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    raise RuntimeError("没有找到可用于确定城市网格的预测文件。")


def select_city_prediction_grids(
    files: Sequence[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """选择离四个目标城市最近的预测网格，并评估 China clip/MASK。"""
    grid, reference_file = find_reference_prediction_grid(files)
    rows: List[dict] = []

    for city, info in TARGET_CITIES.items():
        dist2 = (
            (grid["grid_lat"] - float(info["lat"])) ** 2
            + (grid["grid_lon"] - float(info["lon"])) ** 2
        )
        idx = dist2.idxmin()
        selected = grid.loc[idx]
        rows.append({
            "City": city,
            "target_lat": float(info["lat"]),
            "target_lon": float(info["lon"]),
            "selected_grid_lat": float(selected["grid_lat"]),
            "selected_grid_lon": float(selected["grid_lon"]),
            "lat_r": float(selected["lat_r"]),
            "lon_r": float(selected["lon_r"]),
            "distance_degree": float(np.sqrt(dist2.loc[idx])),
            "reference_grid_file": reference_file.name,
        })

    city_grids = pd.DataFrame(rows)

    if APPLY_CHINA_SHAPE_CLIP:
        china_geometry = build_china_union_geometry()
        city_grids["inside_china_shape"] = points_inside_china(
            city_grids["lat_r"].to_numpy(dtype=float),
            city_grids["lon_r"].to_numpy(dtype=float),
            china_geometry,
        )
    else:
        city_grids["inside_china_shape"] = True

    masked, mask_names, mask_stats = evaluate_manual_masks(
        city_grids["lat_r"].to_numpy(dtype=float),
        city_grids["lon_r"].to_numpy(dtype=float),
        MANUAL_MASK_BOXES,
    )
    city_grids["manual_masked"] = masked
    city_grids["manual_mask_name"] = mask_names
    city_grids["available_for_plot"] = (
        city_grids["inside_china_shape"].astype(bool)
        & ~city_grids["manual_masked"].astype(bool)
    )

    city_mask_rows = city_grids[[
        "City", "lat_r", "lon_r", "inside_china_shape",
        "manual_masked", "manual_mask_name", "available_for_plot",
    ]].copy()
    city_mask_rows.insert(0, "record_type", "city_selection")

    mask_stats = mask_stats.copy()
    mask_stats.insert(0, "record_type", "mask_box_summary")

    combined_mask_stats = pd.concat(
        [city_mask_rows, mask_stats],
        ignore_index=True,
        sort=False,
    )
    return city_grids, combined_mask_stats


def load_prediction_city_records(
    files: Sequence[Path],
    city_grids: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    流式读取预测 PKL，仅保留四个城市对应网格。

    此阶段保留所有 qa_flag，具体 QA01/QA012 在后续分别筛选。
    """
    grid_keys = city_grids[[
        "City", "lat_r", "lon_r",
        "inside_china_shape", "manual_masked",
        "manual_mask_name", "available_for_plot",
    ]].copy()

    records: List[pd.DataFrame] = []
    logs: List[dict] = []

    for index, file_path in enumerate(files, start=1):
        print(
            f"[{index:03d}/{len(files):03d}] "
            f"Prediction {file_path.name} ... ",
            end="",
            flush=True,
        )
        try:
            df = pd.read_pickle(file_path)
            if df.empty:
                print("skipped (empty)")
                logs.append({
                    "file": file_path.name,
                    "status": "empty",
                    "input_row_count": 0,
                    "matched_city_row_count": 0,
                    "message": "",
                })
                continue

            validate_prediction_columns(df, file_path)

            lat_r, lon_r = resample_grid_centers(
                df["grid_lat"],
                df["grid_lon"],
                TARGET_RES,
            )
            work = pd.DataFrame({
                "date": pd.to_datetime(df["date"], errors="coerce"),
                "lat_r": lat_r,
                "lon_r": lon_r,
                "grid_lat": pd.to_numeric(
                    df["grid_lat"], errors="coerce"
                ),
                "grid_lon": pd.to_numeric(
                    df["grid_lon"], errors="coerce"
                ),
                PRED_COL: pd.to_numeric(
                    df[PRED_COL], errors="coerce"
                ),
                SIGMA_COL: pd.to_numeric(
                    df[SIGMA_COL], errors="coerce"
                ),
                QA_COL: pd.to_numeric(
                    df[QA_COL], errors="coerce"
                ),
            })

            matched = work.merge(
                grid_keys,
                on=["lat_r", "lon_r"],
                how="inner",
                validate="many_to_one",
            )

            if not matched.empty:
                matched["Month"] = matched["date"].dt.month
                matched["weekday"] = matched["date"].dt.dayofweek
                matched["source_file"] = file_path.name
                matched["finite_prediction_fields"] = (
                    np.isfinite(matched[PRED_COL])
                    & np.isfinite(matched[SIGMA_COL])
                    & np.isfinite(matched["grid_lat"])
                    & np.isfinite(matched["grid_lon"])
                    & matched["date"].notna()
                )
                records.append(matched)

            print(f"success, matched={len(matched)}")
            logs.append({
                "file": file_path.name,
                "status": "success",
                "input_row_count": int(len(df)),
                "matched_city_row_count": int(len(matched)),
                "message": "",
            })
        except Exception as exc:
            print(f"failed: {type(exc).__name__}: {exc}")
            logs.append({
                "file": file_path.name,
                "status": "failed",
                "input_row_count": np.nan,
                "matched_city_row_count": np.nan,
                "message": f"{type(exc).__name__}: {exc}",
            })

    columns = [
        "City", "date", "Month", "weekday",
        "grid_lat", "grid_lon", "lat_r", "lon_r",
        PRED_COL, SIGMA_COL, QA_COL,
        "finite_prediction_fields",
        "inside_china_shape", "manual_masked",
        "manual_mask_name", "available_for_plot", "source_file",
    ]

    if records:
        all_records = pd.concat(records, ignore_index=True)
        all_records = all_records[columns].sort_values(
            ["City", "date"]
        ).reset_index(drop=True)
    else:
        all_records = pd.DataFrame(columns=columns)

    return all_records, pd.DataFrame(logs)


def make_qa_statistics(
    all_records: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """输出城市总体和城市-月份 QA 数量统计。"""
    base_cols = [
        "City", "qa_flag", "count", "proportion_percent",
        "retained_QA01", "retained_QA012",
    ]
    month_cols = [
        "City", "Month", "qa_flag", "count", "proportion_percent",
        "retained_QA01", "retained_QA012",
    ]

    if all_records.empty:
        return pd.DataFrame(columns=base_cols), pd.DataFrame(columns=month_cols)

    work = all_records.copy()
    work["qa_flag_label"] = work[QA_COL].map(
        lambda x: str(int(x)) if pd.notna(x) else "NaN"
    )

    def add_flags(table: pd.DataFrame) -> pd.DataFrame:
        numeric = pd.to_numeric(table["qa_flag"], errors="coerce")
        table["retained_QA01"] = numeric.isin([0, 1])
        table["retained_QA012"] = numeric.isin([0, 1, 2])
        return table

    city = (
        work.groupby(["City", "qa_flag_label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"qa_flag_label": "qa_flag"})
    )
    city_total = city.groupby("City")["count"].transform("sum")
    city["proportion_percent"] = np.where(
        city_total > 0,
        100.0 * city["count"] / city_total,
        np.nan,
    )
    city = add_flags(city)[base_cols]

    month = (
        work.groupby(
            ["City", "Month", "qa_flag_label"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"qa_flag_label": "qa_flag"})
    )
    month_total = month.groupby(
        ["City", "Month"]
    )["count"].transform("sum")
    month["proportion_percent"] = np.where(
        month_total > 0,
        100.0 * month["count"] / month_total,
        np.nan,
    )
    month = add_flags(month)[month_cols]

    return city, month


def aggregate_prediction_scheme(
    all_records: pd.DataFrame,
    keep_qa: Sequence[int],
    qa_tag: str,
    city_grids: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按给定 QA 集合输出明细、月均和星期均 CSV 数据。"""
    detail_columns = list(all_records.columns) + ["qa_scheme"]
    summary_columns_month = [
        "City", "Month", "mean", "std", "count",
        "valid_day_count", "sigma_mean", "sigma_std",
        "qa_scheme", "inside_china_shape", "manual_masked",
        "manual_mask_name", "available_for_plot",
    ]
    summary_columns_weekday = [
        "City", "weekday", "mean", "std", "count",
        "valid_day_count", "sigma_mean", "sigma_std",
        "qa_scheme", "inside_china_shape", "manual_masked",
        "manual_mask_name", "available_for_plot",
    ]

    if all_records.empty:
        return (
            pd.DataFrame(columns=detail_columns),
            pd.DataFrame(columns=summary_columns_month),
            pd.DataFrame(columns=summary_columns_weekday),
        )

    retain = (
        all_records[QA_COL].isin(list(keep_qa))
        & all_records["finite_prediction_fields"].astype(bool)
        & all_records["inside_china_shape"].astype(bool)
    )
    detail = all_records.loc[retain].copy()
    detail["qa_scheme"] = qa_tag

    if detail.empty:
        return (
            detail.reindex(columns=detail_columns),
            pd.DataFrame(columns=summary_columns_month),
            pd.DataFrame(columns=summary_columns_weekday),
        )

    def aggregate(group_col: str) -> pd.DataFrame:
        out = (
            detail.groupby(["City", group_col], observed=True)
            .agg(
                mean=(PRED_COL, "mean"),
                std=(PRED_COL, "std"),
                count=(PRED_COL, "count"),
                valid_day_count=("date", "nunique"),
                sigma_mean=(SIGMA_COL, "mean"),
                sigma_std=(SIGMA_COL, "std"),
            )
            .reset_index()
        )
        out["std"] = out["std"].fillna(0.0)
        out["sigma_std"] = out["sigma_std"].fillna(0.0)
        out["qa_scheme"] = qa_tag

        metadata = city_grids[[
            "City", "inside_china_shape", "manual_masked",
            "manual_mask_name", "available_for_plot",
        ]]
        out = out.merge(metadata, on="City", how="left", validate="many_to_one")

        # 与参考脚本一致：手动 MASK 后，聚合均值相关字段设为 NaN，
        # 计数保留，便于追踪 MASK 前曾有多少有效记录。
        mask = out["manual_masked"].fillna(False).astype(bool)
        unavailable = ~out["inside_china_shape"].fillna(False).astype(bool)
        out.loc[
            mask | unavailable,
            ["mean", "std", "sigma_mean", "sigma_std"],
        ] = np.nan
        return out

    monthly = aggregate("Month").reindex(columns=summary_columns_month)
    weekday = aggregate("weekday").reindex(columns=summary_columns_weekday)

    return detail.reindex(columns=detail_columns), monthly, weekday


# =============================================================================
# 8. NO2 数据加载和聚合
# =============================================================================
def find_reference_no2_grid(
    files: Sequence[Path],
) -> Tuple[pd.DataFrame, Path]:
    for file_path in files:
        try:
            df = pd.read_pickle(file_path)
            if df.empty:
                continue
            validate_no2_columns(df, file_path)

            grid_lat = pd.to_numeric(df["grid_lat"], errors="coerce")
            grid_lon = pd.to_numeric(df["grid_lon"], errors="coerce")
            finite = np.isfinite(grid_lat) & np.isfinite(grid_lon)
            grid = pd.DataFrame({
                "grid_lat": grid_lat.loc[finite].to_numpy(),
                "grid_lon": grid_lon.loc[finite].to_numpy(),
            }).drop_duplicates()
            if not grid.empty:
                return grid.reset_index(drop=True), file_path
        except Exception as exc:
            print(
                f"  NO2 reference skip {file_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    raise RuntimeError("没有找到可用于确定城市 NO2 网格的输入文件。")


def select_city_no2_grids(
    files: Sequence[Path],
) -> pd.DataFrame:
    grid, reference_file = find_reference_no2_grid(files)
    rows: List[dict] = []

    for city, info in TARGET_CITIES.items():
        dist2 = (
            (grid["grid_lat"] - float(info["lat"])) ** 2
            + (grid["grid_lon"] - float(info["lon"])) ** 2
        )
        idx = dist2.idxmin()
        selected = grid.loc[idx]
        rows.append({
            "City": city,
            "no2_grid_lat": float(selected["grid_lat"]),
            "no2_grid_lon": float(selected["grid_lon"]),
            "no2_distance_degree": float(np.sqrt(dist2.loc[idx])),
            "no2_reference_grid_file": reference_file.name,
        })

    return pd.DataFrame(rows)


def load_and_aggregate_no2(
    files: Sequence[Path],
    city_prediction_grids: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record_columns = [
        "City", "date", "Month", "weekday",
        "no2_grid_lat", "no2_grid_lon", NO2_COL, "source_file",
    ]
    month_columns = [
        "City", "Month", "mean", "std", "count", "valid_day_count",
        "inside_china_shape", "manual_masked", "manual_mask_name",
        "available_for_plot",
    ]
    weekday_columns = [
        "City", "weekday", "mean", "std", "count", "valid_day_count",
        "inside_china_shape", "manual_masked", "manual_mask_name",
        "available_for_plot",
    ]

    if not files:
        return (
            pd.DataFrame(columns=record_columns),
            pd.DataFrame(columns=month_columns),
            pd.DataFrame(columns=weekday_columns),
            pd.DataFrame(),
        )

    no2_grids = select_city_no2_grids(files)
    records: List[pd.DataFrame] = []

    for index, file_path in enumerate(files, start=1):
        print(
            f"[{index:03d}/{len(files):03d}] "
            f"NO2 {file_path.name} ... ",
            end="",
            flush=True,
        )
        try:
            df = pd.read_pickle(file_path)
            if df.empty:
                print("skipped (empty)")
                continue
            validate_no2_columns(df, file_path)

            work = pd.DataFrame({
                "date": pd.to_datetime(df["date"], errors="coerce"),
                "no2_grid_lat": pd.to_numeric(
                    df["grid_lat"], errors="coerce"
                ),
                "no2_grid_lon": pd.to_numeric(
                    df["grid_lon"], errors="coerce"
                ),
                NO2_COL: pd.to_numeric(df[NO2_COL], errors="coerce"),
            })
            matched = work.merge(
                no2_grids,
                on=["no2_grid_lat", "no2_grid_lon"],
                how="inner",
                validate="many_to_one",
            )
            matched = matched[
                matched["date"].notna()
                & np.isfinite(matched[NO2_COL])
            ].copy()

            if not matched.empty:
                matched["Month"] = matched["date"].dt.month
                matched["weekday"] = matched["date"].dt.dayofweek
                matched["source_file"] = file_path.name
                records.append(matched[record_columns])

            print(f"success, matched={len(matched)}")
        except Exception as exc:
            print(f"failed: {type(exc).__name__}: {exc}")

    if records:
        detail = pd.concat(records, ignore_index=True).sort_values(
            ["City", "date"]
        ).reset_index(drop=True)
    else:
        detail = pd.DataFrame(columns=record_columns)

    metadata = city_prediction_grids[[
        "City", "inside_china_shape", "manual_masked",
        "manual_mask_name", "available_for_plot",
    ]]

    def aggregate(group_col: str, columns: Sequence[str]) -> pd.DataFrame:
        if detail.empty:
            return pd.DataFrame(columns=columns)

        out = (
            detail.groupby(["City", group_col], observed=True)
            .agg(
                mean=(NO2_COL, "mean"),
                std=(NO2_COL, "std"),
                count=(NO2_COL, "count"),
                valid_day_count=("date", "nunique"),
            )
            .reset_index()
        )
        out["std"] = out["std"].fillna(0.0)
        out = out.merge(metadata, on="City", how="left", validate="many_to_one")

        if APPLY_MASK_TO_NO2:
            unavailable = ~out["available_for_plot"].fillna(False).astype(bool)
            out.loc[unavailable, ["mean", "std"]] = np.nan

        return out.reindex(columns=columns)

    monthly = aggregate("Month", month_columns)
    weekday = aggregate("weekday", weekday_columns)

    # 合并预测网格和 NO2 网格元数据，便于复核。
    no2_grid_metadata = city_prediction_grids.merge(
        no2_grids,
        on="City",
        how="left",
        validate="one_to_one",
    )
    return detail, monthly, weekday, no2_grid_metadata


# =============================================================================
# 9. 平滑函数
# =============================================================================
def _poly_smooth(
    x: np.ndarray,
    values: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    degree = min(POLY_DEGREE, len(x) - 1)
    if degree < 1:
        return np.full_like(xs, values[0], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeff = np.polyfit(x, values, degree)
    return np.polyval(coeff, xs)


def _pchip_smooth(
    x: np.ndarray,
    values: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        print(
            "  scipy 不可用，SMOOTH_METHOD='pchip' 自动回退到 poly。"
        )
        return _poly_smooth(x, values, xs)

    return PchipInterpolator(x, values, extrapolate=False)(xs)


def smooth_mean_and_band(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    同时平滑中心均值、下边界 mean-std 和上边界 mean+std。

    返回：xs, smooth_mean, smooth_lower, smooth_upper。
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    std = np.where(np.isfinite(std), np.maximum(std, 0.0), 0.0)
    valid = np.isfinite(x) & np.isfinite(mean)
    x = x[valid]
    mean = mean[valid]
    std = std[valid]

    if len(x) == 0:
        return (
            np.array([]), np.array([]), np.array([]), np.array([])
        )

    order = np.argsort(x)
    x = x[order]
    mean = mean[order]
    std = std[order]

    # 防止重复 x 导致插值/拟合不稳定。
    unique = (
        pd.DataFrame({"x": x, "mean": mean, "std": std})
        .groupby("x", as_index=False)
        .agg(mean=("mean", "mean"), std=("std", "mean"))
    )
    x = unique["x"].to_numpy(dtype=float)
    mean = unique["mean"].to_numpy(dtype=float)
    std = unique["std"].to_numpy(dtype=float)

    if len(x) == 1:
        return x, mean, mean - std, mean + std

    xs = np.linspace(x.min(), x.max(), SMOOTH_POINT_COUNT)
    lower = mean - std
    upper = mean + std

    if SMOOTH_METHOD.lower() == "pchip":
        mean_s = _pchip_smooth(x, mean, xs)
        lower_s = _pchip_smooth(x, lower, xs)
        upper_s = _pchip_smooth(x, upper, xs)
    elif SMOOTH_METHOD.lower() == "poly":
        mean_s = _poly_smooth(x, mean, xs)
        lower_s = _poly_smooth(x, lower, xs)
        upper_s = _poly_smooth(x, upper, xs)
    else:
        raise ValueError(
            "SMOOTH_METHOD 只能是 'poly' 或 'pchip'。"
        )

    # 数值拟合可能局部导致上下界交叉，统一重新排序。
    band_lower = np.minimum(lower_s, upper_s)
    band_upper = np.maximum(lower_s, upper_s)
    return xs, mean_s, band_lower, band_upper


# =============================================================================
# 10. 双轴绘图
# =============================================================================
def _plot_dual_axis(
    ax_l,
    df_xc: pd.DataFrame,
    df_n2: pd.DataFrame,
    x_col: str,
    tick_positions: np.ndarray,
    tick_labels: Sequence[str],
    x_label: str,
    no2_y_label: str,
    xco2_y_label: str,
    title: str,
) -> None:
    """左轴 Delta XCO2，右轴 NO2；两类阴影均平滑。"""
    ax_r = ax_l.twinx()

    # -------------------------------------------------------------------------
    # 左轴：Delta XCO2 圆点 + 平滑均值曲线 + 平滑 std 阴影
    # -------------------------------------------------------------------------
    for city, info in TARGET_CITIES.items():
        color = str(info["color"])
        city_data = (
            df_xc[df_xc["City"] == city]
            .sort_values(x_col)
            .copy()
        )
        if city_data.empty:
            continue

        x = pd.to_numeric(city_data[x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(city_data["mean"], errors="coerce").to_numpy()
        yerr = pd.to_numeric(
            city_data["std"], errors="coerce"
        ).fillna(0.0).to_numpy()

        valid_marker = np.isfinite(x) & np.isfinite(y)
        if not valid_marker.any():
            continue

        ax_l.scatter(
            x[valid_marker],
            y[valid_marker],
            color=color,
            s=45,
            marker="o",
            label=city,
            zorder=5,
        )

        xs, mean_s, lower_s, upper_s = smooth_mean_and_band(
            x, y, yerr
        )
        if len(xs) > 0:
            ax_l.fill_between(
                xs,
                lower_s,
                upper_s,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            ax_l.plot(
                xs,
                mean_s,
                color=color,
                lw=2.5,
                zorder=3,
            )

    ax_l.set_ylabel(xco2_y_label, fontweight="bold")

    # -------------------------------------------------------------------------
    # 右轴：NO2 方块 + 平滑 std 阴影；中心虚线可选
    # -------------------------------------------------------------------------
    if not df_n2.empty:
        for city, info in TARGET_CITIES.items():
            color = str(info["color"])
            city_data = (
                df_n2[df_n2["City"] == city]
                .sort_values(x_col)
                .copy()
            )
            if city_data.empty:
                continue

            x = pd.to_numeric(
                city_data[x_col], errors="coerce"
            ).to_numpy()
            y = pd.to_numeric(
                city_data["mean"], errors="coerce"
            ).to_numpy()
            yerr = pd.to_numeric(
                city_data["std"], errors="coerce"
            ).fillna(0.0).to_numpy()

            valid_marker = np.isfinite(x) & np.isfinite(y)
            if not valid_marker.any():
                continue

            ax_r.scatter(
                x[valid_marker],
                y[valid_marker],
                marker="s",
                s=35,
                color=color,
                alpha=0.60,
                zorder=5,
            )

            xs, mean_s, lower_s, upper_s = smooth_mean_and_band(
                x, y, yerr
            )
            if len(xs) > 0:
                ax_r.fill_between(
                    xs,
                    lower_s,
                    upper_s,
                    color=color,
                    alpha=0.07,
                    linewidth=0,
                    zorder=1,
                )
                if DRAW_NO2_SMOOTH_LINE:
                    ax_r.plot(
                        xs,
                        mean_s,
                        color=color,
                        lw=1.6,
                        linestyle="--",
                        alpha=0.60,
                        zorder=3,
                    )

    ax_r.set_ylabel(no2_y_label, fontweight="bold")

    ax_l.set_xlabel(x_label, fontweight="bold")
    ax_l.set_xticks(tick_positions)
    ax_l.set_xticklabels(
        tick_labels,
        fontsize=GLOBAL_FONT_SIZE - 1,
    )
    ax_l.set_title(title, loc="left", pad=12, fontweight="bold")

    city_legend = ax_l.legend(
        loc="upper left",
        ncol=2,
        fontsize=GLOBAL_FONT_SIZE - 2,
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
    )

    if SHOW_MARKER_TYPE_LEGEND:
        ax_l.add_artist(city_legend)
        marker_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=6,
                label=r"Predicted $\Delta$XCO$_2$",
            ),
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=5.5,
                label=r"Tropospheric NO$_2$ VCD",
            ),
        ]
        ax_r.legend(
            handles=marker_handles,
            loc="upper right",
            fontsize=GLOBAL_FONT_SIZE - 3,
            frameon=True,
            facecolor="white",
            framealpha=0.80,
            edgecolor="none",
        )


# =============================================================================
# 11. CSV 读取与完整处理
# =============================================================================
def read_plot_only_csvs() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required = [
        csv_pred_month(PLOT_QA_TAG),
        csv_pred_weekday(PLOT_QA_TAG),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "PLOT_ONLY=True，但以下正文 QA01 CSV 不存在：\n"
            + missing_text
            + "\n请先使用 PLOT_ONLY=False 完整运行一次。"
        )

    pred_month = pd.read_csv(csv_pred_month(PLOT_QA_TAG))
    pred_weekday = pd.read_csv(csv_pred_weekday(PLOT_QA_TAG))

    if csv_no2_month().exists():
        no2_month = pd.read_csv(csv_no2_month())
    else:
        no2_month = pd.DataFrame()

    if csv_no2_weekday().exists():
        no2_weekday = pd.read_csv(csv_no2_weekday())
    else:
        no2_weekday = pd.DataFrame()

    return pred_month, pred_weekday, no2_month, no2_weekday


def run_full_processing() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    prediction_files = sorted(PKL_DIR.glob(FILE_PATTERN))
    if not prediction_files:
        raise FileNotFoundError(str(PKL_DIR / FILE_PATTERN))

    print("Selecting city prediction grids...")
    city_grids, mask_stats = select_city_prediction_grids(
        prediction_files
    )

    print("Selected prediction grids:")
    for row in city_grids.itertuples(index=False):
        print(
            f"  {row.City}: lat_r={row.lat_r:.2f}, "
            f"lon_r={row.lon_r:.2f}, "
            f"inside_china={row.inside_china_shape}, "
            f"manual_masked={row.manual_masked}"
        )

    print("Loading prediction records with all QA flags...")
    all_prediction_records, processing_log = load_prediction_city_records(
        prediction_files,
        city_grids,
    )

    qa_stats_city, qa_stats_month = make_qa_statistics(
        all_prediction_records
    )

    # 城市网格元数据和筛选统计先输出。
    save_csv(city_grids, CSV_CITY_GRID_SELECTION)
    save_csv(mask_stats, CSV_MASK_STATS)
    save_csv(processing_log, CSV_PROCESSING_LOG)
    save_csv(qa_stats_city, CSV_QA_STATS_CITY, float_format="%.4f")
    save_csv(qa_stats_month, CSV_QA_STATS_MONTH, float_format="%.4f")

    plot_pred_month = pd.DataFrame()
    plot_pred_weekday = pd.DataFrame()

    for qa_tag, keep_qa in EXPORT_QA_SCHEMES.items():
        print(f"Aggregating prediction scheme {qa_tag}: keep={keep_qa}")
        detail, pred_month, pred_weekday = aggregate_prediction_scheme(
            all_prediction_records,
            keep_qa=keep_qa,
            qa_tag=qa_tag,
            city_grids=city_grids,
        )

        save_csv(detail, csv_pred_records(qa_tag))
        save_csv(pred_month, csv_pred_month(qa_tag))
        save_csv(pred_weekday, csv_pred_weekday(qa_tag))

        if qa_tag == PLOT_QA_TAG:
            plot_pred_month = pred_month
            plot_pred_weekday = pred_weekday

    print("Loading NO2 records...")
    no2_files = sorted(INPUT_FEATURES_DIR.glob(NO2_FILE_PATTERN))
    if not no2_files:
        print("  NO2 input not found; empty NO2 CSVs will be written.")

    no2_detail, no2_month, no2_weekday, merged_grid_metadata = (
        load_and_aggregate_no2(
            no2_files,
            city_prediction_grids=city_grids,
        )
    )

    save_csv(no2_detail, csv_no2_records())
    save_csv(no2_month, csv_no2_month())
    save_csv(no2_weekday, csv_no2_weekday())

    # 如 NO2 网格可用，补充输出到城市网格选择表中。
    if not merged_grid_metadata.empty:
        save_csv(merged_grid_metadata, CSV_CITY_GRID_SELECTION)

    return plot_pred_month, plot_pred_weekday, no2_month, no2_weekday


# =============================================================================
# 12. 主程序
# =============================================================================
def main() -> None:
    print("=" * 72)
    print("FIG5 city monthly/weekday QA + MASK + smooth-band pipeline")
    print(f"Plot QA: {PLOT_KEEP_QA} ({PLOT_QA_TAG})")
    print(f"CSV QA schemes: {dict(EXPORT_QA_SCHEMES)}")
    print(f"PLOT_ONLY: {PLOT_ONLY}")
    print(f"SMOOTH_METHOD: {SMOOTH_METHOD}")
    print("=" * 72)

    if tuple(PLOT_KEEP_QA) != tuple(EXPORT_QA_SCHEMES[PLOT_QA_TAG]):
        raise ValueError(
            "PLOT_KEEP_QA 与 EXPORT_QA_SCHEMES['QA01'] 不一致。"
        )

    if PLOT_ONLY:
        print("PLOT_ONLY mode: skip PKL and read existing CSV files.")
        pred_month, pred_weekday, no2_month, no2_weekday = (
            read_plot_only_csvs()
        )
    else:
        print("FULL mode: PKL -> QA/MASK -> all CSV -> plot.")
        pred_month, pred_weekday, no2_month, no2_weekday = (
            run_full_processing()
        )

    print("Plotting monthly dual-axis figure...")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0.5, 12.5)
    _plot_dual_axis(
        ax,
        pred_month,
        no2_month,
        "Month",
        np.arange(1, 13),
        MONTH_LABELS,
        "Month",
        "Tropospheric NO$_2$ VCD",
        "Predicted $\\Delta$XCO$_2$ [ppm]",
        "(a) Monthly Time Series — QA 0/1",
    )
    fig.savefig(OUTPUT_MONTHLY, dpi=300)
    plt.close(fig)
    print(f"  Saved figure: {OUTPUT_MONTHLY}")

    print("Plotting weekday dual-axis figure...")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(-0.5, 6.5)
    _plot_dual_axis(
        ax,
        pred_weekday,
        no2_weekday,
        "weekday",
        np.arange(7),
        WEEKDAY_LABELS,
        "Weekday",
        "Tropospheric NO$_2$ VCD",
        "Predicted $\\Delta$XCO$_2$ [ppm]",
        "(b) Weekday Cycle — QA 0/1",
    )
    fig.savefig(OUTPUT_WEEKDAY, dpi=300)
    plt.close(fig)
    print(f"  Saved figure: {OUTPUT_WEEKDAY}")

    print("Done.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()