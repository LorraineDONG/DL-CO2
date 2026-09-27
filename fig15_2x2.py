# -*- coding: utf-8 -*-
"""
fig15_QA012_timespace.py

基于新的 QA=0/1/2 质量控制体系，绘制 2×2 时空分布图：

(a) 各月 QA=0、QA=1、QA=2 占比的横向堆叠柱状图；
(b) QA=0 在各网格全年有效分类记录中的频率；
(c) QA=1 在各网格全年有效分类记录中的频率；
(d) QA=2 在各网格全年有效分类记录中的频率。

说明
----
1. qa_flag=255 表示无效值，不参与 0/1/2 占比和频率分母。
2. 每个空间格点的三个频率之和为 100%：
       frequency(QA=q) =
           count(QA=q) / count(QA in {0,1,2}) × 100%
3. 输入文件固定在：
       /home/whdong/dl/ML-prediction-output_result/A01/
   并仅查找：
       pred_0.1deg_0.1deg*.pkl
4. 为控制内存，代码逐文件读取并分块聚合，不会一次性拼接全年数据。
5. 四个子图使用相同的 GridSpec 单元和固定位置；地图色标采用一个共享
   colorbar，因此四个子图边框大小一致且严格对齐。
6. 支持 MANUAL_MASK_BOXES 手动掩膜指定经纬度矩形；保留坐标行并将绘图值设为 NaN。
7. 子图(a)图例与三个空间图的共享颜色条并排放在整幅图最下方。
8. 修改 GLOBAL_FONT_SIZE 可同步改变图中全部文字字号。
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import PercentFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from shapely.geometry import shape
from shapely.ops import unary_union

try:
    # Shapely >= 2.0：快速判断坐标点是否与中国面相交，边界点也会保留。
    from shapely import intersects_xy
except ImportError:
    intersects_xy = None


# =============================================================================
# 1. 配置
# =============================================================================
PKL_DIR = Path("/home/whdong/dl/ML-prediction-output_result/A01")
FILE_PATTERN = "pred_0.1deg_0.1deg*.pkl"

OUTPUT_DIR = PKL_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FIGURE = OUTPUT_DIR / "FIG15-QA012_Temporal_Spatial.png"
OUTPUT_MONTHLY_TABLE = OUTPUT_DIR / "FIG15-QA_Monthly_Proportions.csv"
OUTPUT_SPATIAL_TABLE = OUTPUT_DIR / "FIG15-QA_Spatial_Frequencies.csv"
OUTPUT_MASK_STATS_TABLE = OUTPUT_DIR / "FIG15-QA_Manual_Mask_Statistics.csv"

WORLD_SHP = Path("/home/whdong/shapefile/world/world.shp")
CHINA_PROV_SHP = Path("/home/whdong/shapefile/china/province.shp")

QA_COL = "qa_flag"
QA_LEVELS = (0, 1, 2)

# 是否仅保留网格中心位于中国省级行政区面内的格点。
APPLY_CHINA_SHAPE_CLIP = True

# 手动空间掩膜总开关。
# True：删除下面 enabled=True 的矩形范围内的格点；
# False：完全不应用手动矩形掩膜。
APPLY_MANUAL_MASK = True

# 手动矩形掩膜。
# None 表示该方向不限制；边界采用闭区间。
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

GRID_RESOLUTION = 0.1
MAP_EXTENT = [70.0, 140.0, 15.0, 55.0]

# 三幅空间频率图使用完全相同的色标范围，便于直接比较。
MAP_VMIN = 0.0
MAP_VMAX = 65.0
MAP_CMAP = "RdYlGn_r"

# 横向堆叠柱颜色。
QA_COLORS = {
    0: "#A6A6A6",
    1: "#8064A2",  
    2: "#2F5597",  
}

QA_NAMES = {
    0: "Recommended",
    1: "Use with caution",
    2: "Not recommended",
}

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# 每累计多少个文件，将空间计数缓冲区压缩一次。
AGGREGATION_CHUNK_SIZE = 31


# -----------------------------------------------------------------------------
# 全局文字字号总开关
# -----------------------------------------------------------------------------
# 修改这一个数值，即可同步改变图中所有文字元素的字号，包括：
#   1) 横纵轴刻度；
#   2) 横纵轴标签；
#   3) 图例；
#   4) 颜色条标签和颜色条刻度数值；
#   5) 子图编号与 QA 标题；
#   6) 柱状图内部百分比数字。
GLOBAL_FONT_SIZE = 12.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": GLOBAL_FONT_SIZE,
    "axes.titlesize": GLOBAL_FONT_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": GLOBAL_FONT_SIZE,
    "xtick.labelsize": GLOBAL_FONT_SIZE,
    "ytick.labelsize": GLOBAL_FONT_SIZE,
    "legend.fontsize": GLOBAL_FONT_SIZE,
    "axes.linewidth": 1.4,
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
# 2. 数据读取和流式聚合
# =============================================================================
def validate_columns(df: pd.DataFrame, file_path: Path) -> None:
    required = {"grid_lat", "grid_lon", "date", QA_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{file_path.name} 缺少字段: {missing}")


def round_grid_centers(
    lat: pd.Series,
    lon: pd.Series,
    resolution: float,
) -> Tuple[pd.Series, pd.Series]:
    """将经纬度映射到固定分辨率网格中心。"""
    decimals = max(2, int(np.ceil(-np.log10(resolution))) + 2)

    lat_r = (
        np.floor(pd.to_numeric(lat, errors="coerce") / resolution)
        * resolution
        + resolution / 2.0
    ).round(decimals)

    lon_r = (
        np.floor(pd.to_numeric(lon, errors="coerce") / resolution)
        * resolution
        + resolution / 2.0
    ).round(decimals)

    return lat_r, lon_r


def compress_spatial_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """将多个文件的空间 QA 计数压缩为一个较小的累计表。"""
    valid = [frame for frame in frames if frame is not None and not frame.empty]

    if not valid:
        return pd.DataFrame(
            columns=["lat_r", "lon_r", QA_COL, "count"]
        )

    combined = pd.concat(valid, ignore_index=True)

    return (
        combined.groupby(
            ["lat_r", "lon_r", QA_COL],
            sort=False,
            observed=True,
        )["count"]
        .sum()
        .reset_index()
    )


def merge_spatial_accumulator(
    accumulator: pd.DataFrame | None,
    chunk: pd.DataFrame,
) -> pd.DataFrame:
    """将新的分块计数合并到全年累计计数中。"""
    if chunk.empty:
        if accumulator is None:
            return chunk
        return accumulator

    if accumulator is None or accumulator.empty:
        return chunk

    return compress_spatial_frames([accumulator, chunk])


def aggregate_all_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回：
        monthly_counts:
            12个月 × QA=0/1/2 的记录数和占比；
        spatial_frequency:
            每个0.1°网格 QA=0/1/2 的全年频率。
    """
    files = sorted(PKL_DIR.glob(FILE_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"未找到输入文件：\n"
            f"目录：{PKL_DIR}\n"
            f"模式：{FILE_PATTERN}"
        )

    monthly_count_array = np.zeros((12, 3), dtype=np.int64)

    buffer: List[pd.DataFrame] = []
    spatial_accumulator: pd.DataFrame | None = None

    successful_files = 0
    duplicate_grid_day_records = 0

    for index, file_path in enumerate(files, start=1):
        print(
            f"[{index:03d}/{len(files):03d}] {file_path.name} ... ",
            end="",
            flush=True,
        )

        try:
            df = pd.read_pickle(file_path)
            if df.empty:
                print("skipped (empty)")
                continue

            validate_columns(df, file_path)

            dates = pd.to_datetime(df["date"], errors="coerce")
            qa = pd.to_numeric(df[QA_COL], errors="coerce")
            lat_r, lon_r = round_grid_centers(
                df["grid_lat"],
                df["grid_lon"],
                GRID_RESOLUTION,
            )

            work = pd.DataFrame({
                "date": dates,
                "month": dates.dt.month,
                "lat_r": lat_r,
                "lon_r": lon_r,
                QA_COL: qa,
            })

            work = work[
                work[QA_COL].isin(QA_LEVELS)
                & work["month"].between(1, 12)
                & np.isfinite(work["lat_r"])
                & np.isfinite(work["lon_r"])
            ].copy()

            if work.empty:
                print("skipped (no QA 0/1/2)")
                continue

            work[QA_COL] = work[QA_COL].astype(np.int8)
            work["month"] = work["month"].astype(np.int8)

            # 每个日期—网格只计数一次，使空间图表示 QA 天数频率。
            duplicate_mask = work.duplicated(
                subset=["date", "lat_r", "lon_r"],
                keep="first",
            )
            duplicate_grid_day_records += int(duplicate_mask.sum())
            if duplicate_mask.any():
                work = work.loc[~duplicate_mask].copy()

            # 月度 QA 计数
            file_monthly = (
                work.groupby(
                    ["month", QA_COL],
                    observed=True,
                )
                .size()
            )
            for (month, qa_flag), count in file_monthly.items():
                monthly_count_array[int(month) - 1, int(qa_flag)] += int(count)

            # 空间 QA 计数
            file_spatial = (
                work.groupby(
                    ["lat_r", "lon_r", QA_COL],
                    sort=False,
                    observed=True,
                )
                .size()
                .reset_index(name="count")
            )
            buffer.append(file_spatial)
            successful_files += 1

            if len(buffer) >= AGGREGATION_CHUNK_SIZE:
                chunk = compress_spatial_frames(buffer)
                spatial_accumulator = merge_spatial_accumulator(
                    spatial_accumulator,
                    chunk,
                )
                buffer = []

            print("success")

        except Exception as exc:
            print(f"failed: {type(exc).__name__}: {exc}")

    if buffer:
        chunk = compress_spatial_frames(buffer)
        spatial_accumulator = merge_spatial_accumulator(
            spatial_accumulator,
            chunk,
        )

    if successful_files == 0:
        raise RuntimeError("没有成功处理任何输入文件。")

    if spatial_accumulator is None or spatial_accumulator.empty:
        raise RuntimeError("空间 QA 计数为空。")

    # -------------------------
    # 月度计数和占比
    # -------------------------
    monthly_counts = pd.DataFrame(
        monthly_count_array,
        columns=[f"count_qa{qa}" for qa in QA_LEVELS],
    )
    monthly_counts.insert(0, "month_name", MONTH_LABELS)
    monthly_counts.insert(0, "month", np.arange(1, 13))

    monthly_total = monthly_count_array.sum(axis=1)

    for qa in QA_LEVELS:
        monthly_counts[f"percent_qa{qa}"] = np.divide(
            monthly_count_array[:, qa] * 100.0,
            monthly_total,
            out=np.full(12, np.nan, dtype=float),
            where=monthly_total > 0,
        )

    monthly_counts["total_valid_qa_records"] = monthly_total

    # -------------------------
    # 空间计数和频率
    # -------------------------
    spatial_counts = (
        spatial_accumulator.pivot_table(
            index=["lat_r", "lon_r"],
            columns=QA_COL,
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    spatial_counts.columns.name = None

    for qa in QA_LEVELS:
        if qa not in spatial_counts.columns:
            spatial_counts[qa] = 0

    spatial_counts = spatial_counts.rename(
        columns={qa: f"count_qa{qa}" for qa in QA_LEVELS}
    )

    spatial_total = sum(
        spatial_counts[f"count_qa{qa}"]
        for qa in QA_LEVELS
    )
    spatial_counts["total_valid_qa_days"] = spatial_total

    for qa in QA_LEVELS:
        spatial_counts[f"percent_qa{qa}"] = np.divide(
            spatial_counts[f"count_qa{qa}"] * 100.0,
            spatial_total,
            out=np.full(len(spatial_counts), np.nan, dtype=float),
            where=spatial_total.to_numpy() > 0,
        )

    print(
        f"\n成功处理文件数：{successful_files}/{len(files)}"
    )
    if duplicate_grid_day_records:
        print(
            "检测并剔除重复的日期—网格记录："
            f"{duplicate_grid_day_records:,}"
        )

    return monthly_counts, spatial_counts


# =============================================================================
# 3. 中国范围裁剪
# =============================================================================
def clip_spatial_table_to_china(
    spatial_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    仅保留网格中心位于 CHINA_PROV_SHP 面要素并集内的空间记录。
    """
    if not APPLY_CHINA_SHAPE_CLIP:
        return spatial_df.copy()

    if not CHINA_PROV_SHP.exists():
        print(
            f"警告：中国省界文件不存在，跳过中国范围裁剪："
            f"{CHINA_PROV_SHP}"
        )
        return spatial_df.copy()

    import shapefile

    sf = shapefile.Reader(
        str(CHINA_PROV_SHP),
        encoding="gbk",
    )
    geometries = [
        shape(item.__geo_interface__)
        for item in sf.shapes()
    ]
    china_geometry = unary_union(geometries)

    if not china_geometry.is_valid:
        repaired = china_geometry.buffer(0)
        if not repaired.is_empty:
            china_geometry = repaired

    lon = spatial_df["lon_r"].to_numpy(dtype=float)
    lat = spatial_df["lat_r"].to_numpy(dtype=float)

    if intersects_xy is not None:
        inside = np.asarray(
            intersects_xy(china_geometry, lon, lat),
            dtype=bool,
        )
    else:
        # Shapely 1.x 回退方式
        from shapely.geometry import Point
        from shapely.prepared import prep

        prepared = prep(china_geometry)
        inside = np.fromiter(
            (
                prepared.covers(Point(x, y))
                for x, y in zip(lon, lat)
            ),
            dtype=bool,
            count=len(spatial_df),
        )

    clipped = spatial_df.loc[inside].copy()

    print(
        "中国省级行政区面裁剪："
        f"{len(spatial_df):,} -> {len(clipped):,} 个网格"
    )

    return clipped


def validate_manual_mask_boxes(
    mask_boxes: Sequence[Mapping[str, object]],
) -> None:
    """检查手动矩形掩膜配置是否合法。"""
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
                        f"手动掩膜 {name!r} 的 {field_name} "
                        f"不是有效数字或 None: {value!r}"
                    ) from exc

        if lon_min is not None and lon_max is not None:
            if float(lon_min) > float(lon_max):
                raise ValueError(
                    f"手动掩膜 {name!r}: lon_min 不能大于 lon_max。"
                )

        if lat_min is not None and lat_max is not None:
            if float(lat_min) > float(lat_max):
                raise ValueError(
                    f"手动掩膜 {name!r}: lat_min 不能大于 lat_max。"
                )

        if all(
            value is None
            for value in [lon_min, lon_max, lat_min, lat_max]
        ):
            raise ValueError(
                f"手动掩膜 {name!r} 没有设置任何经纬度边界，"
                "这会掩膜全部格点，请至少设置一个边界。"
            )


def apply_manual_masks_to_spatial_table(
    spatial_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对聚合后的 QA 空间频率表应用多个经纬度矩形掩膜。

    关键处理方式与 fig3and5_flagABCD_mask.py 保持一致：
      - 保留被掩膜格点对应的 lon_r / lat_r 坐标行；
      - 将 percent_qa0 / percent_qa1 / percent_qa2 设置为 NaN；
      - 记录 manual_masked 和 manual_mask_name；
      - 保留原始 count_qa* 与 total_valid_qa_days，便于追踪掩膜前数据量。

    不能直接删除整行。若删除的是完整经度带，该经度坐标会从 pivot
    结果中消失，pcolormesh(shading="nearest") 会让相邻格点跨越空缺，
    从视觉上看起来像没有应用掩膜。

    返回：
      masked_spatial_df
      manual_mask_stats
    """
    out = spatial_df.copy()

    value_columns = [f"percent_qa{qa}" for qa in QA_LEVELS]
    required = {"lat_r", "lon_r", *value_columns}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise KeyError(
            f"空间频率表缺少手动掩膜所需字段: {missing}"
        )

    out["manual_masked"] = False
    out["manual_mask_name"] = ""

    if not APPLY_MANUAL_MASK:
        stats = pd.DataFrame([{
            "group_type": "spatial_frequency",
            "group_name": "QA=0/1/2",
            "mask_name": "ALL_MASKS_DISABLED",
            "lon_min": np.nan,
            "lon_max": np.nan,
            "lat_min": np.nan,
            "lat_max": np.nan,
            "masked_grid_count": 0,
        }])
        return out, stats

    enabled_boxes = [
        box
        for box in MANUAL_MASK_BOXES
        if bool(box.get("enabled", True))
    ]

    if not enabled_boxes:
        stats = pd.DataFrame([{
            "group_type": "spatial_frequency",
            "group_name": "QA=0/1/2",
            "mask_name": "NO_ENABLED_MASK_BOX",
            "lon_min": np.nan,
            "lon_max": np.nan,
            "lat_min": np.nan,
            "lat_max": np.nan,
            "masked_grid_count": 0,
        }])
        return out, stats

    validate_manual_mask_boxes(enabled_boxes)

    lon = pd.to_numeric(out["lon_r"], errors="coerce").to_numpy()
    lat = pd.to_numeric(out["lat_r"], errors="coerce").to_numpy()

    total_mask = np.zeros(len(out), dtype=bool)
    mask_names: List[List[str]] = [[] for _ in range(len(out))]
    stats_rows: List[dict] = []

    for index, box in enumerate(enabled_boxes, start=1):
        name = str(box.get("name", f"mask_{index}"))
        lon_min = box.get("lon_min")
        lon_max = box.get("lon_max")
        lat_min = box.get("lat_min")
        lat_max = box.get("lat_max")

        box_mask = np.isfinite(lon) & np.isfinite(lat)

        if lon_min is not None:
            box_mask &= lon >= float(lon_min)
        if lon_max is not None:
            box_mask &= lon <= float(lon_max)
        if lat_min is not None:
            box_mask &= lat >= float(lat_min)
        if lat_max is not None:
            box_mask &= lat <= float(lat_max)

        hit_indices = np.flatnonzero(box_mask)
        for row_index in hit_indices:
            mask_names[row_index].append(name)

        total_mask |= box_mask

        masked_count = int(box_mask.sum())
        stats_rows.append({
            "group_type": "spatial_frequency",
            "group_name": "QA=0/1/2",
            "mask_name": name,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "masked_grid_count": masked_count,
        })

        print(
            f"手动掩膜 [{name}]："
            f"{masked_count:,} 个空间格点"
        )

    # 与参考脚本相同：不删除坐标行，只将真正用于绘图的值设为 NaN。
    out.loc[total_mask, value_columns] = np.nan
    out["manual_masked"] = total_mask
    out["manual_mask_name"] = [
        ";".join(names)
        for names in mask_names
    ]

    stats_rows.append({
        "group_type": "spatial_frequency",
        "group_name": "QA=0/1/2",
        "mask_name": "UNION_OF_ENABLED_MASKS",
        "lon_min": np.nan,
        "lon_max": np.nan,
        "lat_min": np.nan,
        "lat_max": np.nan,
        "masked_grid_count": int(total_mask.sum()),
    })

    print(
        "手动空间掩膜合计："
        f"{int(total_mask.sum()):,}/{len(out):,} 个网格已设为 NaN，"
        "坐标行保留。"
    )

    return out, pd.DataFrame(stats_rows)


# =============================================================================
# 4. 绘图函数
# =============================================================================
def plot_monthly_horizontal_bars(
    ax,
    monthly_counts: pd.DataFrame,
) -> None:
    """
    子图(a)：12个月 QA=0/1/2 占比的横向堆叠柱状图。
    """
    y = np.arange(12)
    left = np.zeros(12, dtype=float)

    for qa in QA_LEVELS:
        values = monthly_counts[f"percent_qa{qa}"].fillna(0).to_numpy()

        ax.barh(
            y,
            values,
            left=left,
            height=0.68,
            color=QA_COLORS[qa],
            edgecolor="white",
            linewidth=0.6,
            label=(
                f"QA={qa}"
                f"({monthly_counts[f'count_qa{qa}'].sum():,})"
            ),
            zorder=3,
        )

        # 仅标注足够宽的色块，避免文字拥挤。
        for row_index, (segment_left, value) in enumerate(
            zip(left, values)
        ):
            if value >= 5.0:
                ax.text(
                    segment_left + value / 2.0,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=GLOBAL_FONT_SIZE,
                    color="black",
                    zorder=4,
                )

        left += values

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 11.6)
    ax.set_yticks(y)
    ax.set_yticklabels(MONTH_LABELS)
    ax.invert_yaxis()

    ax.set_xlabel("Monthly proportion (%)")
    ax.set_ylabel("Month")
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.tick_params(axis="both", labelsize=GLOBAL_FONT_SIZE)

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
        color="gray",
        zorder=0,
    )
    ax.grid(axis="y", visible=False)

    ax.text(
        0.025,
        0.965,
        "(a) ",
        transform=ax.transAxes,
        fontsize=GLOBAL_FONT_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "pad": 2,
        },
        zorder=6,
    )

    # 公共图例在 main() 中放到整幅图最下方。


def add_map_features(
    ax,
    show_left_labels: bool,
    show_bottom_labels: bool,
) -> None:
    ax.set_extent(
        MAP_EXTENT,
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.8,
        edgecolor="#333333",
        zorder=3,
    )

    if WORLD_SHP.exists():
        world_border = cfeature.ShapelyFeature(
            shpreader.Reader(str(WORLD_SHP)).geometries(),
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#333333",
            linewidth=0.6,
            linestyle=":",
        )
        ax.add_feature(world_border, zorder=3)
    else:
        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.6,
            linestyle=":",
            edgecolor="#333333",
            zorder=3,
        )

    if CHINA_PROV_SHP.exists():
        import shapefile

        sf = shapefile.Reader(
            str(CHINA_PROV_SHP),
            encoding="gbk",
        )
        province_border = cfeature.ShapelyFeature(
            [
                shape(item.__geo_interface__)
                for item in sf.shapes()
            ],
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#777777",
            linewidth=0.35,
            linestyle=":",
        )
        ax.add_feature(province_border, zorder=3)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.4,
        linestyle="--",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.bottom_labels = show_bottom_labels

    gl.xlocator = mticker.FixedLocator([75, 95, 115, 135])
    gl.ylocator = mticker.FixedLocator([20, 30, 40, 50])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": GLOBAL_FONT_SIZE}
    gl.ylabel_style = {"size": GLOBAL_FONT_SIZE}


def plot_qa_frequency_map(
    ax,
    spatial_df: pd.DataFrame,
    qa_flag: int,
    panel_label: str,
    show_left_labels: bool,
    show_bottom_labels: bool,
):
    """
    子图(b)–(d)：指定 QA 等级在各格点全年有效分类日期中的频率。
    """
    value_col = f"percent_qa{qa_flag}"

    matrix = (
        spatial_df.pivot(
            index="lat_r",
            columns="lon_r",
            values=value_col,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    values = np.ma.masked_invalid(
        matrix.to_numpy(dtype=float)
    )

    pcm = ax.pcolormesh(
        matrix.columns.to_numpy(),
        matrix.index.to_numpy(),
        values,
        cmap=MAP_CMAP,
        vmin=MAP_VMIN,
        vmax=MAP_VMAX,
        shading="nearest",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    add_map_features(
        ax,
        show_left_labels=show_left_labels,
        show_bottom_labels=show_bottom_labels,
    )

    ax.text(
        0.025,
        0.965,
        (
            f"{panel_label} QA={qa_flag} "
        ),
        transform=ax.transAxes,
        fontsize=GLOBAL_FONT_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "pad": 2,
        },
        zorder=6,
    )

    return pcm


def force_equal_aligned_panel_positions(
    fig,
    grid_spec,
    axes,
) -> None:
    """
    强制四个子图使用四个 GridSpec 单元的完整矩形位置。

    Cartopy 默认会根据地图纵横比调整 GeoAxes 的实际框体。
    此处将三个地图轴的 aspect 设置为 auto，并重新赋予标准
    GridSpec 位置，从而确保四个边框宽高完全一致且横纵对齐。

    本图的地图范围宽高比约为 70°:40°，同时 figure 尺寸已按近似
    相同比例设置，因此视觉形变很小。
    """
    target_positions = [
        grid_spec[0, 0].get_position(fig),
        grid_spec[0, 1].get_position(fig),
        grid_spec[1, 0].get_position(fig),
        grid_spec[1, 1].get_position(fig),
    ]

    for ax, position in zip(axes, target_positions):
        ax.set_aspect("auto")
        ax.set_position(position)


# =============================================================================
# 5. 主程序
# =============================================================================
def main() -> None:
    monthly_counts, spatial_frequency = aggregate_all_files()

    spatial_frequency = clip_spatial_table_to_china(
        spatial_frequency
    )
    spatial_frequency, manual_mask_stats = apply_manual_masks_to_spatial_table(
        spatial_frequency
    )

    monthly_counts.to_csv(
        OUTPUT_MONTHLY_TABLE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    spatial_frequency.to_csv(
        OUTPUT_SPATIAL_TABLE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    manual_mask_stats.to_csv(
        OUTPUT_MASK_STATS_TABLE,
        index=False,
        encoding="utf-8-sig",
    )

    fig = plt.figure(figsize=(16.0, 10.2))

    grid_spec = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.985,
        bottom=0.165,
        top=0.975,
        wspace=0.07,
        hspace=0.125,
    )

    ax_a = fig.add_subplot(grid_spec[0, 0])
    ax_b = fig.add_subplot(
        grid_spec[0, 1],
        projection=ccrs.PlateCarree(),
    )
    ax_c = fig.add_subplot(
        grid_spec[1, 0],
        projection=ccrs.PlateCarree(),
    )
    ax_d = fig.add_subplot(
        grid_spec[1, 1],
        projection=ccrs.PlateCarree(),
    )

    plot_monthly_horizontal_bars(
        ax_a,
        monthly_counts,
    )

    plot_qa_frequency_map(
        ax_b,
        spatial_frequency,
        qa_flag=0,
        panel_label="(b)",
        show_left_labels=True,
        show_bottom_labels=True,
    )
    plot_qa_frequency_map(
        ax_c,
        spatial_frequency,
        qa_flag=1,
        panel_label="(c)",
        show_left_labels=True,
        show_bottom_labels=True,
    )
    plot_qa_frequency_map(
        ax_d,
        spatial_frequency,
        qa_flag=2,
        panel_label="(d)",
        show_left_labels=True,
        show_bottom_labels=True,
    )

    # 四个边框强制采用相同宽度、高度和对齐位置。
    force_equal_aligned_panel_positions(
        fig,
        grid_spec,
        [ax_a, ax_b, ax_c, ax_d],
    )

    # 子图(a)的图例作为整幅图的公共图例，放在底部左侧。
    legend_handles, legend_labels = ax_a.get_legend_handles_labels()
    figure_legend = fig.legend(
    legend_handles,
    legend_labels,
    loc="lower left",
    bbox_to_anchor=(0.105, 0.105),
    ncol=3,
    frameon=False,
    framealpha=0.90,
    edgecolor="silver",
    fontsize=GLOBAL_FONT_SIZE,
    borderaxespad=0.0,
    handlelength=1.8,
    handletextpad=0.5,
    columnspacing=1.2,
)

    # 三幅地图共用一个色标，放在底部右侧，与图例处于同一水平区域。
    # 独立色标轴不会压缩任何地图子图。
    colorbar_ax = fig.add_axes(
        [0.505, 0.105, 0.43, 0.022]
    )
    scalar_mappable = ScalarMappable(
        norm=Normalize(
            vmin=MAP_VMIN,
            vmax=MAP_VMAX,
        ),
        cmap=MAP_CMAP,
    )
    scalar_mappable.set_array([])

    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_ax,
        orientation="horizontal",
        extend="neither",
    )
    colorbar.set_label(
        "Annual QA-Flag Frequency (%)",
        fontsize=GLOBAL_FONT_SIZE,
        fontweight="bold",
    )
    colorbar.ax.tick_params(
        direction="in",
        length=4,
        labelsize=GLOBAL_FONT_SIZE,
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\n图像已保存：{OUTPUT_FIGURE}")
    print(f"月度统计表：{OUTPUT_MONTHLY_TABLE}")
    print(f"空间统计表：{OUTPUT_SPATIAL_TABLE}")
    print(f"手动掩膜统计表：{OUTPUT_MASK_STATS_TABLE}")

    enabled_mask_names = [
        str(box.get("name", "unnamed_mask"))
        for box in MANUAL_MASK_BOXES
        if bool(box.get("enabled", True))
    ]
    if APPLY_MANUAL_MASK and enabled_mask_names:
        print(
            "启用的手动空间掩膜："
            + ", ".join(enabled_mask_names)
        )
    elif APPLY_MANUAL_MASK:
        print("手动掩膜开关已启用，但没有启用的矩形。")
    else:
        print("手动空间掩膜已关闭。")

    total = sum(
        monthly_counts[f"count_qa{qa}"].sum()
        for qa in QA_LEVELS
    )

    print(f"\nQA=0/1/2 总有效记录数：{total:,}")
    for qa in QA_LEVELS:
        count = int(
            monthly_counts[f"count_qa{qa}"].sum()
        )
        proportion = (
            count / total * 100.0
            if total > 0
            else np.nan
        )
        print(
            f"  QA={qa}: {count:>12,} "
            f"({proportion:6.2f}%)"
        )


if __name__ == "__main__":
    main()