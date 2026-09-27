# -*- coding: utf-8 -*-
"""
fig5_city_4panel_compare_from_csv.py

Use existing intermediate CSV files only (no PKL reading) to draw two 2x2 figures:
  1) monthly cycle (Jan-Dec)
  2) weekday cycle (Mon-Sun)

Each subplot corresponds to one city:
    Beijing, Shanghai, Wuhan, Guangzhou

Within each subplot:
    - Left y-axis: QA01 DeltaXCO2, QA012 DeltaXCO2
    - Right y-axis: NO2 VCD

All three series keep markers and use explicit cubic-polynomial fitted lines and uncertainty bands.
This script is intended for re-plotting only after the CSV files have already been
created by the previous pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. Configuration
# =============================================================================
ACTIVE_VERSION = "A01"
BASE_DIR = Path(f"/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}")

# The preceding QA/MASK processing script writes its CSV files to
# "figures_case_study_qa_mask". The older plotting script used
# "figures_case_study". This script detects the correct directory automatically.
INPUT_DIR_CANDIDATES = [
    BASE_DIR / "figures_case_study_qa_mask",
    BASE_DIR / "figures_case_study",
]

REQUIRED_CSV_NAMES = [
    "FIG5_intermediate_pred_month_QA01.csv",
    "FIG5_intermediate_pred_weekday_QA01.csv",
    "FIG5_intermediate_pred_month_QA012.csv",
    "FIG5_intermediate_pred_weekday_QA012.csv",
    "FIG5_intermediate_no2_month.csv",
    "FIG5_intermediate_no2_weekday.csv",
]


def detect_input_dir() -> Path:
    """Find a directory containing all six required CSV files."""
    checked = []
    for candidate in INPUT_DIR_CANDIDATES:
        missing = [name for name in REQUIRED_CSV_NAMES if not (candidate / name).exists()]
        checked.append((candidate, missing))
        if not missing:
            return candidate

    # Additional fallback: search direct subdirectories of the version directory.
    for candidate in sorted(BASE_DIR.glob("figures*")):
        if not candidate.is_dir() or candidate in INPUT_DIR_CANDIDATES:
            continue
        missing = [name for name in REQUIRED_CSV_NAMES if not (candidate / name).exists()]
        checked.append((candidate, missing))
        if not missing:
            return candidate

    details = ["No directory contains all required CSV files."]
    for candidate, missing in checked:
        details.append(f"  Checked: {candidate}")
        details.append("    Missing: " + ", ".join(missing))
    raise FileNotFoundError("\n".join(details))


INPUT_DIR = detect_input_dir()
OUTPUT_DIR = INPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing CSV files from the previous script
CSV_QA01_MONTH = INPUT_DIR / "FIG5_intermediate_pred_month_QA01.csv"
CSV_QA01_WEEKDAY = INPUT_DIR / "FIG5_intermediate_pred_weekday_QA01.csv"
CSV_QA012_MONTH = INPUT_DIR / "FIG5_intermediate_pred_month_QA012.csv"
CSV_QA012_WEEKDAY = INPUT_DIR / "FIG5_intermediate_pred_weekday_QA012.csv"
CSV_NO2_MONTH = INPUT_DIR / "FIG5_intermediate_no2_month.csv"
CSV_NO2_WEEKDAY = INPUT_DIR / "FIG5_intermediate_no2_weekday.csv"

# Output figures
OUTPUT_MONTHLY = OUTPUT_DIR / "FIG5_City_Monthly_4panel_QA01_QA012_NO2_poly3.png"
OUTPUT_WEEKDAY = OUTPUT_DIR / "FIG5_City_Weekday_4panel_QA01_QA012_NO2_poly3.png"

TARGET_CITIES = {
    "Beijing":   {"color": "#9B59B6"},
    "Shanghai":  {"color": "#34495E"},
    "Wuhan":     {"color": "#E67E22"},
    "Guangzhou": {"color": "#A2B836"},
}
CITY_ORDER = ["Beijing", "Shanghai", "Wuhan", "Guangzhou"]

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
WEEKDAY_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# Explicit polynomial fitting configuration.
# With 12 monthly points and 7 weekday points, cubic fitting is used by default.
POLY_DEGREE = 3
N_DENSE = 300

GLOBAL_FONT_SIZE = 12
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": GLOBAL_FONT_SIZE,
    "axes.titlesize": GLOBAL_FONT_SIZE + 1,
    "axes.titleweight": "bold",
    "axes.labelsize": GLOBAL_FONT_SIZE,
    "xtick.labelsize": GLOBAL_FONT_SIZE - 1,
    "ytick.labelsize": GLOBAL_FONT_SIZE - 1,
    "legend.fontsize": GLOBAL_FONT_SIZE - 1,
    "axes.linewidth": 1.4,
    "lines.linewidth": 1.9,
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

# Style for three series
STYLE_QA01 = {
    "marker": "o",
    "linestyle": "-",
    "linewidth": 2.2,
    "markersize": 5.8,
    "band_alpha": 0.14,
    "label": "QA(0,1) ΔXCO$_2$",
}
STYLE_QA012 = {
    "marker": "D",
    "linestyle": "--",
    "linewidth": 2.0,
    "markersize": 5.4,
    "band_alpha": 0.11,
    "label": "QA(0,1,2) ΔXCO$_2$",
}
STYLE_NO2 = {
    "marker": "s",
    "linestyle": ":",
    "linewidth": 2.0,
    "markersize": 5.2,
    "band_alpha": 0.09,
    "label": "NO$_2$ VCD",
}

# =============================================================================
# 2. IO helpers
# =============================================================================
def _require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def load_all_csvs():
    qa01_month = _require_csv(CSV_QA01_MONTH)
    qa01_week = _require_csv(CSV_QA01_WEEKDAY)
    qa012_month = _require_csv(CSV_QA012_MONTH)
    qa012_week = _require_csv(CSV_QA012_WEEKDAY)
    no2_month = _require_csv(CSV_NO2_MONTH)
    no2_week = _require_csv(CSV_NO2_WEEKDAY)
    return qa01_month, qa01_week, qa012_month, qa012_week, no2_month, no2_week


# =============================================================================
# 3. Smoothing helpers
# =============================================================================
def _prepare_xy(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean, sort, and collapse duplicate x positions before fitting."""
    work = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x), errors="coerce"),
        "y": pd.to_numeric(pd.Series(y), errors="coerce"),
    }).dropna()

    if work.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Duplicate x values are averaged so np.polyfit receives one value per period.
    work = (
        work.groupby("x", as_index=False, sort=True)["y"]
        .mean()
        .sort_values("x")
    )
    return (
        work["x"].to_numpy(dtype=float),
        work["y"].to_numpy(dtype=float),
    )


def _polyfit_curve(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = POLY_DEGREE,
    n_dense: int = N_DENSE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit an explicit polynomial curve and evaluate it on a dense x grid.

    The requested degree is 3. When fewer than four valid x positions are
    available, the degree is reduced automatically to n_points - 1.
    """
    x_clean, y_clean = _prepare_xy(x, y)

    if len(x_clean) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if len(x_clean) == 1:
        return x_clean.copy(), y_clean.copy()

    fit_degree = min(int(degree), len(x_clean) - 1)
    xs = np.linspace(x_clean.min(), x_clean.max(), int(n_dense))

    try:
        coefficients = np.polyfit(x_clean, y_clean, fit_degree)
        ys = np.polyval(coefficients, xs)
        return xs, ys
    except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError):
        # A raw line is retained only as a defensive fallback if fitting fails.
        return x_clean, y_clean


def _polyfit_band(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    degree: int = POLY_DEGREE,
    n_dense: int = N_DENSE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a smooth mean +/- std band using explicit polynomial fitting.

    The mean and standard deviation are fitted separately. Any negative fitted
    standard deviation caused by polynomial overshoot is clipped to zero.
    """
    raw = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x), errors="coerce"),
        "mean": pd.to_numeric(pd.Series(y), errors="coerce"),
        "std": pd.to_numeric(pd.Series(yerr), errors="coerce"),
    })
    raw["std"] = raw["std"].fillna(0.0).clip(lower=0.0)
    raw = raw.dropna(subset=["x", "mean"])

    if raw.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    grouped = (
        raw.groupby("x", as_index=False, sort=True)
        .agg(mean=("mean", "mean"), std=("std", "mean"))
        .sort_values("x")
    )

    x_clean = grouped["x"].to_numpy(dtype=float)
    mean_clean = grouped["mean"].to_numpy(dtype=float)
    std_clean = grouped["std"].to_numpy(dtype=float)

    if len(x_clean) == 1:
        return (
            x_clean.copy(),
            mean_clean - std_clean,
            mean_clean + std_clean,
        )

    xs, mean_fit = _polyfit_curve(
        x_clean,
        mean_clean,
        degree=degree,
        n_dense=n_dense,
    )
    xs_std, std_fit = _polyfit_curve(
        x_clean,
        std_clean,
        degree=degree,
        n_dense=n_dense,
    )

    if len(xs) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # Defensive interpolation only aligns the grids if a fallback returned a
    # different number of points. It does not replace the polynomial fitting.
    if len(xs_std) != len(xs) or not np.allclose(xs_std, xs):
        std_fit = np.interp(xs, xs_std, std_fit)

    std_fit = np.maximum(std_fit, 0.0)
    return xs, mean_fit - std_fit, mean_fit + std_fit


# =============================================================================
# 4. Limits helpers
# =============================================================================
def _global_range(frames: Iterable[pd.DataFrame], mean_col: str = "mean", std_col: str = "std") -> Tuple[float, float]:
    low_vals = []
    high_vals = []
    for df in frames:
        if df is None or df.empty:
            continue
        y = pd.to_numeric(df[mean_col], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(df[std_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        mask = np.isfinite(y)
        if not mask.any():
            continue
        y = y[mask]
        e = e[mask[:len(e)]] if len(e) == len(mask) else e
        if len(e) != len(y):
            e = np.zeros_like(y)
        low_vals.append(np.nanmin(y - e))
        high_vals.append(np.nanmax(y + e))

    if not low_vals or not high_vals:
        return 0.0, 1.0

    vmin = float(np.nanmin(low_vals))
    vmax = float(np.nanmax(high_vals))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.05, 0.2)
        return vmin - delta, vmax + delta
    pad = 0.08 * (vmax - vmin)
    return vmin - pad, vmax + pad


# =============================================================================
# 5. Plotting helpers
# =============================================================================
def _plot_one_series(ax, df_city: pd.DataFrame, xcol: str, color: str, style: dict, zbase: int = 3):
    if df_city is None or df_city.empty:
        return None

    df_city = df_city.sort_values(xcol)
    x = pd.to_numeric(df_city[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_city["mean"], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(df_city["std"], errors="coerce").fillna(0).to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return None
    x = x[valid]
    y = y[valid]
    if len(e) == len(valid):
        e = e[valid]
    else:
        e = np.zeros_like(y)

    xs, yl, yu = _polyfit_band(x, y, e)
    xline, yline = _polyfit_curve(x, y)

    if len(xs) > 0:
        ax.fill_between(xs, yl, yu, color=color, alpha=style["band_alpha"], zorder=zbase-2)

    line_handle = None
    if len(xline) > 0:
        line_handle, = ax.plot(
            xline, yline,
            color=color,
            lw=style["linewidth"],
            ls=style["linestyle"],
            zorder=zbase-1,
            label=style["label"],
        )

    ax.plot(
        x, y,
        linestyle="None",
        marker=style["marker"],
        markersize=style["markersize"],
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        color=color,
        zorder=zbase,
    )
    return line_handle


def _plot_city_panel(
    ax_l,
    city: str,
    df_qa01: pd.DataFrame,
    df_qa012: pd.DataFrame,
    df_no2: pd.DataFrame,
    xcol: str,
    xticks,
    xticklabels,
    left_ylim: Tuple[float, float],
    right_ylim: Tuple[float, float],
    panel_title: str,
    show_left_ylabel: bool,
    show_right_ylabel: bool,
):
    base_color = TARGET_CITIES[city]["color"]
    qa012_color = base_color
    qa01_color = base_color
    no2_color = base_color

    ax_r = ax_l.twinx()

    h1 = _plot_one_series(ax_l, df_qa01, xcol, qa01_color, STYLE_QA01, zbase=5)
    h2 = _plot_one_series(ax_l, df_qa012, xcol, qa012_color, STYLE_QA012, zbase=4)
    h3 = _plot_one_series(ax_r, df_no2, xcol, no2_color, STYLE_NO2, zbase=4)

    ax_l.set_xlim(min(xticks) - 0.5, max(xticks) + 0.5)
    ax_l.set_xticks(xticks)
    ax_l.set_xticklabels(xticklabels)
    ax_l.set_ylim(*left_ylim)
    ax_r.set_ylim(*right_ylim)
    ax_l.set_title(panel_title, loc="left", pad=10, fontweight="bold")

    if show_left_ylabel:
        ax_l.set_ylabel("Predicted $\\Delta$XCO$_2$ [ppm]", fontweight="bold")
    else:
        ax_l.set_ylabel("")
    if show_right_ylabel:
        ax_r.set_ylabel("Tropospheric NO$_2$ VCD", fontweight="bold")
    else:
        ax_r.set_ylabel("")

    ax_l.grid(True, alpha=0.35, linestyle="--")

    handles = [h for h in [h1, h2, h3] if h is not None]
    labels = [h.get_label() for h in handles]
    return ax_r, handles, labels


def _make_4panel_figure(
    df_qa01: pd.DataFrame,
    df_qa012: pd.DataFrame,
    df_no2: pd.DataFrame,
    xcol: str,
    xticks,
    xticklabels,
    big_title: str,
    output_path: Path,
):
    left_frames = []
    right_frames = []
    for city in CITY_ORDER:
        left_frames.append(df_qa01[df_qa01["City"] == city])
        left_frames.append(df_qa012[df_qa012["City"] == city])
        right_frames.append(df_no2[df_no2["City"] == city])

    left_ylim = _global_range(left_frames)
    right_ylim = _global_range(right_frames)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), sharex=False)
    axes = axes.ravel()

    all_handles = None
    all_labels = None
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for idx, city in enumerate(CITY_ORDER):
        ax = axes[idx]
        qa01_city = df_qa01[df_qa01["City"] == city].copy()
        qa012_city = df_qa012[df_qa012["City"] == city].copy()
        no2_city = df_no2[df_no2["City"] == city].copy()

        _, handles, labels = _plot_city_panel(
            ax_l=ax,
            city=city,
            df_qa01=qa01_city,
            df_qa012=qa012_city,
            df_no2=no2_city,
            xcol=xcol,
            xticks=xticks,
            xticklabels=xticklabels,
            left_ylim=left_ylim,
            right_ylim=right_ylim,
            panel_title=f"{panel_labels[idx]} {city}",
            show_left_ylabel=(idx % 2 == 0),
            show_right_ylabel=(idx % 2 == 1),
        )
        if all_handles is None and handles:
            all_handles = handles
            all_labels = labels

    for ax in axes[2:]:
        ax.set_xlabel("Month" if xcol == "Month" else "Weekday", fontweight="bold")

    fig.suptitle(big_title, y=0.985, fontsize=GLOBAL_FONT_SIZE + 2, fontweight="bold")

    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc="upper center",
            ncol=3,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.955),
        )

    fig.subplots_adjust(top=0.88, hspace=0.25, wspace=0.18)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# 6. Main
# =============================================================================
def main():
    print("=" * 60)
    print("Plot from existing CSV files only")
    print("=" * 60)
    print(f"Input directory : {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Fitting method  : degree-{POLY_DEGREE} polynomial for means and std bands")

    qa01_month, qa01_week, qa012_month, qa012_week, no2_month, no2_week = load_all_csvs()

    print(
        "Loaded rows -> "
        f"QA01 month: {len(qa01_month)}, "
        f"QA01 weekday: {len(qa01_week)}, "
        f"QA012 month: {len(qa012_month)}, "
        f"QA012 weekday: {len(qa012_week)}, "
        f"NO2 month: {len(no2_month)}, "
        f"NO2 weekday: {len(no2_week)}"
    )

    _make_4panel_figure(
        df_qa01=qa01_month,
        df_qa012=qa012_month,
        df_no2=no2_month,
        xcol="Month",
        xticks=np.arange(1, 13),
        xticklabels=MONTH_LABELS,
        big_title="City-Level Monthly Cycles with Cubic Polynomial Fits",
        output_path=OUTPUT_MONTHLY,
    )

    _make_4panel_figure(
        df_qa01=qa01_week,
        df_qa012=qa012_week,
        df_no2=no2_week,
        xcol="weekday",
        xticks=np.arange(7),
        xticklabels=WEEKDAY_LABELS,
        big_title="City-Level Weekday Cycles with Cubic Polynomial Fits",
        output_path=OUTPUT_WEEKDAY,
    )

    print("Done!")


if __name__ == "__main__":
    main()