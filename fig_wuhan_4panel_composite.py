# -*- coding: utf-8 -*-
"""
fig_wuhan_4panel_composite_v4.py

2x2 composite figure for Wuhan case study:
  (a) Wuhan-region annual mean ΔXCO2 map + oblique transect + Wuhan point.
  (b) Oblique cross-section across Wuhan.
  (c) Wuhan monthly cycle: QA(0,1) ΔXCO2 + NO2 VCD.
  (d) Wuhan weekday cycle: QA(0,1) ΔXCO2 + NO2 VCD.

This revised version addresses:
  1) robust local shapefile rendering without triggering Natural Earth download;
  2) identical y-axis limits between panel (c) and panel (d);
  3) Wuhan point annotation in panel (a);
  4) red ΔXCO2 curves in panels (c) and (d), consistent with panel (b);
  5) a wider panel (b) and a narrower panel (a), while preserving the overall
     left and right alignment of the figure;
  6) one global font-size variable for all figure text;
  7) reduced vertical spacing and reduced spacing between panels (c) and (d);
  8) Wuhan annotation without a background box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader

try:
    import geopandas as gpd
except Exception:
    gpd = None


# =============================================================================
# 1. User configuration
# =============================================================================
ACTIVE_VERSION = "A01"
BASE_DIR = Path(f"/home/whdong/dl/ML-prediction-output_result/{ACTIVE_VERSION}")

XCO2_NC_PATH = Path(
    "/home/whdong/dl/ML-prediction-output_result/A01/nc_files_QA01/"
    "xco2en_stats_20230101_20231231_QA01.nc"
)
NO2_NC_PATH = Path(
    "/home/whdong/dl/ML-prediction-output_result/A01/nc_files_QA01/"
    "tropomi_no2_stats_20230101_20231231.nc"
)
SHAPEFILE_PATH = Path("/home/whdong/shapefile/china/province.shp")
CUSTOM_CMAP_PATH = Path("/home/whdong/WhGrYlRd.txt")

INPUT_DIR_CANDIDATES = [
    BASE_DIR / "figures_case_study_qa_mask",
    BASE_DIR / "figures_case_study",
]
REQUIRED_CSV_NAMES = [
    "FIG5_intermediate_pred_month_QA01.csv",
    "FIG5_intermediate_pred_weekday_QA01.csv",
    "FIG5_intermediate_no2_month.csv",
    "FIG5_intermediate_no2_weekday.csv",
]

OUTPUT_DIR = BASE_DIR / "figures_composite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG = OUTPUT_DIR / "FIG_Wuhan_Composite_4panel.png"

# Panel (a) map extent
MAP_EXTENT = [114, 116.6, 28.0, 33.0]

# Wuhan oblique transect parameters
ANCHOR_LON = 114.3
ANCHOR_LAT = 30.6
SLOPE = -0.7
LON_START = 113.3
LON_END = 116.0
NUM_SAMPLES = 100
LOCATION_NAME = "Wuhan"

# Polynomial fitting for panel (c) and (d)
POLY_DEGREE = 3
N_DENSE = 300

# Layout knobs
FIGSIZE = (15, 11)
FIG_DPI = 300
SUBPLOT_LEFT = 0.065
SUBPLOT_RIGHT = 0.94
SUBPLOT_BOTTOM = 0.07
SUBPLOT_TOP = 0.955
SUBPLOT_WSPACE_TOP = 0.24
SUBPLOT_WSPACE_BOTTOM = 0.18  # smaller gap between panels (c) and (d)
SUBPLOT_HSPACE = 0.12         # smaller gap between the upper and lower rows
TOP_WIDTH_RATIOS = [0.84, 1.36]   # panel (a) narrower, panel (b) wider
BOTTOM_WIDTH_RATIOS = [1.0, 1.0]  # keep lower row balanced

# Vertical map colorbar; position is derived from panel (a) after layout.
MAP_COLORBAR_PAD = 0.008
MAP_COLORBAR_WIDTH = 0.013

# Change only this value to adjust every text element in the whole figure:
# axes labels, tick labels, legends, annotations, panel labels, and colorbar.
GLOBAL_FONT_SIZE = 12
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
    "axes.linewidth": 1.3,
    "lines.linewidth": 1.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.top": False,
    "ytick.right": False,
    "axes.grid": False,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Panel (c) and (d) styles
STYLE_XCO2 = {
    "color": "#d63031",  # changed to red, consistent with panel (b)
    "marker": "o",
    "linestyle": "-",
    "linewidth": 2.2,
    "markersize": 5.8,
    "band_alpha": 0.16,
    "label": "QA(0,1) $\\Delta$XCO$_2$",
}
STYLE_NO2 = {
    "color": "#1f77b4",
    "marker": "s",
    "linestyle": "--",
    "linewidth": 2.0,
    "markersize": 5.4,
    "band_alpha": 0.10,
    "label": "NO$_2$ VCD",
}


# =============================================================================
# 2. Utility functions
# =============================================================================
def detect_input_dir() -> Path:
    checked = []
    for candidate in INPUT_DIR_CANDIDATES:
        missing = [name for name in REQUIRED_CSV_NAMES if not (candidate / name).exists()]
        checked.append((candidate, missing))
        if not missing:
            return candidate

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


def require_path(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {name} not found: {path}")


def get_custom_cmap(txt_path: Path = CUSTOM_CMAP_PATH):
    if txt_path.exists():
        colors = np.genfromtxt(txt_path, delimiter=" ")
        return mpl.colors.ListedColormap(colors / 255.0)
    return "viridis"


def generate_oblique_profile(
    anchor_lon: float = ANCHOR_LON,
    anchor_lat: float = ANCHOR_LAT,
    slope: float = SLOPE,
    lon_start: float = LON_START,
    lon_end: float = LON_END,
    num_samples: int = NUM_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    lon_profile = np.linspace(lon_start, lon_end, num_samples)
    lat_profile = anchor_lat + slope * (lon_profile - anchor_lon)
    return lon_profile, lat_profile


def infer_uncertainty_var(ds: xr.Dataset) -> str:
    candidates = [
        "uncertainty_1sigma_mean",
        "uncertainty_mean",
        "unc_mean",
        "sigma_mean",
    ]
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError("No supported uncertainty variable found.")


def infer_xco2_mean_var(ds: xr.Dataset) -> str:
    candidates = ["xco2_enhanced_mean", "pred_mean", "xco2_mean"]
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError("No supported XCO2 mean variable found.")


def infer_no2_spread_var(ds: xr.Dataset) -> Tuple[str, str]:
    """Return NO2 spread variable name and kind: ``std`` or ``var``."""
    std_candidates = [
        "no2_tvcd_std", "no2_tvcd_stdev", "no2_tvcd_sigma",
        "no2_std", "no2_sigma",
    ]
    var_candidates = [
        "no2_tvcd_var", "no2_tvcd_variance",
        "no2_var", "no2_variance",
    ]
    for name in std_candidates:
        if name in ds.data_vars:
            return name, "std"
    for name in var_candidates:
        if name in ds.data_vars:
            return name, "var"
    raise KeyError(
        "No supported NO2 standard-deviation or variance variable found. "
        f"Available variables: {', '.join(ds.data_vars)}"
    )


def add_local_province_boundaries(ax, shapefile_path: Path) -> bool:
    """Draw local province boundaries robustly and avoid Natural Earth fallback."""
    if not shapefile_path.exists():
        print(f"Warning: local shapefile not found -> {shapefile_path}")
        return False

    # Try geopandas first
    if gpd is not None:
        try:
            gdf = gpd.read_file(shapefile_path)
            ax.add_geometries(
                gdf.geometry,
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="#333333",
                linewidth=0.6,
                alpha=0.75,
                zorder=2,
            )
            return True
        except Exception as exc:
            print(f"Warning: geopandas failed to read shapefile, retry with cartopy reader. Reason: {exc}")

    # Then try cartopy shapereader with common Chinese-shapefile encodings
    for enc in ["gbk", "utf-8", None]:
        try:
            if enc is None:
                reader = shpreader.Reader(str(shapefile_path))
            else:
                reader = shpreader.Reader(str(shapefile_path), encoding=enc)
            ax.add_geometries(
                list(reader.geometries()),
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="#333333",
                linewidth=0.6,
                alpha=0.75,
                zorder=2,
            )
            return True
        except Exception as exc:
            last_exc = exc
            continue

    print(f"Warning: local shapefile could not be drawn. Last error: {last_exc}")
    return False


def _prepare_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    work = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x), errors="coerce"),
        "y": pd.to_numeric(pd.Series(y), errors="coerce"),
    }).dropna()
    if work.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    work = work.groupby("x", as_index=False, sort=True)["y"].mean().sort_values("x")
    return work["x"].to_numpy(dtype=float), work["y"].to_numpy(dtype=float)


def polyfit_curve(x: np.ndarray, y: np.ndarray, degree: int = POLY_DEGREE, n_dense: int = N_DENSE):
    x_clean, y_clean = _prepare_xy(x, y)
    if len(x_clean) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if len(x_clean) == 1:
        return x_clean.copy(), y_clean.copy()
    fit_degree = min(int(degree), len(x_clean) - 1)
    xs = np.linspace(x_clean.min(), x_clean.max(), int(n_dense))
    try:
        coeff = np.polyfit(x_clean, y_clean, fit_degree)
        ys = np.polyval(coeff, xs)
        return xs, ys
    except Exception:
        return x_clean, y_clean


def polyfit_band(x: np.ndarray, y: np.ndarray, yerr: np.ndarray, degree: int = POLY_DEGREE, n_dense: int = N_DENSE):
    raw = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x), errors="coerce"),
        "y": pd.to_numeric(pd.Series(y), errors="coerce"),
        "e": pd.to_numeric(pd.Series(yerr), errors="coerce"),
    }).dropna(subset=["x", "y"])
    if raw.empty:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    raw["e"] = raw["e"].fillna(0.0)
    raw = raw.groupby("x", as_index=False, sort=True).agg({"y": "mean", "e": "mean"}).sort_values("x")

    x_clean = raw["x"].to_numpy(dtype=float)
    y_clean = raw["y"].to_numpy(dtype=float)
    e_clean = np.maximum(raw["e"].to_numpy(dtype=float), 0.0)

    if len(x_clean) == 1:
        return x_clean.copy(), y_clean - e_clean, y_clean + e_clean

    xs, ys = polyfit_curve(x_clean, y_clean, degree=degree, n_dense=n_dense)
    _, es = polyfit_curve(x_clean, e_clean, degree=degree, n_dense=n_dense)
    es = np.maximum(es, 0.0)
    return xs, ys - es, ys + es


def global_range(frames: Iterable[pd.DataFrame]) -> Tuple[float, float]:
    low_vals = []
    high_vals = []
    for df in frames:
        if df is None or df.empty:
            continue
        y = pd.to_numeric(df["mean"], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(df["std"], errors="coerce").fillna(0).to_numpy(dtype=float)
        valid = np.isfinite(y)
        if not valid.any():
            continue
        y = y[valid]
        e = e[valid] if len(e) == len(valid) else np.zeros_like(y)
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
# 3. Panel (a): Map
# =============================================================================
def plot_panel_a_map(ax, nc_path: Path, shapefile_path: Path):
    ds = xr.open_dataset(nc_path)
    xvar = infer_xco2_mean_var(ds)
    lons = ds["lon"].values
    lats = ds["lat"].values
    xco2 = ds[xvar].values

    cmap = get_custom_cmap()
    mesh = ax.pcolormesh(
        lons, lats, xco2,
        cmap=cmap,
        vmin=1.0, vmax=5.0,
        transform=ccrs.PlateCarree(),
        zorder=1,
        shading="nearest",
    )
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    # Draw only local shapefile; do not fallback to internet-downloaded features.
    add_local_province_boundaries(ax, shapefile_path)

    # Transect
    lons_tr, lats_tr = generate_oblique_profile()
    ax.plot(
        lons_tr, lats_tr,
        color="#2ca25f",
        linewidth=2.6,
        transform=ccrs.PlateCarree(),
        zorder=4,
        label="Oblique transect",
    )
    ax.scatter(
        [lons_tr[0], lons_tr[-1]], [lats_tr[0], lats_tr[-1]],
        color="#2ca25f", s=38, marker="o",
        edgecolors="black", linewidths=0.6,
        transform=ccrs.PlateCarree(), zorder=5,
    )

    # Wuhan point + text annotation (new request)
    ax.scatter(
        [ANCHOR_LON], [ANCHOR_LAT],
        color="#6c5ce7",
        s=300,
        marker="*",
        edgecolors="black",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=6,
        label="Wuhan",
    )

    xticks = np.arange(113, 117, 1)
    yticks = np.arange(28, 33, 1)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color="gray", alpha=0.35, linestyle=":")

    # Force the map axes to fill the full subplot cell so that panel (a) and
    # panel (b) have aligned top and bottom boundaries. Without this, Cartopy
    # may preserve geographic aspect ratio and visually shrink panel (a).
    ax.set_aspect("auto")

    ax.text(
    0.025,
    0.025,
    "(a)",
    transform=ax.transAxes,
    fontsize=18,
    fontweight="bold",
    ha="left",
    va="bottom",
    color="black",
    zorder=20,
    )
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.90,markerscale=1)

    # The vertical colorbar is created in main() after final layout.
    return mesh



# =============================================================================
# 4. Panel (b): Oblique transect profile
# =============================================================================
def plot_panel_b_transect(ax1, xco2_nc: Path, no2_nc: Path):
    ds_xco2 = xr.open_dataset(xco2_nc)
    ds_no2 = xr.open_dataset(no2_nc)

    xvar = infer_xco2_mean_var(ds_xco2)
    uvar = infer_uncertainty_var(ds_xco2)
    if "no2_tvcd_mean" not in ds_no2.data_vars:
        raise KeyError("Variable 'no2_tvcd_mean' not found in NO2 NetCDF.")
    no2_spread_var, no2_spread_kind = infer_no2_spread_var(ds_no2)

    if len(ds_no2["lon"].values) > 1 and len(ds_no2["lat"].values) > 1:
        res_lon = np.diff(ds_no2["lon"].values)[0]
        res_lat = np.diff(ds_no2["lat"].values)[0]
        ds_no2 = ds_no2.assign_coords(
            lon=ds_no2["lon"] + res_lon / 2.0,
            lat=ds_no2["lat"] + res_lat / 2.0,
        )

    no2_array = ds_no2["no2_tvcd_mean"].values
    grad_lat, grad_lon = np.gradient(no2_array, ds_no2["lat"].values, ds_no2["lon"].values)
    grad_abs = np.sqrt(grad_lon ** 2 + grad_lat ** 2)
    ds_no2["no2_gradient"] = (("lat", "lon"), grad_abs)

    lon_profile, lat_profile = generate_oblique_profile()
    lon_da = xr.DataArray(lon_profile, dims="profile")
    lat_da = xr.DataArray(lat_profile, dims="profile")

    slice_xco2 = ds_xco2.interp(lon=lon_da, lat=lat_da, method="linear")
    slice_no2 = ds_no2.interp(lon=lon_da, lat=lat_da, method="linear")

    xco2_mean = slice_xco2[xvar].values
    xco2_sigma = slice_xco2[uvar].values
    no2_mean = slice_no2["no2_tvcd_mean"].values
    no2_spread_raw = slice_no2[no2_spread_var].values
    if no2_spread_kind == "var":
        no2_std = np.sqrt(np.maximum(no2_spread_raw, 0.0))
    else:
        no2_std = np.maximum(no2_spread_raw, 0.0)
    no2_grad = slice_no2["no2_gradient"].values

    ax2 = ax1.twinx()
    # ax3 = ax1.twinx()
    # ax3.spines["right"].set_position(("outward", 48))

    color_xco2 = "#d63031"
    color_no2 = "#0984e3"
    color_grad = "#2d3436"

    line1, = ax1.plot(lon_profile, xco2_mean, color=color_xco2, linewidth=2.3, label=r"$\Delta$XCO$_2$")
    fill1 = ax1.fill_between(
        lon_profile, xco2_mean - xco2_sigma, xco2_mean + xco2_sigma,
        color=color_xco2, alpha=0.16,
        label=r"$\Delta$XCO$_2$ uncertainty ($\pm$1$\sigma$)",
    )
    line2, = ax2.plot(
        lon_profile, no2_mean, color=color_no2, linewidth=1.8,
        linestyle="-", label="NO$_2$ VCD"
    )
    fill2 = ax2.fill_between(
        lon_profile,
        no2_mean - no2_std,
        no2_mean + no2_std,
        color=color_no2,
        alpha=0.13,
        label=r"NO$_2$ variability ($\pm$1$\sigma$)",
    )
    # line3, = ax3.plot(lon_profile, no2_grad, color=color_grad, linewidth=1.5, linestyle="--", label=r"$|\nabla$NO$_2$|")

    lat_start_geo = lat_profile[0]
    lat_end_geo = lat_profile[-1]
    ax1.set_xlabel(
        f"Longitude",
        fontweight="bold", labelpad=8,
    )
    ax1.set_ylabel(r"Predicted $\Delta$XCO$_2$ [ppm]", color=color_xco2, fontweight="bold")
    ax2.set_ylabel(r"Tropospheric NO$_2$ column [$\mu$mol m$^{-2}$]", color=color_no2, fontweight="bold")
    # ax3.set_ylabel(r"$|\nabla$NO$_2$| [$\mu$mol m$^{-2}$ degree$^{-1}$]", color=color_grad, fontweight="bold")

    ax1.tick_params(axis="y", labelcolor=color_xco2)
    ax2.tick_params(axis="y", labelcolor=color_no2)
    # ax3.tick_params(axis="y", labelcolor=color_grad)

    ax1.set_xlim(LON_START, LON_END)
    ax1.xaxis.set_major_formatter(
    FuncFormatter(lambda value, position: f"{value:g}°E"))
    if np.isfinite(xco2_mean).any():
        ax1.set_ylim(0, max(10.0, np.nanmax(xco2_mean + xco2_sigma) * 1.15))
    if np.isfinite(no2_mean).any():
        ax2.set_ylim(0, np.nanmax(no2_mean + no2_std) * 1.15)
    # if np.isfinite(no2_grad).any():
    #     ax3.set_ylim(0, np.nanmax(no2_grad) * 1.20)

    ax1.grid(True, linestyle=":", alpha=0.40, color="gray")
    ax1.axvline(x=ANCHOR_LON, color="#6c5ce7", linestyle=":", linewidth=1.4, alpha=0.85)
    ax1.text(
        ANCHOR_LON + 0.04,
        ax1.get_ylim()[1] * 0.92,
        "Wuhan urban center",
        color="#6c5ce7",
        fontsize=GLOBAL_FONT_SIZE,
        rotation=90,
        va="top", ha="left", style="italic",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0),
    )

    handles = [line1, fill1, line2, fill2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right", frameon=True, facecolor="white", framealpha=0.90)
    ax1.text(
    0.025,
    0.025,
    "(b)",
    transform=ax1.transAxes,
    fontsize=18,
    fontweight="bold",
    ha="left",
    va="bottom",
    color="black",
    zorder=20,
    )   


# =============================================================================
# 5. Panel (c) and (d): Wuhan cycles
# =============================================================================
def load_wuhan_cycle_csvs(input_dir: Path):
    qa01_month = pd.read_csv(input_dir / "FIG5_intermediate_pred_month_QA01.csv")
    qa01_week = pd.read_csv(input_dir / "FIG5_intermediate_pred_weekday_QA01.csv")
    no2_month = pd.read_csv(input_dir / "FIG5_intermediate_no2_month.csv")
    no2_week = pd.read_csv(input_dir / "FIG5_intermediate_no2_weekday.csv")

    qa01_month = qa01_month[qa01_month["City"] == "Wuhan"].copy()
    qa01_week = qa01_week[qa01_week["City"] == "Wuhan"].copy()
    no2_month = no2_month[no2_month["City"] == "Wuhan"].copy()
    no2_week = no2_week[no2_week["City"] == "Wuhan"].copy()
    return qa01_month, qa01_week, no2_month, no2_week


def plot_one_smoothed_series(ax, df_city: pd.DataFrame, xcol: str, style: dict, zbase: int = 4):
    if df_city is None or df_city.empty:
        return None
    df_city = df_city.sort_values(xcol)
    x = pd.to_numeric(df_city[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_city["mean"], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(df_city["std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return None
    x = x[valid]
    y = y[valid]
    e = e[valid] if len(e) == len(valid) else np.zeros_like(y)

    xs_band, ylow, yhigh = polyfit_band(x, y, e)
    xs_line, ys_line = polyfit_curve(x, y)

    if len(xs_band) > 0:
        ax.fill_between(xs_band, ylow, yhigh, color=style["color"], alpha=style["band_alpha"], zorder=zbase - 2)

    handle = None
    if len(xs_line) > 0:
        handle, = ax.plot(xs_line, ys_line, color=style["color"], ls=style["linestyle"], lw=style["linewidth"], zorder=zbase - 1, label=style["label"])

    ax.plot(
        x, y,
        linestyle="None",
        marker=style["marker"],
        markersize=style["markersize"],
        markerfacecolor=style["color"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        color=style["color"],
        zorder=zbase,
    )
    return handle


def plot_wuhan_cycle_panel(ax_l, df_xco2: pd.DataFrame, df_no2: pd.DataFrame, xcol: str, xticks, xticklabels, title: str, xlabel: str, left_ylim: Tuple[float, float], right_ylim: Tuple[float, float], panel_label,):
    ax_r = ax_l.twinx()
    handle_x = plot_one_smoothed_series(ax_l, df_xco2, xcol, STYLE_XCO2, zbase=5)
    handle_n = plot_one_smoothed_series(ax_r, df_no2, xcol, STYLE_NO2, zbase=4)

    ax_l.set_xlim(min(xticks), max(xticks))
    ax_l.margins(x=0)
    ax_l.set_xticks(xticks)
    ax_l.set_xticklabels(xticklabels)
    ax_l.set_xlabel(xlabel, fontweight="bold")
    ax_l.set_ylabel(r"Predicted $\Delta$XCO$_2$ [ppm]", color=STYLE_XCO2["color"], fontweight="bold")
    ax_r.set_ylabel(r"Tropospheric NO$_2$ VCD", color=STYLE_NO2["color"], fontweight="bold")
    ax_l.tick_params(axis="y", labelcolor=STYLE_XCO2["color"])
    ax_r.tick_params(axis="y", labelcolor=STYLE_NO2["color"])
    ax_l.grid(True, linestyle="--", linewidth=0.7, alpha=0.35, color="#B0B0B0")

    # Apply common limits to panels (c) and (d)
    ax_l.set_ylim(*left_ylim)
    ax_r.set_ylim(*right_ylim)

    # handles = [h for h in [handle_x, handle_n] if h is not None]
    # labels = [h.get_label() for h in handles]
    # if handles:
    #     ax_l.legend(handles, labels, loc="upper left", frameon=True, facecolor="white", framealpha=0.90)

    ax_l.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_r.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax_l.text(
    0.025,
    0.025,
    panel_label,
    transform=ax_l.transAxes,
    fontsize=18,
    fontweight="bold",
    ha="left",
    va="bottom",
    color="black",
    zorder=20,
    )


# =============================================================================
# 6. Main
# =============================================================================
def main():
    require_path(XCO2_NC_PATH, "XCO2 NetCDF")
    require_path(NO2_NC_PATH, "NO2 NetCDF")
    require_path(SHAPEFILE_PATH, "province shapefile")

    input_dir = detect_input_dir()
    qa01_month, qa01_week, no2_month, no2_week = load_wuhan_cycle_csvs(input_dir)

    # Common y-axis limits for panels (c) and (d)
    cycle_left_ylim = global_range([qa01_month, qa01_week])
    cycle_right_ylim = global_range([no2_month, no2_week])

    fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)

    # Two row layout to let top row have asymmetric widths while preserving full figure alignment.
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=SUBPLOT_HSPACE,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=TOP_WIDTH_RATIOS, wspace=SUBPLOT_WSPACE_TOP)
    bottom = outer[1].subgridspec(1, 2, width_ratios=BOTTOM_WIDTH_RATIOS, wspace=SUBPLOT_WSPACE_BOTTOM)

    ax_a = fig.add_subplot(top[0, 0], projection=ccrs.PlateCarree())
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    map_mesh = plot_panel_a_map(ax_a, XCO2_NC_PATH, SHAPEFILE_PATH)
    plot_panel_b_transect(ax_b, XCO2_NC_PATH, NO2_NC_PATH)
    plot_wuhan_cycle_panel(
        ax_c,
        qa01_month,
        no2_month,
        xcol="Month",
        xticks=np.arange(1, 13),
        xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        title="(c) Wuhan monthly cycle",
        xlabel="Month",
        left_ylim=cycle_left_ylim,
        right_ylim=cycle_right_ylim,
         panel_label="(c)",
    )
    plot_wuhan_cycle_panel(
        ax_d,
        qa01_week,
        no2_week,
        xcol="weekday",
        xticks=np.arange(7),
        xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        title="(d) Wuhan weekday cycle",
        xlabel="Weekday",
        left_ylim=cycle_left_ylim,
        right_ylim=cycle_right_ylim,
         panel_label="(d)",
    )

    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM,
        top=SUBPLOT_TOP,
    )

    # Create a vertical colorbar with exactly the same top, bottom, and height
    # as panel (a). Its horizontal position can be adjusted with the two
    # configuration values near the top of this script.
    fig.canvas.draw()
    map_pos = ax_a.get_position()
    cax = fig.add_axes([
        map_pos.x1 + MAP_COLORBAR_PAD,
        map_pos.y0,
        MAP_COLORBAR_WIDTH,
        map_pos.height,
    ])
    cb = fig.colorbar(map_mesh, cax=cax, orientation="vertical")
    cb.set_label(
        r"Predicted $\Delta$XCO$_2$ mean [ppm]",
        fontsize=GLOBAL_FONT_SIZE,
        labelpad=8,
    )
    cb.ax.tick_params(labelsize=GLOBAL_FONT_SIZE)

    fig.savefig(OUTPUT_FIG, dpi=FIG_DPI)
    plt.close(fig)

    print("=" * 70)
    print("Composite figure completed")
    print("=" * 70)
    print(f"Input XCO2 NC : {XCO2_NC_PATH}")
    print(f"Input NO2 NC  : {NO2_NC_PATH}")
    print(f"Shapefile     : {SHAPEFILE_PATH}")
    print(f"CSV directory : {input_dir}")
    print(f"Output figure : {OUTPUT_FIG}")
    print(f"Cycle left y-limits  : {cycle_left_ylim}")
    print(f"Cycle right y-limits : {cycle_right_ylim}")
    print("Panel (b) NO2 shading: mean +/- 1 standard deviation")


if __name__ == "__main__":
    main()