# -*- coding: utf-8 -*-
"""
fig_high_sigma_fingerprint.py — 高 sigma 指纹图 (Cohen"s d)

比较 sigma 前 10% 与后 50% 样本在 20 个特征上的标准化均值差异。
产出: FIG-High_Sigma_Fingerprint.png
"""

import os, json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_FILE = "/home/whdong/dl/data/TABLE-SHPXCO2en_sif_no2withmean-ND49_era5_ndvi_meic_ntl_dem_co_0.1deg.pkl"
FEATURES_JSON = "/home/whdong/dl/best_params/selected_features-A01.json"
SCALER_PATH = "/home/whdong/dl/models/XCO2en_SHP-xgb_scaler-A1.pkl"
MODEL_BASE_PATH = "/home/whdong/dl/models/XCO2en_SHP-xgb_model-A01"
OUTPUT_DIR = "/home/whdong/dl/figures/scatter_density_A01"
TARGET = "xco2_enhanced"
SEEDS = [42, 100, 2023, 888, 999, 1234, 5678, 777, 666, 1024]
os.makedirs(OUTPUT_DIR, exist_ok=True)

FN = {
    "no2_trop_mean": "NO2 Mean", "no2_trop_log": "Log(NO2)",
    "era5_blh": "PBLH", "era5_blh_lag1": "PBLH Lag1", "era5_blh_lead1": "PBLH Lead1",
    "era5_d2m": "Dewpt T2m", "era5_d2m_lag1": "Dewpt T2m Lag1", "era5_d2m_lead1": "Dewpt T2m Lead1",
    "era5_sp": "SP", "era5_sp_lag1": "SP Lag1", "era5_sp_lead1": "SP Lead1",
    "era5_ssrd": "SSRD", "era5_ssrd_lag1": "SSRD Lag1", "era5_ssrd_lead1": "SSRD Lead1",
    "era5_t2m": "T2m", "era5_t2m_lag1": "T2m Lag1", "era5_t2m_lead1": "T2m Lead1",
    "era5_tcwv": "TCWV", "era5_tcwv_lag1": "TCWV Lag1", "era5_tcwv_lead1": "TCWV Lead1",
    "era5_u100": "U100", "era5_u100_lag1": "U100 Lag1", "era5_u100_lead1": "U100 Lead1",
    "era5_u10": "U10", "era5_u10_lag1": "U10 Lag1", "era5_u10_lead1": "U10 Lead1",
    "era5_v100": "V100", "era5_v100_lag1": "V100 Lag1", "era5_v100_lead1": "V100 Lead1",
    "era5_v10": "V10", "era5_v10_lag1": "V10 Lag1", "era5_v10_lead1": "V10 Lead1",
    "meic_nox": "MEIC NOx", "ndvi": "NDVI", "ntl": "NTL", "dem_mean": "Elevation",
    "ndvi_t2m_cross": "NDVI x T2m", "ssrd_t2m_cross": "SSRD x T2m", "ntl_nox_cross": "NTL x NOx",
    "doy_sin": "DOY sin", "doy_cos": "DOY cos", "month_sin": "Month sin", "month_cos": "Month cos",
    "grid_lon": "Lon", "grid_lat": "Lat", "era5_wind_speed": "Wind Speed"
}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "axes.unicode_minus": False, "font.size": 11, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 5, "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.6, "grid.alpha": 0.35,
    "grid.color": "#B0B0B0", "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05})

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return (np.mean(a)-np.mean(b)) / sp if sp > 0 else 0

def main():
    print("Loading data...")
    df = pd.read_pickle(DATA_FILE)
    dc = df.dropna().copy()
    dc["no2_trop_log"] = np.log(dc["no2_trop"])
    dc["date"] = pd.to_datetime(dc["date"])
    dc["month"] = dc["date"].dt.month
    dc["doy"] = dc["date"].dt.dayofyear
    dc["month_sin"] = np.sin(2 * np.pi * dc["month"] / 12.0)
    dc["month_cos"] = np.cos(2 * np.pi * dc["month"] / 12.0)
    dc["doy_sin"] = np.sin(2 * np.pi * dc["doy"] / 365.25)
    dc["doy_cos"] = np.cos(2 * np.pi * dc["doy"] / 365.25)
    dc["ndvi_t2m_cross"] = dc["ndvi"] * dc["era5_t2m"]
    dc["ssrd_t2m_cross"] = dc["era5_ssrd"] * dc["era5_t2m"]
    dc["ntl_nox_cross"] = dc["ntl"] * dc["meic_nox"]
    dc["era5_wind_speed"] = np.sqrt(dc["era5_u100"]**2+dc["era5_v100"]**2)
    with open(FEATURES_JSON) as f: sf = json.load(f)

    _, dt = train_test_split(dc, test_size=0.2, random_state=42)
    Xr = dt[sf].values; yt = dt[TARGET].values
    es = joblib.load(SCALER_PATH); Xs = es.transform(Xr)

    print("Running inference...")
    tp = [joblib.load(f"{MODEL_BASE_PATH}_seed{s}.pkl").predict(Xs) for s in SEEDS]
    ep = np.mean(tp, axis=0); sigma = np.std(tp, axis=0)

    hi = sigma >= np.percentile(sigma, 90)
    lo = sigma < np.percentile(sigma, 50)
    print(f"  High sigma: {hi.sum():,}, Low sigma: {lo.sum():,}")

    # Raw feature values (not scaled) for meaningful comparison
    Xraw = Xr  # keep original scale
    results = []
    for i, fn in enumerate(sf):
        v = Xraw[:, i]
        d = cohens_d(v[hi], v[lo])
        lbl = FN.get(fn, fn)
        results.append({"feature": fn, "label": lbl, "cohens_d": d})

    results = sorted(results, key=lambda r: abs(r["cohens_d"]), reverse=True)
    labs = [r["label"] for r in results]
    ds = [r["cohens_d"] for r in results]
    nf = len(results)

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in ds]
    yp = np.arange(nf)
    ax.barh(yp, ds, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(yp); ax.set_yticklabels(labs, fontsize=12)
    ax.set_xlabel("Cohen's d (High σ Top10% - Low σ Bottom50%)", fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlim(-max(abs(d) for d in ds)*1.15, max(abs(d) for d in ds)*1.35)

    # Annotate values
    x_range = max(abs(d) for d in ds)
    for i, d in enumerate(ds):
        if d >= 0:
            ax.text(d + 0.03 * x_range, i, f"{d:+.2f}", va="center",
                    fontsize=12, color=colors[i], ha="left")
        else:
            ax.text(d - 0.03 * x_range, i, f"{d:+.2f}", va="center",
                    fontsize=12, color=colors[i], ha="right")

    # ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    # ax.set_title("High Sigma Fingerprint", fontsize=13, pad=12)
    sp = os.path.join(OUTPUT_DIR, "FIGtest-High_Sigma_Fingerprint.png")
    plt.tight_layout(); plt.savefig(sp, dpi=300); plt.close()
    print(f"Saved: {sp}")
    
    print("\n===== Cohen's d 值 (Top 10% sigma vs Bottom 50% sigma) =====")
    print(f"{'Feature':>25}  {'Cohen_d':>8}  {'|d| Rank':>8}")
    print("-" * 45)
    for rank, r in enumerate(results, 1):
        print(f"{r['label']:>25}  {r['cohens_d']:>+8.4f}  {rank:>8}")
    print("=" * 45)

    # Print stats
    print(f"\nTop 5 positive (higher in high-sigma group):")
    for r in results[:5]:
        print(f"  {r['label']:>20}  d = {r['cohens_d']:+.3f}")
    print(f"\nBottom 5 negative (lower in high-sigma group):")
    for r in results[-5:]:
        print(f"  {r['label']:>20}  d = {r['cohens_d']:+.3f}")

if __name__ == "__main__":
    main()
