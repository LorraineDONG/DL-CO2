#!/usr/bin/env python3
"""E2测试集密度散点图。只读取预测CSV，不训练、不筛除高值或大误差。
依赖: pip install numpy pandas matplotlib
修改下面CSV_FILE后，直接在IDE运行。使用训练内CV选中的模型预测，
不要按测试R2挑种子。已知最好结果为fixed45_oldparams (R2约0.75454)。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# 必须指向selected_model下的预测CSV，而不是E2_baseline的同名文件。
CSV_FILE = Path('/home/whdong/dl/E2_feature_parameter_experiments/run_20261009_200758_099885/selected_model/common_test_predictions.csv')
OUTPUT_DIR = None  # None: 保存到CSV所在目录的density_figures文件夹
TRUE_COLUMN = 'xco2_enhanced'
PRED_COLUMN = 'prediction'  # 如列名不同，改为实际预测列名；不要选择单个seed列
SUBSET_COLUMN = 'test_subset'
MODEL_LABEL = 'E2: 45 features, original parameters'
DRAW_SUBSETS = True
EXPECTED_N = 4382  # 合并测试集身份的数量检查；其他正式划分请相应修改
GRID_BINS = 90
DPI = 400

def draw_plot(data, title, stem, output):
    x = pd.to_numeric(data[TRUE_COLUMN], errors='raise').to_numpy(float)
    y = pd.to_numeric(data[PRED_COLUMN], errors='raise').to_numpy(float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('需要至少3条有限数值；请核查原始文件，不自动删除异常记录。')
    sst = np.sum((x-x.mean())**2)
    if sst <= 0:
        raise ValueError('观测值没有变化，R2及拟合线无法计算。')
    error = y-x
    slope, intercept = np.polyfit(x, y, 1)
    m = dict(n=len(x), r2=float(1-np.sum(error**2)/sst),
             rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
             bias=float(error.mean()), slope_pred_on_true=float(slope), intercept=float(intercept))
    # 二维直方图给每个点赋局部计数；不是KDE，也不是每格一个hexbin。
    hist, xe, ye = np.histogram2d(x, y, bins=GRID_BINS)
    ix = np.clip(np.searchsorted(xe, x, side='right')-1, 0, GRID_BINS-1)
    iy = np.clip(np.searchsorted(ye, y, side='right')-1, 0, GRID_BINS-1)
    density = hist[ix, iy]
    order = np.argsort(density)  # 密集点最后绘制
    low = float(min(x.min(), y.min()))
    high = float(max(x.max(), y.max()))
    padding = (high-low)*0.05
    limits = (low-padding, high+padding)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.linewidth':1.1})
    fig, ax = plt.subplots(figsize=(7.6, 6.6), constrained_layout=True)
    dots = ax.scatter(x[order], y[order], c=density[order], s=13,
                      cmap='viridis', norm=LogNorm(vmin=1, vmax=max(2, density.max())),
                      edgecolors='none', rasterized=True)
    line_x = np.array(limits)
    ax.plot(line_x, line_x, '--', color='0.45', lw=1.4, label='1:1 line')
    ax.plot(line_x, slope*line_x+intercept, color='#d84a3a', lw=1.8,
            label=f'OLS fit: y = {slope:.3f}x {intercept:+.3f}')
    text = (f'N = {len(x):,}\n'+r'$R^2$'+f' = {m["r2"]:.4f}\n'
            f'RMSE = {m["rmse"]:.3f} ppm\nMAE = {m["mae"]:.3f} ppm\n'
            f'Bias = {m["bias"]:+.3f} ppm')
    ax.text(.04, .96, text, transform=ax.transAxes, va='top', linespacing=1.5,
            bbox=dict(facecolor='white', alpha=.92, edgecolor='0.8', boxstyle='round,pad=0.5'))
    ax.set(xlim=limits, ylim=limits, xlabel=r'Observed $\Delta XCO_2$ (ppm)',
           ylabel=r'Predicted $\Delta XCO_2$ (ppm)', title=title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right', fontsize=9, framealpha=.95)
    ax.grid(alpha=.15)
    fig.colorbar(dots, ax=ax, pad=.025, shrink=.88, label='Samples per 2D bin (log scale)')
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=DPI)
    plt.close(fig)
    print(stem, json.dumps(m, ensure_ascii=False))
    return dict(subset=stem, **m)

def main():
    data = pd.read_csv(CSV_FILE)
    missing = {TRUE_COLUMN, PRED_COLUMN} - set(data.columns)
    if missing:
        raise ValueError(f'缺少列{missing}；实际列名: {list(data.columns)}')
    if EXPECTED_N is not None and len(data) != EXPECTED_N:
        raise ValueError(f'预期{EXPECTED_N}条合并测试记录，实际{len(data)}条，请检查输入。')
    if '_row_id' in data and data['_row_id'].duplicated().any():
        raise ValueError('存在重复_row_id，请核查是否重复拼接。')
    output = Path(OUTPUT_DIR) if OUTPUT_DIR else CSV_FILE.parent / 'density_figures'
    output.mkdir(parents=True, exist_ok=True)
    rows = [draw_plot(data, MODEL_LABEL+'\nCombined satellite holdout', 'E2_combined_test', output)]
    if DRAW_SUBSETS and SUBSET_COLUMN in data:
        for subset in ('historical_reference', 'added_satellite_holdout'):
            selected = data.loc[data[SUBSET_COLUMN] == subset]
            if not selected.empty:
                rows.append(draw_plot(selected, MODEL_LABEL+'\n'+subset,
                                      'E2_'+subset, output))
    pd.DataFrame(rows).to_csv(output/'E2_density_metrics.csv', index=False)
    (output/'plot_source.json').write_text(json.dumps(dict(
        csv_file=str(CSV_FILE.resolve()), model_label=MODEL_LABEL,
        true_column=TRUE_COLUMN, pred_column=PRED_COLUMN,
        bias_definition='prediction - observation',
        r2_definition='1 - SSE / SST; not squared Pearson correlation',
        fit_definition='OLS prediction on observation; display only, no calibration',
        density_definition=f'2D histogram counts, {GRID_BINS} bins per axis'), indent=2), encoding='utf-8')
    print('图片及指标保存至:', output)

if __name__ == '__main__':
    main()
