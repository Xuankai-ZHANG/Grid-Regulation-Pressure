# -*- coding: utf-8 -*-
"""
plot_partA.py — Scatter Plot + Box Plot
  Fig A1: Peak Concentration vs. Ramp Contribution scatter (colored by base_share)
  Fig A2: Boxplot by zone (Peak Concentration / Ramp Contribution)

Data source: data/grid_data.xlsx (sheet: mesh_main)
Output:     output/A1.png
          output/A2.png
"""
import sys, os, warnings
import io as _io
if sys.platform == 'win32':
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(_DIR, 'data', 'grid_data.xlsx')
OUTPUT_DIR = os.path.join(_DIR, 'output_plot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════
print('Loading data...')
df = pd.read_excel(DATA_FILE, sheet_name='mesh_main')
print(f'  {len(df)} grids loaded')

# ── Normalization ────────────────────────────────────
x_min, x_max = df['peak_concent'].min(), df['peak_concent'].max()
df['x'] = (df['peak_concent'] - x_min) / (x_max - x_min)

log_ramp = np.log10(df['ramp_contribution'].clip(lower=1e-8))
r_min, r_max = log_ramp.min(), log_ramp.max()
df['y'] = (log_ramp - r_min) / (r_max - r_min)

df['base_share_rank'] = df['base_share'].rank(method='average') / len(df)

# ════════════════════════════════════════════════════════
# 2. Scatter plot
# ════════════════════════════════════════════════════════
ZONE_TYPES   = ['Core Zone', 'Commuter Belt', 'Outer Zone']
ZONE_MARKERS = ['*', 'o', '^']

sample = df.sample(n=min(3000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(13, 4), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

scatter_handles = []
for zone, marker in zip(ZONE_TYPES, ZONE_MARKERS):
    z = sample[sample['zone_type'] == zone]
    sc = ax.scatter(
        z['x'], z['y'],
        c=z['base_share_rank'], cmap='RdYlGn_r',
        s=35, marker=marker, alpha=0.60,
        linewidths=0, vmin=0, vmax=1, label=zone
    )
    scatter_handles.append(sc)

cbar = plt.colorbar(scatter_handles[0], ax=ax, pad=0.02)
cbar.ax.yaxis.set_tick_params(color='black')
cbar.outline.set_edgecolor('black')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black', fontsize=12)
cbar.set_label('Mean Load Share', color='black', fontsize=12)
cbar.set_ticks([0, 0.5, 1.0])
cbar.ax.set_yticklabels(['Low', 'Mid', 'High'], color='black', fontsize=12)

legend_elements = [
    Line2D([0], [0], marker='*', color='none', markerfacecolor='#666666',
           markersize=np.sqrt(35) * 1.4, label='Core Zone'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#666666',
           markersize=np.sqrt(35) * 1.4, label='Commuter Belt'),
    Line2D([0], [0], marker='^', color='none', markerfacecolor='#666666',
           markersize=np.sqrt(35) * 1.4, label='Outer Zone'),
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False,
          title_fontsize=12, fontsize=12)

ax.xaxis.grid(False)
ax.yaxis.grid(True, linestyle='--', color='#e0e0e0', linewidth=0.5, alpha=0.6)
ax.set_xlabel('Peak Concentration', color='#333333', fontsize=15)
ax.set_ylabel('Ramp Contribution', color='#333333', fontsize=15)
ax.tick_params(colors='#333333', labelsize=12)
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')
    spine.set_linewidth(2.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0.35, 0.88)
ax.set_ylim(0.27, 0.92)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'A1.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved scatter: {out_path}')

# ════════════════════════════════════════════════════════
# 3. Box plot
# ════════════════════════════════════════════════════════
VIOL_COLORS = ['#0BA8A0', '#D88040', '#4080C0']
VIOL_LABELS = ['Core Zone', 'Commuter Belt', 'Outer Zone']


def draw_boxplot_subplot(ax, zone_data_list, colors, labels, ylabel):
    positions = [i * 2 + 1 for i in range(len(zone_data_list))]
    xlim_right = positions[-1] + 1.2
    bp = ax.boxplot(
        zone_data_list, positions=positions, widths=1.0,
        patch_artist=True, showfliers=False,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=2.0, color='white'),
    )
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i]); patch.set_alpha(0.7); patch.set_edgecolor('none')
    for i, w in enumerate(bp['whiskers']):
        w.set(color=colors[i // 2], linewidth=1.2, alpha=0.8)
    for i, c in enumerate(bp['caps']):
        c.set(color=colors[i // 2], linewidth=1.2, alpha=0.8)

    sample_counts = [10, 70, 30]
    markers = ['*', 'o', '^']
    marker_sizes = [50, 25, 25]
    for i, data in enumerate(zone_data_list):
        if len(data) == 0:
            continue
        n_s = min(sample_counts[i], len(data))
        s_data = np.random.choice(data, size=n_s, replace=False)
        x_jitter = positions[i] + np.random.normal(0, 0.12, n_s)
        ax.scatter(x_jitter, s_data, c=colors[i], s=marker_sizes[i],
                   marker=markers[i], alpha=1, edgecolor='white',
                   linewidth=0.5, zorder=5)

    for i, vals in enumerate(zone_data_list):
        if len(vals) == 0:
            continue
        mean_v = np.mean(vals)
        ax.text(positions[i], mean_v - 0.2, f'mean={mean_v:.3f}',
                ha='center', va='top', fontsize=12, color='#222222', zorder=15,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0))

    ax.set_xlim(-0.2, xlim_right)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(labelsize=12)
    ax.set_facecolor('white')
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.6, alpha=0.7, zorder=0)
    ax.xaxis.grid(False)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for sp in ax.spines.values():
        sp.set_color('#333333'); sp.set_linewidth(2.0)


zone_masks = [df['zone_type'] == z for z in ZONE_TYPES]
zone_data_peak = [df.loc[m, 'x'].dropna().values for m in zone_masks]
zone_data_ramp = [df.loc[m, 'y'].dropna().values for m in zone_masks]

fig_box, (ax_bp, ax_br) = plt.subplots(1, 2, figsize=(13, 3), dpi=300)
fig_box.patch.set_facecolor('white')
draw_boxplot_subplot(ax_bp, zone_data_peak, VIOL_COLORS, VIOL_LABELS, 'Peak Concentration')
draw_boxplot_subplot(ax_br, zone_data_ramp, VIOL_COLORS, VIOL_LABELS, 'Ramp Contribution')
plt.tight_layout()

out_box = os.path.join(OUTPUT_DIR, 'A2.png')
fig_box.savefig(out_box, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_box)
print(f'Saved boxplot: {out_box}')

print('\nDone.')