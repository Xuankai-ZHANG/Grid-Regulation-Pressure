# -*- coding: utf-8 -*-
"""
plot_partB.py — Policy Fairness Analysis 

Figure B1:
  Gini coefficient dual-axis chart (line + bar)

Figure B2:
  KDE density plots of per-capita regulation burden by zone (with pie insets)

Figure B3:
  3D heatmap — Income × Pressure quintile, by zone (before policy)

Figure B4:
  3D heatmap — Income × Pressure quintile, by zone (after 25% policy)

Figure B5:
  3D heatmap — Income × Pressure quintile, by implementation rate

Data source: data/grid_data.xlsx (sheets: mesh_main, policy_rates)
Output:     output/B1.png
          output/B2.png
          output/B3.png
          output/B4.png
          output/B5.png
"""


import sys, os, warnings
import io as _io
if sys.platform == 'win32':
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import gaussian_kde
from datetime import datetime
warnings.filterwarnings('ignore')

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(_DIR, 'data', 'grid_data.xlsx')
OUTPUT_DIR = os.path.join(_DIR, 'output_plot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ════════════════════════════════════════════════════════
# 0. Global style constants
# ════════════════════════════════════════════════════════
FC = dict(title=12, label=16, tick=16, annot=12, small=12, lw=1.5, outer_edge_lw=4.0)
PAL = dict(line='#E64B35', bar='#4DBBD5', base='#333333',
           before='#333333', rate10='#4DBBD5', rate25='#E64B35')
CMAP_3D = 'RdBu_r'
ZONE_COLORS = ['#0BA8A0', '#D88040', '#4080C0']   # Core / Commuter / Outer
PIE_GRAY    = '#cccccc'
ZONES       = ['Core Zone', 'Commuter Belt', 'Outer Zone']
ZONES_SHORT = ['Core', 'Commuter', 'Outer']


WEALTH_TICK = ['I1', 'I2', 'I3', 'I4', 'I5']
RESP        = ['P1', 'P2', 'P3', 'P4', 'P5']

# ════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════
print('Loading data...')
df = pd.read_excel(DATA_FILE, sheet_name='mesh_main')
df_rates = pd.read_excel(DATA_FILE, sheet_name='policy_rates')
print(f'  {len(df)} grids, {len(df_rates)} policy rate rows')

# ── Inline calculations────
df['per_burden'] = df['pressure_share'] / df['pop_share']   # pb (before)
df['delta']      = df['pressure_share'] - df['base_share']   # fairness gap

for rate, tag in [(0.10, '10'), (0.25, '25')]:
    pci_r = df['pressure_share'] * (1 - rate) + df['pop_share'] * rate
    pci_r = pci_r / pci_r.sum()
    df[f'pci_{tag}']  = pci_r
    df[f'pb_{tag}']  = pci_r / df['pop_share']
    df[f'da_{tag}']  = pci_r - df['base_share']

pci_zone = {
    'base': df.groupby('zone_type')['pressure_share'].sum(),
    '10':   df.groupby('zone_type')['pci_10'].sum(),
    '25':   df.groupby('zone_type')['pci_25'].sum(),
}

# ════════════════════════════════════════════════════════
# 2. Figure B1 — Gini dual-axis chart
# ════════════════════════════════════════════════════════
print('\nDrawing Figure B1: Gini dual-axis chart...')
rates   = df_rates['rate'].values
gini_a  = df_rates['gini_after'].values
gini0   = float(df_rates['gini_before'].iloc[0])
red_pct = df_rates['reduction_pct'].values
x       = np.arange(len(rates))
xlabels = [f'{r:.0%}' for r in rates]

fig1, ax1 = plt.subplots(figsize=(9, 9), dpi=300)
ax2 = ax1.twinx()

bw = 0.55
bars = ax2.bar(x, red_pct, bw, color=PAL['bar'], alpha=0.25, edgecolor='none', zorder=1)
for bar, val in zip(bars, red_pct):
    ax2.text(bar.get_x() + bw / 2, bar.get_height() + 0.6,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=18, color='black')

ax1.axhline(gini0, color=PAL['base'], ls='--', lw=FC['lw'],
            label=f'Baseline (Gini = {gini0:.4f})', zorder=2)
ax1.plot(x, gini_a, 'o-', color=PAL['line'], lw=2.5, markersize=5,
         markerfacecolor='white', markeredgewidth=1.5,
         label='Gini After Reallocation', zorder=5)

ax1.set_xticks(x); ax1.set_xticklabels(xlabels, fontsize=18)
ax1.set_xlabel('Implementation Rate', fontsize=22)
ax1.set_ylabel('Gini Coefficient', fontsize=22, color='black')
ax2.set_ylabel('Gini Reduction (%)', fontsize=22, color='black', rotation=270, labelpad=12)
ax1.tick_params(axis='y', labelsize=18, colors='black')
ax2.tick_params(axis='y', labelsize=18, colors='black')
ax1.tick_params(axis='x', labelsize=18)
ax1.set_ylim(0, gini0 * 1.18)
ax2.set_ylim(0, max(red_pct) * 1.28)

for sp_name, sp in ax1.spines.items():
    if sp_name in ('top', 'right'): sp.set_visible(False)
    else: sp.set_color('#333333'); sp.set_linewidth(2.0)
ax2.spines['right'].set_color('#333333'); ax2.spines['right'].set_linewidth(2.0)
for sp_name in ('top', 'left', 'bottom'): ax2.spines[sp_name].set_visible(False)

ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0); ax2.grid(False)

h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h1, l1, loc='lower center', bbox_to_anchor=(0.5, 0.88),
           frameon=False, ncol=2, fontsize=18,
           labelspacing=0.3, handletextpad=0.4, columnspacing=0.6)

fig1.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.15)
out5a = os.path.join(OUTPUT_DIR, 'B1.png')
fig1.savefig(out5a, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f'  Saved: {out5a}')

# ════════════════════════════════════════════════════════
# 3. Figure B2 — KDE density plots (with pie insets)
# ════════════════════════════════════════════════════════
print('Drawing Figure B2: KDE density plots...')
PIE_S    = 0.24
PIE_GAP  = 0.005
PIE_KEYS = [('base', '0%'), ('10', '10%'), ('25', '25%')]
PIE_LCOLORS = [PAL['before'], PAL['rate10'], PAL['rate25']]

fig2 = plt.figure(figsize=(22, 7), dpi=300)
gs = gridspec.GridSpec(1, 3, wspace=0.2)
xticks = np.arange(0, 11, 2)

for idx, zone in enumerate(ZONES):
    ax_kde = fig2.add_subplot(gs[idx])
    sub = df[df['zone_type'] == zone]

    if zone in ['Commuter Belt', 'Outer Zone']:
        x_lim = (0, 10)
    else:
        lo = float(sub['per_burden'].quantile(0.01))
        hi = float(sub[['per_burden', 'pb_10', 'pb_25']].quantile(0.99).max())
        x_lim = (max(0, lo - 0.05), hi + 0.05)
    xx = np.linspace(x_lim[0], x_lim[1], 600)

    def _kde(series):
        vals = series.clip(x_lim[0], x_lim[1]).dropna().values
        return gaussian_kde(vals, bw_method='scott')(xx) if len(vals) >= 2 else np.zeros_like(xx)

    y0, y10, y25 = _kde(sub['per_burden']), _kde(sub['pb_10']), _kde(sub['pb_25'])

    for y_vals, color in [(y0, PIE_LCOLORS[0]), (y10, PIE_LCOLORS[1]), (y25, PIE_LCOLORS[2])]:
        ax_kde.fill_between(xx, y_vals, alpha=0.10, color=color)
    ax_kde.plot(xx, y0, color=PIE_LCOLORS[0], lw=3, ls='-', zorder=4)
    ax_kde.plot(xx, y10, color=PIE_LCOLORS[1], lw=3, ls='--', zorder=4)
    ax_kde.plot(xx, y25, color=PIE_LCOLORS[2], lw=3, ls='-.', zorder=4)

    for col, color in [('per_burden', PIE_LCOLORS[0]), ('pb_10', PIE_LCOLORS[1]),
                       ('pb_25', PIE_LCOLORS[2])]:
        ax_kde.plot(float(sub[col].mean()), 0, marker='^', color=color, markersize=15, zorder=5)

    ax_kde.set_title(ZONES[idx], fontsize=22, fontweight=2)
    ax_kde.set_xlabel('Per capita Regulation Burden', fontsize=22)
    if idx == 0:
        ax_kde.set_ylabel('Density', fontsize=22)
    ax_kde.set_xlim(x_lim)
    if zone in ['Commuter Belt', 'Outer Zone']:
        ax_kde.set_xticks(xticks)
    ax_kde.tick_params(labelsize=18)
    for spine in ax_kde.spines.values():
        spine.set_visible(True); spine.set_linewidth(2.0)
    ax_kde.grid(axis='x', linestyle='--', alpha=0.22, zorder=0)

    # Pie chart insets
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    for pie_idx, (key, label) in enumerate(PIE_KEYS):
        pie_x = 0.28 + pie_idx * (PIE_S + PIE_GAP)
        pie_y = 0.6
        pie_ax = inset_axes(ax_kde, width=f'{PIE_S*100}%', height=f'{PIE_S*100}%',
                            loc='lower left',
                            bbox_to_anchor=(pie_x, pie_y, 1, 1),
                            bbox_transform=ax_kde.transAxes, borderpad=0)
        cur_val   = pci_zone[key].get(zone, 0)
        other_val = sum(pci_zone[key].get(z, 0) for z in ZONES if z != zone)
        wedges, _ = pie_ax.pie(
            [cur_val, other_val],
            colors=[ZONE_COLORS[idx], PIE_GRAY],
            startangle=90, counterclock=False,
            wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
        )
        pie_ax.set_aspect('equal')
        total = cur_val + other_val
        if total > 0:
            pie_ax.text(0, -1.3, f'{cur_val/total*100:.0f}%',
                        ha='center', va='center', fontsize=18, color=PIE_LCOLORS[pie_idx])
        pie_ax.set_xticks([]); pie_ax.set_yticks([])

fig2.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.15)
out5b = os.path.join(OUTPUT_DIR, 'B2.png')
fig2.savefig(out5b, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f'  Saved: {out5b}')

# ════════════════════════════════════════════════════════
# 4. Figure B3-B5 — 3D Heatmaps
# ════════════════════════════════════════════════════════
def _piv_zone(zone, col):
    """income_quantile × pressure_quantile"""
    sub = df[df['zone_type'] == zone]
    return (sub.groupby(['income_quantile', 'pressure_quantile'], observed=True)[col]
            .sum().unstack('pressure_quantile')[RESP])


def _piv_all(col):
    """All zones: income_quantile × pressure_quantile"""
    return (df.groupby(['income_quantile', 'pressure_quantile'], observed=True)[col]
            .sum().unstack('pressure_quantile')[RESP])


def _draw_3d(layers, y_labels, x_labels, z_tick_labels,
             ylabel, xlabel, zlabel, fname_stem):
    """Generic 3D stacked heatmap """
    NR, NC, NL = len(y_labels), len(x_labels), len(layers)
    Z_FULL  = float(NC)
    LAYER_Z = [Z_FULL * i / (NL - 1) for i in range(NL)]

    fig = plt.figure(figsize=(10, 8), dpi=300)
    fig.subplots_adjust(left=0.12, right=0.85, bottom=0.12, top=0.92)
    ax = fig.add_subplot(111, projection='3d')

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
        axis.pane.set_linewidth(0)
        axis.pane.set_visible(False)

    cmap_3d = plt.get_cmap(CMAP_3D)
    shared_norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    layer_vms = []
    for _, piv, _ in layers:
        vals = piv.values.flatten()
        vm = float(np.max(np.abs(vals[~np.isnan(vals)])))
        layer_vms.append(vm)
    vm_min, vm_max = min(layer_vms), max(layer_vms)
    alpha_min, alpha_max = 0.60, 1.0
    alpha_values = (
        [alpha_min + (vm - vm_min) / (vm_max - vm_min) * (alpha_max - alpha_min)
         for vm in layer_vms]
        if vm_max > vm_min else [0.80] * NL
    )

    for li, (_, piv, _lbl) in enumerate(layers):
        z_pos = LAYER_Z[li]
        alpha = alpha_values[li]
        vals  = piv.values.flatten()
        vm    = float(np.max(np.abs(vals[~np.isnan(vals)])))
        piv_n = piv / vm

        verts, fcs = [], []
        for i in range(NR):
            for j in range(NC):
                v = piv_n.iloc[i, j]
                clr = cmap_3d(shared_norm(v)) if not np.isnan(v) else (0.88, 0.88, 0.88, 1)
                verts.append([(j, i, z_pos), (j+1, i, z_pos),
                              (j+1, i+1, z_pos), (j, i+1, z_pos)])
                fcs.append((clr[0], clr[1], clr[2]))

        ax.add_collection3d(
            Poly3DCollection(verts, facecolors=fcs, edgecolors='white',
                             linewidth=1.2, alpha=alpha, zsort='average')
        )

    ax.set_xlim(0, NC); ax.set_ylim(0, NR); ax.set_zlim(0, Z_FULL)
    ax.set_xticks(np.arange(0.5, NC)); ax.set_xticklabels(x_labels, fontsize=FC['tick'])
    ax.set_yticks(np.arange(0.5, NR)); ax.set_yticklabels(y_labels, fontsize=FC['tick'])
    ax.set_zticks(LAYER_Z);            ax.set_zticklabels(z_tick_labels, fontsize=FC['tick'])
    ax.tick_params(axis='z', pad=4)
    ax.tick_params(axis='x', pad=0.5)
    ax.tick_params(axis='y', pad=0.5)

    ax.set_xlabel(xlabel, fontsize=FC['label'], labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FC['label'], labelpad=8)
    ax.text(-0.65, NR + 0.5, Z_FULL * 1.12, zlabel, fontsize=FC['label'],
            ha='left', va='bottom', rotation=0)

    ax.plot([NC, NC], [0, 0], [0, Z_FULL], color='#888888', lw=1.2, linestyle='--', zorder=5)
    ax.plot([0, 0], [NR, NR], [0, Z_FULL], color='#888888', lw=1.2, linestyle='--', zorder=5)

    ax.view_init(elev=16, azim=225)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(0)
        axis._axinfo['grid']['color'] = (1, 1, 1, 0)
        axis._axinfo['tick']['size']  = 0
        axis._axinfo['tick']['width'] = 0

    out_path = os.path.join(OUTPUT_DIR, f'{fname_stem}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out_path}')


print('\nDrawing Figure B3: 3D before policy...')
_draw_3d(
    layers=[
        (0, _piv_zone('Core Zone',    'delta'), 'Core'),
        (1, _piv_zone('Commuter Belt','delta'), 'Commuter'),
        (2, _piv_zone('Outer Zone',   'delta'), 'Outer'),
    ],
    y_labels=WEALTH_TICK, x_labels=RESP,
    z_tick_labels=['Core\nZone', 'Commuter\nBelt', 'Outer\nZone'],
    ylabel='Income Quantile', xlabel='Pressure Quantile', zlabel='Zone Type',
    fname_stem='B3',
)

print('Drawing Figure B4: 3D after 25% policy...')
_draw_3d(
    layers=[
        (0, _piv_zone('Core Zone',    'da_25'), 'Core'),
        (1, _piv_zone('Commuter Belt','da_25'), 'Commuter'),
        (2, _piv_zone('Outer Zone',   'da_25'), 'Outer'),
    ],
    y_labels=WEALTH_TICK, x_labels=RESP,
    z_tick_labels=['Core\nZone', 'Commuter\nBelt', 'Outer\nZone'],
    ylabel='Income Quantile', xlabel='Pressure Quantile', zlabel='Zone Type',
    fname_stem='B4',
)

print('Drawing Figure B5: 3D by implementation rate...')
_draw_3d(
    layers=[
        (0, _piv_all('delta'), 'Original\n(0%)'),
        (1, _piv_all('da_10'), 'Policy\n10%'),
        (2, _piv_all('da_25'), 'Policy\n25%'),
    ],
    y_labels=WEALTH_TICK, x_labels=RESP,
    z_tick_labels=['0%', '10%', '25%'],
    ylabel='Income Quantile', xlabel='Pressure Quantile', zlabel='Implementation Rate',
    fname_stem='B5',
)

print('\nDone. All files saved to:', OUTPUT_DIR)