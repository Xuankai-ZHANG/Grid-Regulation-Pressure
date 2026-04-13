# -*- coding: utf-8 -*-
"""
figure_maps.py — Three Maps
  Map 1: Load Share (peak_amplifier)      
  Map 3: Per Capita Regulation Burden   

Data source: data/grid_data.xlsx (sheet: mesh_main)
Output:     output/map_load_share.png
          output/map_ramp_rate.png
          output/map_per_burden.png
"""
import sys, os, warnings
import io as _io
if sys.platform == 'win32':
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ─── Path configuration ───────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(_DIR, 'data', 'grid_data.xlsx')
GEOJSON     = os.path.join(_DIR, 'data', 'kanto_boundary.geojson')
OUTPUT_DIR  = os.path.join(_DIR, 'output_plot')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(_DIR, 'utils'))
from map_utils import (
    CMAP_BLUE_GREEN_12,
    prepare_brick_coords, draw_map
)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════
print('Loading data...')
df = pd.read_excel(DATA_FILE, sheet_name='mesh_main')
# Ensure dataframe contains grid_code column
if 'grid_code' not in df.columns and 'mesh_code' in df.columns:
    df['grid_code'] = df['mesh_code']
print(f'  {len(df)} grids loaded')

# ── Calculate per captia regulation burden ───────
df['per_burden'] = df['pressure_share'] / df['pop_share']

# ── Coordinate calculation ───────────
print('Computing brick coordinates...')
df, gdf, lon_min, lat_min, lon_step, lat_step, bnd_xlim, bnd_ylim = \
    prepare_brick_coords(df, GEOJSON)

# ════════════════════════════════════════════════════════
# 2. Draw and save three maps
# ════════════════════════════════════════════════════════
maps = [
    ('peak_concent', CMAP_BLUE_GREEN_12, 'map_peak_concentration.png',   'Map 1: Peak Concentration'),
    ('ramp_contribution', CMAP_BLUE_GREEN_12, 'map_ramp_contribution.png', 'Map 2: Ramp Contribution'),
    ('per_burden',     CMAP_BLUE_GREEN_12, 'map_per_captia_regulation_burden.png',   'Map 3: Per Capita Regulation Burden'),
]

for value_col, cmap_12, fname, label in maps:
    print(f'Drawing {label}...')
    fig = draw_map(
        df, value_col, cmap_12,
        gdf, lon_min, lat_min, lon_step, lat_step,
        bnd_xlim, bnd_ylim,
        figsize=(16, 12)
    )
    out_path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out_path}')

print('\nDone. All maps saved to:', OUTPUT_DIR)
