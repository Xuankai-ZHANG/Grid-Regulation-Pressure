# -*- coding: utf-8 -*-
"""
map_utils.py — 共享地图绘制工具
整合自 brick_utils.py, plot_utils.py, plot_f1_split.py
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────
# 全局砖块尺寸
# ─────────────────────────────────────────────
BRICK_WIDTH  = 1.5
BRICK_HEIGHT = 1.0

# ─────────────────────────────────────────────
# 配色方案（与原脚本完全一致）
# f1_split / figure_maps 通用配色（蓝绿渐变）
# ─────────────────────────────────────────────
CMAP_F1 = LinearSegmentedColormap.from_list(
    'peak_concentration',
    ['#4A5BA6', '#6B8FC4', '#A8C5E0', '#D5E8D5',
     '#B8D5C0', '#99C998', '#6B9F6B', '#4A6B4A'],
    N=256
)
CMAP_F1_12 = ListedColormap([
    '#4A5BA6', '#5B6FB8', '#6B8FC4', '#7BA4D0',
    '#A8C5E0', '#C5D5E8', '#B8D5C0', '#99C998',
    '#7FB87F', '#6B9F6B', '#5A8D5A', '#4A6B4A'
])

# f3 per_burden 配色（蓝→绿渐变，12色离散版）
CMAP_F3 = LinearSegmentedColormap.from_list(
    'responsibility_br',
    ['#4A5BA6', '#5B6FB8', '#6B8FC4', '#7BA4D0',
     '#A8C5E0', '#C5D5E8', '#B8D5C0', '#99C998',
     '#7FB87F', '#6B9F6B', '#5A8D5A', '#4A6B4A'],
    N=256
)
CMAP_F3_12 = ListedColormap([CMAP_F3(i / 11) for i in range(12)])


# ═══════════════════════════════════════════════════════
# 坐标转换函数（来自 brick_utils.py）
# ═══════════════════════════════════════════════════════
def mesh_to_latlon(mesh_code):
    """JIS 3rd Mesh Code → (lat, lon) 中心点坐标"""
    s = str(int(mesh_code))
    lat = int(s[0:2]) / 1.5 * 3600
    lon = (int(s[2:4]) + 100) * 3600
    lat += int(s[4]) * 5 * 60
    lon += int(s[5]) * 7.5 * 60
    lat += int(s[6]) * 30
    lon += int(s[7]) * 45
    return (lat + 15) / 3600, (lon + 22.5) / 3600


def estimate_mesh_grid_spacing(points, percentile=5):
    """估算 mesh 网格的经纬度间距"""
    lons = np.sort(points[:, 0])
    lats = np.sort(points[:, 1])
    lon_diffs = np.diff(lons); lon_diffs = lon_diffs[lon_diffs > 1e-10]
    lat_diffs = np.diff(lats); lat_diffs = lat_diffs[lat_diffs > 1e-10]
    lon_step = np.percentile(lon_diffs, percentile) if len(lon_diffs) > 0 else 0.00833
    lat_step = np.percentile(lat_diffs, percentile) if len(lat_diffs) > 0 else 0.0125
    return lon_step, lat_step


def latlon_to_grid(lon, lat, lon_min, lat_min, lon_step, lat_step):
    """经纬度 → 网格索引"""
    return (lon - lon_min) / lon_step, (lat - lat_min) / lat_step


def grid_to_brick(grid_i, grid_j, brick_width=BRICK_WIDTH, brick_height=BRICK_HEIGHT):
    """网格索引 → 砖块坐标"""
    return grid_i * brick_width, grid_j * brick_height


def transform_boundary_to_brick(geometry, lon_min, lat_min, lon_step, lat_step,
                                 brick_width=BRICK_WIDTH, brick_height=BRICK_HEIGHT):
    """地理边界几何 → 砖块坐标列表"""
    brick_coords = []
    if geometry.geom_type == 'Polygon':
        coords = np.array(geometry.exterior.coords)
        gi, gj = latlon_to_grid(coords[:, 0], coords[:, 1], lon_min, lat_min, lon_step, lat_step)
        bx, by = grid_to_brick(gi, gj, brick_width, brick_height)
        brick_coords.append((bx + brick_width / 2, by + brick_height / 2))
    elif geometry.geom_type == 'MultiPolygon':
        for poly in geometry.geoms:
            coords = np.array(poly.exterior.coords)
            gi, gj = latlon_to_grid(coords[:, 0], coords[:, 1], lon_min, lat_min, lon_step, lat_step)
            bx, by = grid_to_brick(gi, gj, brick_width, brick_height)
            brick_coords.append((bx + brick_width / 2, by + brick_height / 2))
    return brick_coords


def get_brick_bounds(df_clean, brick_width=BRICK_WIDTH, brick_height=BRICK_HEIGHT):
    """获取砖块坐标的显示范围（含边距）"""
    mx, my = df_clean['brick_x'].min(), df_clean['brick_y'].min()
    xx, yy = df_clean['brick_x'].max(), df_clean['brick_y'].max()
    margin_x, margin_y = brick_width * 2, brick_height * 2
    return (mx - margin_x, xx + brick_width + margin_x), (my - margin_y, yy + brick_height + margin_y)


# ═══════════════════════════════════════════════════════
# 一步完成坐标计算的封装函数
# ═══════════════════════════════════════════════════════
def prepare_brick_coords(df, geojson_path):
    """
    从 mesh_code 一步计算 brick_x, brick_y，并返回边界 GeoDataFrame。

    Parameters
    ----------
    df : pd.DataFrame，含 mesh_code 列
    geojson_path : str，边界 GeoJSON 路径

    Returns
    -------
    df : 含 lat, lon, brick_x, brick_y 的 DataFrame（原地修改副本）
    gdf : GeoDataFrame 边界，失败时为 None
    lon_min, lat_min, lon_step, lat_step : 坐标系参数
    bnd_xlim, bnd_ylim : 边界砖块坐标范围（供 style_map_ax 使用）
    """
    df = df.copy()
    df['mesh_code'] = df['mesh_code'].astype(str)
    latlon = df['mesh_code'].apply(lambda x: pd.Series(mesh_to_latlon(int(x))))
    df['lat'] = latlon[0]
    df['lon'] = latlon[1]

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception:
        gdf = None

    points = df[['lon', 'lat']].values
    lon_min, lat_min = points[:, 0].min(), points[:, 1].min()

    if gdf is not None:
        for geom in gdf.geometry:
            b = geom.bounds
            lon_min = min(lon_min, b[0])
            lat_min = min(lat_min, b[1])

    lon_step, lat_step = estimate_mesh_grid_spacing(points, percentile=5)
    gi, gj = latlon_to_grid(df['lon'].values, df['lat'].values, lon_min, lat_min, lon_step, lat_step)
    df['brick_x'], df['brick_y'] = grid_to_brick(gi, gj)

    # 边界砖块范围
    bnd_xlim, bnd_ylim = None, None
    if gdf is not None:
        all_x, all_y = [], []
        for geom in gdf.geometry:
            for bx, by in transform_boundary_to_brick(geom, lon_min, lat_min, lon_step, lat_step):
                all_x.extend(bx); all_y.extend(by)
        if all_x:
            margin = max(BRICK_WIDTH, BRICK_HEIGHT) * 2
            bnd_xlim = (min(all_x) - margin, max(all_x) + margin)
            bnd_ylim = (min(all_y) - margin, max(all_y) + margin)

    return df, gdf, lon_min, lat_min, lon_step, lat_step, bnd_xlim, bnd_ylim


# ═══════════════════════════════════════════════════════
# 绘图函数（来自 plot_f1_split.py）
# ═══════════════════════════════════════════════════════
def draw_brick_layer(ax, df, value_col, cmap_12, use_discrete=True):
    """
    绘制砖块层。use_discrete=True 时使用 12 色离散方案（默认）。
    cmap_12 应为 ListedColormap（12色），cmap 应为连续版。
    """
    df_plot = df[[value_col, 'brick_x', 'brick_y']].dropna()
    if use_discrete:
        values = df_plot[value_col].values
        sorted_indices = np.argsort(values)
        ranks = np.empty_like(sorted_indices, dtype=int)
        ranks[sorted_indices] = np.arange(len(sorted_indices))
        disc_idx = (ranks * 12 // len(df_plot)).clip(0, 11)
        for i, (_, row) in enumerate(df_plot.iterrows()):
            ax.add_patch(Rectangle(
                (row['brick_x'], row['brick_y']), BRICK_WIDTH, BRICK_HEIGHT,
                facecolor=cmap_12.colors[disc_idx[i]],
                edgecolor='none', alpha=0.9, zorder=5
            ))
    else:
        norm = Normalize(vmin=df_plot[value_col].min(), vmax=df_plot[value_col].max())
        for _, row in df_plot.iterrows():
            ax.add_patch(Rectangle(
                (row['brick_x'], row['brick_y']), BRICK_WIDTH, BRICK_HEIGHT,
                facecolor=cmap_12(norm(row[value_col])),
                edgecolor='none', alpha=0.9, zorder=5
            ))


def draw_admin_boundary(ax, gdf, lon_min, lat_min, lon_step, lat_step):
    """绘制行政边界（灰色线条）"""
    if gdf is None:
        return
    for geom in gdf.geometry:
        for bx, by in transform_boundary_to_brick(geom, lon_min, lat_min, lon_step, lat_step):
            ax.plot(bx, by, color='#666666', linewidth=1.0, alpha=0.85, zorder=10)


def style_map_ax(ax, df, bnd_xlim=None, bnd_ylim=None):
    """地图轴样式统一设置"""
    xlim, ylim = get_brick_bounds(df)
    if bnd_xlim:
        xlim = (min(xlim[0], bnd_xlim[0]), max(xlim[1], bnd_xlim[1]))
    if bnd_ylim:
        ylim = (min(ylim[0], bnd_ylim[0]), max(ylim[1], bnd_ylim[1]))
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('white')


def draw_map(df, value_col, cmap_12, gdf, lon_min, lat_min, lon_step, lat_step,
             bnd_xlim=None, bnd_ylim=None, figsize=(16, 12)):
    """
    一步完成地图绘制：砖块层 + 行政边界 + 轴样式。
    返回 fig。
    """
    fig, ax = plt.subplots(figsize=figsize)
    draw_brick_layer(ax, df, value_col, cmap_12)
    draw_admin_boundary(ax, gdf, lon_min, lat_min, lon_step, lat_step)
    style_map_ax(ax, df, bnd_xlim, bnd_ylim)
    return fig
