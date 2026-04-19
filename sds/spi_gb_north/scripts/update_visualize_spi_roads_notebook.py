import json
from pathlib import Path


NOTEBOOK_PATH = Path(
    r"C:\Users\AtifA\Desktop\Wasif SDS Project\sds\sds\spi_gb_north\visualize_spi_roads.ipynb"
)


def lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    nb["cells"] = [
        md_cell(
            """
# SPI Scenic Map with Editable Style Controls

This notebook is set up so you can edit the map styling from a single config cell.

Change these easily near the top:
- road thickness and color
- tehsil boundary thickness, color, and dotted pattern
- major region colors and label positions
- SPI palette colors
- figure size, legend, colorbar, margins, and title visibility
"""
        ),
        code_cell(
            """
# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import geopandas as gpd
import rasterio

print('Libraries loaded successfully.')
print(f'  matplotlib : {matplotlib.__version__}')
print(f'  geopandas  : {gpd.__version__}')
print(f'  rasterio   : {rasterio.__version__}')
"""
        ),
        md_cell(
            """
## Editable Config

This is the only cell you should usually need to edit.
"""
        ),
        code_cell(
            """
CONFIG = {
    'paths': {
        'base_dir': Path('.'),
        'processed_dir': Path('data/processed'),
        'interim_dir': Path('data/interim'),
        'raw_dir': Path('data/raw'),
        'outputs_dir': Path('outputs'),
        'spi_raster': Path('data/processed/spi_index.tif'),
        'tehsils_file': Path('data/interim/tehsils_aoi_32643.gpkg'),
        'roads_file': Path('data/interim/roads_all_32643.gpkg'),
        'aoi_file': Path('data/interim/aoi_boundary_north_kpk_gb_ajk_32643.gpkg'),
        'adm1_file': Path('data/raw/admin_boundaries/geoBoundaries-PAK-ADM1.geojson'),
        'output_png': Path('outputs/spi_major_regions_roads_map.png'),
    },
    'regions': {
        'region_names': ['Gilgit-Baltistan', 'Azad Kashmir', 'Khyber Pakhtunkhwa'],
        'region_labels': {
            'Gilgit-Baltistan': 'Gilgit-Baltistan',
            'Azad Kashmir': 'Kashmir',
            'Khyber Pakhtunkhwa': 'Northern KPK',
        },
        'region_outline': {
            'Gilgit-Baltistan': {'edgecolor': '#2563eb', 'linewidth': 2.0, 'zorder': 7},
            'Azad Kashmir': {'edgecolor': '#7c3aed', 'linewidth': 2.0, 'zorder': 7},
            'Khyber Pakhtunkhwa': {'edgecolor': '#0f766e', 'linewidth': 2.0, 'zorder': 7},
        },
        'label_position_factors': {
            'Gilgit-Baltistan': {'x': 1.12, 'y': 0.86},
            'Azad Kashmir': {'x': 1.10, 'y': 0.17},
            'Khyber Pakhtunkhwa': {'x': -0.13, 'y': 0.60},
        },
        'label_style': {
            'fontsize': 14,
            'text_color': 'black',
            'fontweight': 'bold',
            'box_facecolor': 'white',
            'box_alpha': 0.96,
            'box_pad': 0.4,
            'box_linewidth': 1.5,
            'leader_linewidth': 1.4,
        },
    },
    'roads': {
        'classes': [
            'motorway', 'motorway_link',
            'trunk', 'trunk_link',
            'primary', 'primary_link',
            'secondary', 'secondary_link',
        ],
        'style': {
            'motorway': {'color': '#d97706', 'lw': 1.10, 'alpha': 0.78, 'zorder': 4},
            'motorway_link': {'color': '#d97706', 'lw': 0.90, 'alpha': 0.78, 'zorder': 4},
            'trunk': {'color': '#d97706', 'lw': 0.95, 'alpha': 0.78, 'zorder': 4},
            'trunk_link': {'color': '#d97706', 'lw': 0.80, 'alpha': 0.78, 'zorder': 4},
            'primary': {'color': '#d97706', 'lw': 0.80, 'alpha': 0.78, 'zorder': 4},
            'primary_link': {'color': '#d97706', 'lw': 0.68, 'alpha': 0.78, 'zorder': 4},
            'secondary': {'color': '#d97706', 'lw': 0.66, 'alpha': 0.78, 'zorder': 4},
            'secondary_link': {'color': '#d97706', 'lw': 0.56, 'alpha': 0.78, 'zorder': 4},
        },
    },
    'tehsil_boundaries': {
        'edgecolor': 'black',
        'linewidth': 0.72,
        'linestyle': (0, (1.2, 2.2)),
        'alpha': 0.72,
        'zorder': 6,
    },
    'aoi_outline': {
        'edgecolor': 'black',
        'linewidth': 1.0,
        'alpha': 0.88,
        'zorder': 8,
    },
    'spi': {
        'palette': ['#6e3b1f', '#c79a63', '#efe6c8', '#8fce6a', '#3b8f45', '#0f5b2d'],
        'percentile_clip': [2, 98],
        'vcenter': 0.0,
        'alpha': 0.96,
        'interpolation': 'bilinear',
    },
    'figure': {
        'figsize': (18, 22),
        'dpi': 150,
        'figure_facecolor': '#f7f5ef',
        'axes_facecolor': '#f7f5ef',
        'show_title': False,
        'title': 'Northern Pakistan Scenic Map',
        'title_fontsize': 18,
        'title_pad': 14,
    },
    'layout': {
        'xlim_pad_left': 0.20,
        'xlim_pad_right': 0.20,
        'ylim_pad_bottom': 0.03,
        'ylim_pad_top': 0.03,
        'tight_layout_rect': [0.02, 0.02, 0.86, 0.98],
    },
    'colorbar': {
        'show': True,
        'axes': [0.88, 0.18, 0.018, 0.54],
        'tick_labelsize': 9,
        'tick_color': 'black',
        'outline_color': 'black',
    },
    'legend': {
        'show': True,
        'location': 'lower left',
        'fontsize': 9,
        'framealpha': 0.94,
        'facecolor': 'white',
        'edgecolor': '#444444',
        'title': 'Layers',
        'title_fontsize': 10,
    },
}

paths = CONFIG['paths']
paths['outputs_dir'].mkdir(exist_ok=True)

for key in ['spi_raster', 'tehsils_file', 'roads_file', 'aoi_file', 'adm1_file']:
    f = paths[key]
    status = 'OK' if f.exists() else 'MISSING'
    print(f'{status}  {f}')
"""
        ),
        md_cell(
            """
## Load Vector Data
"""
        ),
        code_cell(
            """
tehsils = gpd.read_file(CONFIG['paths']['tehsils_file'])
print(f'Tehsils  : {len(tehsils):,} features | CRS: {tehsils.crs}')
print(f'  columns: {tehsils.columns.tolist()}')

aoi = gpd.read_file(CONFIG['paths']['aoi_file'])
print(f'\\nAOI      : {len(aoi):,} features | CRS: {aoi.crs}')

adm1 = gpd.read_file(CONFIG['paths']['adm1_file']).to_crs(aoi.crs)
major_regions = adm1[adm1['shapeName'].isin(CONFIG['regions']['region_names'])].copy()
major_regions = gpd.overlay(major_regions, aoi[['geometry']], how='intersection')
major_regions['region_label'] = major_regions['shapeName'].map(CONFIG['regions']['region_labels'])
print(f'\\nMajor regions: {len(major_regions):,} features')
print(major_regions[['shapeName', 'region_label']].to_string(index=False))

roads_all = gpd.read_file(CONFIG['paths']['roads_file'])
print(f'\\nRoads    : {len(roads_all):,} features | CRS: {roads_all.crs}')
print('\\nHighway type counts:')
print(roads_all['highway'].value_counts().to_string())
"""
        ),
        md_cell(
            """
## Filter Roads
"""
        ),
        code_cell(
            """
roads = roads_all[roads_all['highway'].isin(CONFIG['roads']['classes'])].copy()
print(f'Filtered roads: {len(roads):,} features (from {len(roads_all):,})')
print('\\nBreakdown:')
print(roads['highway'].value_counts().to_string())
"""
        ),
        md_cell(
            """
## Load SPI Raster
"""
        ),
        code_cell(
            """
with rasterio.open(CONFIG['paths']['spi_raster']) as src:
    spi_data = src.read(1).astype(np.float32)
    nodata = src.nodata
    bounds = src.bounds

if nodata is not None:
    spi_data[spi_data == nodata] = np.nan
spi_data[np.isinf(spi_data)] = np.nan

valid = spi_data[~np.isnan(spi_data)]
print(f'SPI raster shape : {spi_data.shape}')
print(f'NoData value     : {nodata}')
print(f'Valid pixels     : {len(valid):,}')
print(f'SPI value range  : [{np.nanmin(spi_data):.3f}, {np.nanmax(spi_data):.3f}]')
print(f'Mean SPI         : {np.nanmean(spi_data):.3f}')
print(f'Spatial bounds   : {bounds}')
"""
        ),
        md_cell(
            """
## Build Style Objects
"""
        ),
        code_cell(
            """
extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

cmap = mcolors.LinearSegmentedColormap.from_list(
    'scenic_greens',
    CONFIG['spi']['palette'],
    N=256
)
vmin, vmax = np.nanpercentile(spi_data, CONFIG['spi']['percentile_clip'])
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=CONFIG['spi']['vcenter'], vmax=vmax)

print(f'Colour limits: vmin={vmin:.3f}  vcenter={CONFIG["spi"]["vcenter"]:.1f}  vmax={vmax:.3f}')
"""
        ),
        md_cell(
            """
## Render Map
"""
        ),
        code_cell(
            """
fig, ax = plt.subplots(
    figsize=CONFIG['figure']['figsize'],
    dpi=CONFIG['figure']['dpi']
)
fig.patch.set_facecolor(CONFIG['figure']['figure_facecolor'])
ax.set_facecolor(CONFIG['figure']['axes_facecolor'])

im = ax.imshow(
    spi_data,
    extent=extent,
    origin='upper',
    cmap=cmap,
    norm=norm,
    alpha=CONFIG['spi']['alpha'],
    interpolation=CONFIG['spi']['interpolation'],
    zorder=1
)

for hw_class, style in CONFIG['roads']['style'].items():
    subset = roads[roads['highway'] == hw_class]
    if len(subset):
        subset.plot(
            ax=ax,
            color=style['color'],
            linewidth=style['lw'],
            alpha=style['alpha'],
            zorder=style['zorder']
        )

tehsils.plot(
    ax=ax,
    facecolor='none',
    edgecolor=CONFIG['tehsil_boundaries']['edgecolor'],
    linewidth=CONFIG['tehsil_boundaries']['linewidth'],
    linestyle=CONFIG['tehsil_boundaries']['linestyle'],
    alpha=CONFIG['tehsil_boundaries']['alpha'],
    zorder=CONFIG['tehsil_boundaries']['zorder']
)

for _, row in major_regions.iterrows():
    style = CONFIG['regions']['region_outline'].get(row['shapeName'], {})
    gpd.GeoSeries([row.geometry], crs=major_regions.crs).plot(
        ax=ax,
        facecolor='none',
        edgecolor=style.get('edgecolor', '#333333'),
        linewidth=style.get('linewidth', 2.0),
        alpha=0.95,
        zorder=style.get('zorder', 7)
    )

aoi.plot(
    ax=ax,
    facecolor='none',
    edgecolor=CONFIG['aoi_outline']['edgecolor'],
    linewidth=CONFIG['aoi_outline']['linewidth'],
    alpha=CONFIG['aoi_outline']['alpha'],
    zorder=CONFIG['aoi_outline']['zorder']
)

minx, miny, maxx, maxy = aoi.total_bounds
dx = maxx - minx
dy = maxy - miny

for _, row in major_regions.iterrows():
    rp = row.geometry.representative_point()
    pos = CONFIG['regions']['label_position_factors'][row['shapeName']]
    lx = minx + pos['x'] * dx
    ly = miny + pos['y'] * dy
    outline = CONFIG['regions']['region_outline'].get(row['shapeName'], {})
    label_style = CONFIG['regions']['label_style']
    ax.annotate(
        row['region_label'],
        xy=(rp.x, rp.y),
        xytext=(lx, ly),
        textcoords='data',
        fontsize=label_style['fontsize'],
        color=label_style['text_color'],
        ha='center',
        va='center',
        fontweight=label_style['fontweight'],
        bbox=dict(
            boxstyle=f"round,pad={label_style['box_pad']}",
            facecolor=label_style['box_facecolor'],
            edgecolor=outline.get('edgecolor', 'black'),
            linewidth=label_style['box_linewidth'],
            alpha=label_style['box_alpha'],
        ),
        arrowprops=dict(
            arrowstyle='-',
            color=outline.get('edgecolor', 'black'),
            lw=label_style['leader_linewidth'],
            shrinkA=6,
            shrinkB=5,
        ),
        zorder=9
    )

if CONFIG['colorbar']['show']:
    cbar_ax = fig.add_axes(CONFIG['colorbar']['axes'])
    cb = fig.colorbar(im, cax=cbar_ax, extend='both')
    cb.ax.tick_params(
        colors=CONFIG['colorbar']['tick_color'],
        labelsize=CONFIG['colorbar']['tick_labelsize']
    )
    cb.outline.set_edgecolor(CONFIG['colorbar']['outline_color'])

if CONFIG['legend']['show']:
    legend_items = [
        Line2D([0], [0], color=CONFIG['roads']['style']['trunk']['color'], linewidth=1.6, label='Road network'),
        Line2D(
            [0], [0],
            color=CONFIG['tehsil_boundaries']['edgecolor'],
            linewidth=1.0,
            linestyle=CONFIG['tehsil_boundaries']['linestyle'],
            label='Tehsil boundaries'
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc=CONFIG['legend']['location'],
        fontsize=CONFIG['legend']['fontsize'],
        framealpha=CONFIG['legend']['framealpha'],
        facecolor=CONFIG['legend']['facecolor'],
        edgecolor=CONFIG['legend']['edgecolor'],
        title=CONFIG['legend']['title'],
        title_fontsize=CONFIG['legend']['title_fontsize']
    )

ax.set_xlim(
    minx - CONFIG['layout']['xlim_pad_left'] * dx,
    maxx + CONFIG['layout']['xlim_pad_right'] * dx
)
ax.set_ylim(
    miny - CONFIG['layout']['ylim_pad_bottom'] * dy,
    maxy + CONFIG['layout']['ylim_pad_top'] * dy
)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
for spine in ax.spines.values():
    spine.set_visible(False)

if CONFIG['figure']['show_title']:
    ax.set_title(
        CONFIG['figure']['title'],
        fontsize=CONFIG['figure']['title_fontsize'],
        pad=CONFIG['figure']['title_pad'],
        color='black',
        fontweight='bold'
    )

plt.tight_layout(rect=CONFIG['layout']['tight_layout_rect'])
fig.savefig(
    CONFIG['paths']['output_png'],
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
print(f"Map saved -> {CONFIG['paths']['output_png']}")
plt.show()
"""
        ),
    ]

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
