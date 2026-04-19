import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects
from matplotlib.lines import Line2D
import geopandas as gpd
import rasterio

BASE_DIR  = Path('.')
PROCESSED = BASE_DIR / 'data' / 'processed'
INTERIM   = BASE_DIR / 'data' / 'interim'
OUTPUTS   = BASE_DIR / 'outputs'

SPI_RASTER   = PROCESSED / 'spi_index.tif'
TEHSILS_FILE = INTERIM   / 'tehsils_aoi_32643.gpkg'
ROADS_FILE   = INTERIM   / 'roads_all_32643.gpkg'
AOI_FILE     = INTERIM   / 'aoi_boundary_north_kpk_gb_ajk_32643.gpkg'
ADM1_FILE    = BASE_DIR / 'data' / 'raw' / 'admin_boundaries' / 'geoBoundaries-PAK-ADM1.geojson'

# Load data
tehsils = gpd.read_file(TEHSILS_FILE)
aoi = gpd.read_file(AOI_FILE)
roads_all = gpd.read_file(ROADS_FILE)
ROAD_CLASSES = ['motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link']
roads = roads_all[roads_all['highway'].isin(ROAD_CLASSES)].copy()

adm1 = gpd.read_file(ADM1_FILE).to_crs(tehsils.crs)
# Intersect ADM1 with AOI to get the major regions inside the study area
regions = gpd.clip(adm1, aoi)
# Filter empty
regions = regions[~regions.is_empty & (regions.geom_type.isin(['Polygon', 'MultiPolygon']))]

with rasterio.open(SPI_RASTER) as src:
    spi_data  = src.read(1).astype('float32')
    nodata    = src.nodata
    bounds    = src.bounds
if nodata is not None:
    spi_data[spi_data == nodata] = float('nan')
spi_data[~np.isfinite(spi_data)] = float('nan')

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
# Make green more scenic: Use RdYlGn, but shift center so most scenic are distinctly green
cmap = plt.cm.RdYlGn
vmin, vmax = np.nanpercentile(spi_data, [2, 98])
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

fig, ax = plt.subplots(figsize=(18, 22), dpi=150)
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

# 1. SPI Raster
im = ax.imshow(spi_data, extent=extent, origin='upper', cmap=cmap, norm=norm, alpha=0.90, interpolation='bilinear', zorder=1)

# 2. Tehsil boundaries (dotted black lines)
tehsils.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8, linestyle=':', alpha=0.8, zorder=7)

# 3. Major Regions enclosed with distinct color
# Let's assign distinct colors to the regions
colors = ['#e74c3c', '#9b59b6', '#3498db', '#e67e22', '#2ecc71']
for i, (_, row) in enumerate(regions.iterrows()):
    c = colors[i % len(colors)]
    gpd.GeoDataFrame([row], crs=regions.crs).plot(ax=ax, facecolor='none', edgecolor=c, linewidth=3.0, zorder=8)
    
    # Label region
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    name = row['shapeName'].upper()
    ax.annotate(name, xy=(cx, cy), fontsize=16, color=c, ha='center', va='center', fontweight='bold', zorder=10,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground='#1a1a2e')])

# 4. Roads (orange, thick)
roads.plot(ax=ax, color='orange', linewidth=2.0, alpha=0.9, zorder=6)

# 5. AOI outline
aoi.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=2.0, linestyle='-', alpha=0.8, zorder=9)

# 6. Colorbar (no label as requested)
cbar_ax = fig.add_axes([0.88, 0.25, 0.025, 0.50])
cb = fig.colorbar(im, cax=cbar_ax, extend='both')
# Remove SPI label
# cb.set_label('Scenic Potential Index (SPI)', ...)
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=10)
cb.outline.set_edgecolor('white')

# Legend
legend_items = [
    Line2D([0],[0], color='orange', linewidth=2.0, label='Major Roads'),
    Line2D([0],[0], color='black', linestyle=':', linewidth=1.0, label='Tehsil Boundary'),
]
for i, (_, row) in enumerate(regions.iterrows()):
    c = colors[i % len(colors)]
    name = row['shapeName'].upper()
    legend_items.append(Line2D([0],[0], color=c, linewidth=3.0, label=f'Region: {name}'))

legend = ax.legend(handles=legend_items, loc='lower left', fontsize=10, framealpha=0.6,
                   facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(rect=[0, 0, 0.87, 1])
OUT_PNG = OUTPUTS / 'spi_regions_roads_map.png'
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved -> {OUT_PNG}')
