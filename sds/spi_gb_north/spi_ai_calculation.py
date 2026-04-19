#!/usr/bin/env python3
"""
SPI/AI Calculation - Complete Implementation
Load all preprocessed data, verify, visualize, and prepare for SPI calculation.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from rasterio.plot import show
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('SCENIC POTENTIAL INDEX (SPI) AND ACCESSIBILITY INDEX (AI) CALCULATION')
print('='*80)

# Configure paths and parameters
cwd = Path.cwd()
if cwd.name == 'spi_gb_north':
    root = cwd
elif (cwd / 'spi_gb_north').exists():
    root = cwd / 'spi_gb_north'
elif (cwd / 'sds' / 'spi_gb_north').exists():
    root = cwd / 'sds' / 'spi_gb_north'
else:
    root = cwd

# Data directories
interim = root / 'data' / 'interim'
processed = root / 'data' / 'processed'
outputs = root / 'outputs'

# Create output directories
for d in [processed, outputs]:
    d.mkdir(parents=True, exist_ok=True)

# SPI parameters (from project proposal)
SPI_WEIGHTS = {
    'terrain': 0.40,  # Terrain Ruggedness (TPI)
    'forest': 0.25,   # Forest Cover
    'water': 0.20,    # Water Bodies
    'snow': 0.15      # Snow/Ice Extent
}

print(f'Root directory: {root}')
print(f'Interim data: {interim}')
print(f'Output directory: {outputs}')
print(f'SPI Weights: {SPI_WEIGHTS}')

# File paths for preprocessed data
data_paths = {
    'dem': interim / 'dem_32643_100m.tif',
    'landcover': interim / 'landcover_2024_32643_100m.tif',
    'forest_mask': interim / 'forest_mask_32643_100m.tif',
    'water_mask': interim / 'water_mask_32643_100m.tif',
    'tpi_raw': interim / 'tpi_products' / 'tpi_raw_radius2_32643_100m.tif',
    'tpi_zscore': interim / 'tpi_products' / 'tpi_zscore_radius2_32643_100m.tif',
    'tpi_landforms': interim / 'tpi_products' / 'tpi_landforms_radius2_32643_100m.tif',
    'snow_days': interim / 'snow_aligned_32643_100m' / 'snow_days_aligned_32643_100m.tif',
    'snow_frequency': interim / 'snow_aligned_32643_100m' / 'snow_frequency_aligned_32643_100m.tif',
    'roads_distance': interim / 'dist_to_roads_all_32643_100m.tif',
    'aoi_boundary': interim / 'aoi_boundary_32643.gpkg',
    'tehsils': interim / 'tehsils_aoi_32643.gpkg',
    'roads': interim / 'roads_all_32643.gpkg'
}

# Verify all required files exist
missing_files = []
for name, path in data_paths.items():
    if not path.exists():
        missing_files.append(f"{name}: {path}")

if missing_files:
    print('\n❌ Missing required data files:')
    for f in missing_files:
        print(f'  {f}')
    raise FileNotFoundError('Cannot proceed without all preprocessed data files')
else:
    print('\n✓ All required data files found')

print('\n' + '='*80)
print('STEP 1: Load and Verify Preprocessed Data Layers')
print('='*80)

# Load all raster data layers
def load_raster_data(path, name):
    """Load raster data and return array, metadata"""
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = {
            'crs': src.crs,
            'transform': src.transform,
            'shape': data.shape,
            'nodata': src.nodata,
            'dtype': data.dtype,
            'bounds': src.bounds
        }

        # Print basic info
        finite_count = np.isfinite(data).sum() if np.issubdtype(data.dtype, np.floating) else (data != src.nodata).sum()
        total_pixels = data.size
        coverage_pct = (finite_count / total_pixels) * 100

        print(f'✓ {name}: {data.shape} | {data.dtype} | {coverage_pct:.1f}% valid pixels')

        return data, meta

# Load raster layers
print('Loading raster data layers...')
rasters = {}
metadata = {}

# Core SPI components
rasters['dem'], metadata['dem'] = load_raster_data(data_paths['dem'], 'DEM')
rasters['tpi_zscore'], metadata['tpi_zscore'] = load_raster_data(data_paths['tpi_zscore'], 'TPI Z-score')
rasters['forest_mask'], metadata['forest_mask'] = load_raster_data(data_paths['forest_mask'], 'Forest Mask')
rasters['water_mask'], metadata['water_mask'] = load_raster_data(data_paths['water_mask'], 'Water Mask')
rasters['snow_frequency'], metadata['snow_frequency'] = load_raster_data(data_paths['snow_frequency'], 'Snow Frequency')

# Additional layers
rasters['roads_distance'], metadata['roads_distance'] = load_raster_data(data_paths['roads_distance'], 'Roads Distance')
rasters['landcover'], metadata['landcover'] = load_raster_data(data_paths['landcover'], 'Land Cover')

# Load vector data
print('\nLoading vector data...')
vectors = {}
vectors['aoi'] = gpd.read_file(data_paths['aoi_boundary'])
vectors['tehsils'] = gpd.read_file(data_paths['tehsils'])
vectors['roads'] = gpd.read_file(data_paths['roads'])

print(f'✓ AOI boundary: {len(vectors["aoi"])} features')
print(f'✓ Tehsils: {len(vectors["tehsils"])} features')
print(f'✓ Roads: {len(vectors["roads"])} features')

# Verify all rasters have consistent dimensions and CRS
ref_shape = metadata['dem']['shape']
ref_crs = metadata['dem']['crs']

print(f'\nVerifying data consistency...')
print(f'Reference: DEM {ref_shape} | CRS: {ref_crs}')

for name, meta in metadata.items():
    if meta['shape'] != ref_shape:
        print(f'❌ Shape mismatch in {name}: {meta["shape"]} vs {ref_shape}')
    elif meta['crs'] != ref_crs:
        print(f'❌ CRS mismatch in {name}: {meta["crs"]} vs {ref_crs}')
    else:
        print(f'✓ {name}: shape and CRS match')

print('\n✓ All data layers loaded and verified')

print('\n' + '='*80)
print('STEP 2: Data Quality Verification and Statistics')
print('='*80)

# Compute statistics for each SPI component
def compute_raster_stats(data, nodata, name):
    """Compute comprehensive statistics for a raster"""
    if np.issubdtype(data.dtype, np.floating):
        valid_mask = np.isfinite(data)
    else:
        valid_mask = data != nodata

    valid_data = data[valid_mask]

    stats = {
        'name': name,
        'total_pixels': data.size,
        'valid_pixels': len(valid_data),
        'coverage_pct': (len(valid_data) / data.size) * 100,
        'min': float(np.nanmin(data)) if np.issubdtype(data.dtype, np.floating) else float(data.min()),
        'max': float(np.nanmax(data)) if np.issubdtype(data.dtype, np.floating) else float(data.max()),
        'mean': float(np.nanmean(data)) if np.issubdtype(data.dtype, np.floating) else float(valid_data.mean()),
        'std': float(np.nanstd(data)) if np.issubdtype(data.dtype, np.floating) else float(valid_data.std()),
        'dtype': str(data.dtype)
    }

    return stats

print('Computing data quality statistics...')

# SPI component statistics
spi_stats = []
spi_components = {
    'Terrain (TPI)': rasters['tpi_zscore'],
    'Forest Cover': rasters['forest_mask'],
    'Water Bodies': rasters['water_mask'],
    'Snow/Ice': rasters['snow_frequency']
}

for name, data in spi_components.items():
    key = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    if 'terrain' in key:
        key = 'tpi_zscore'
    elif 'forest' in key:
        key = 'forest_mask'
    elif 'water' in key:
        key = 'water_mask'
    elif 'snow' in key:
        key = 'snow_frequency'

    nodata = metadata[key]['nodata']
    stats = compute_raster_stats(data, nodata, name)
    spi_stats.append(stats)

    print(f"\n{name}:")
    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"  Coverage: {stats['coverage_pct']:.1f}%")

# Additional statistics
print(f"\nDEM Elevation:")
dem_stats = compute_raster_stats(rasters['dem'], metadata['dem']['nodata'], 'DEM')
print(f"  Range: [{dem_stats['min']:.0f}, {dem_stats['max']:.0f}] meters")
print(f"  Mean ± Std: {dem_stats['mean']:.0f} ± {dem_stats['std']:.0f} meters")

print(f"\nRoads Distance:")
roads_stats = compute_raster_stats(rasters['roads_distance'], metadata['roads_distance']['nodata'], 'Roads Distance')
print(f"  Range: [{roads_stats['min']:.0f}, {roads_stats['max']:.0f}] meters")
print(f"  Mean ± Std: {roads_stats['mean']:.0f} ± {roads_stats['std']:.0f} meters")

print('\n✓ Data quality verification complete')

print('\n' + '='*80)
print('STEP 3: Visualize Preprocessed Data Layers')
print('='*80)

# Create comprehensive visualization of all preprocessed layers
def create_data_overview_plot():
    """Create a comprehensive overview of all data layers"""

    # Crop to AOI for better visualization
    def crop_to_aoi(arr):
        ys, xs = np.where(np.isfinite(arr))
        if len(ys) == 0:
            return arr
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        return arr[y0:y1, x0:x1]

    # Prepare cropped data
    dem_crop = crop_to_aoi(rasters['dem'])
    tpi_crop = crop_to_aoi(rasters['tpi_zscore'])
    forest_crop = crop_to_aoi(rasters['forest_mask'])
    water_crop = crop_to_aoi(rasters['water_mask'])
    snow_crop = crop_to_aoi(rasters['snow_frequency'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Preprocessed Data Layers Overview - GB + North KP + AJK', fontsize=16, fontweight='bold')

    # DEM
    ax = axes[0, 0]
    im = ax.imshow(dem_crop, cmap='terrain', vmin=np.nanmin(dem_crop), vmax=np.nanmax(dem_crop))
    ax.set_title('Digital Elevation Model\\n(meters)')
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, shrink=0.8)

    # TPI Z-score
    ax = axes[0, 1]
    im = ax.imshow(tpi_crop, cmap='RdYlBu_r', vmin=-3, vmax=3)
    ax.set_title('Terrain Ruggedness (TPI)\\n(z-scores)')
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Forest Cover
    ax = axes[0, 2]
    im = ax.imshow(forest_crop, cmap='Greens', vmin=0, vmax=1)
    ax.set_title('Forest Cover\\n(binary mask)')
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Water Bodies
    ax = axes[1, 0]
    im = ax.imshow(water_crop, cmap='Blues', vmin=0, vmax=1)
    ax.set_title('Water Bodies\\n(binary mask)')
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Snow Frequency
    ax = axes[1, 1]
    im = ax.imshow(snow_crop, cmap='Purples', vmin=0, vmax=np.nanmax(snow_crop))
    ax.set_title('Snow/Ice Frequency\\n(0-1 scale)')
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Roads Distance
    ax = axes[1, 2]
    roads_crop = crop_to_aoi(rasters['roads_distance'])
    im = ax.imshow(roads_crop, cmap='plasma_r', vmin=0, vmax=5000)
    ax.set_title('Distance to Roads\\n(meters)')
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(outputs / 'preprocessed_data_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f'✓ Overview plot saved: {outputs / "preprocessed_data_overview.png"}')

create_data_overview_plot()

print('\n' + '='*80)
print('STEP 4: SPI Calculation Preparation')
print('='*80)

# Prepare SPI components for calculation
print('Preparing SPI components for calculation...')

# Extract SPI components with proper scaling
terrain_component = rasters['tpi_zscore'].copy()  # Already z-scored
forest_component = rasters['forest_mask'].astype(np.float32)  # 0-1 binary
water_component = rasters['water_mask'].astype(np.float32)  # 0-1 binary
snow_component = rasters['snow_frequency'].copy()  # 0-1 frequency

# Create NoData mask (where any component is NoData)
nodata_mask = (
    np.isnan(terrain_component) |
    np.isnan(forest_component) |
    np.isnan(water_component) |
    np.isnan(snow_component)
)

# Apply NoData mask to all components
terrain_component[nodata_mask] = np.nan
forest_component[nodata_mask] = np.nan
water_component[nodata_mask] = np.nan
snow_component[nodata_mask] = np.nan

print('✓ SPI components prepared')
print(f'  Valid pixels for SPI calculation: {(~nodata_mask).sum():,}')
print(f'  Coverage: {(~nodata_mask).sum() / nodata_mask.size * 100:.1f}%')

# Display component statistics
print('\\nSPI Component Statistics (pre-weighted):')
components_data = {
    'Terrain': terrain_component,
    'Forest': forest_component,
    'Water': water_component,
    'Snow': snow_component
}

for name, data in components_data.items():
    valid_data = data[~nodata_mask]
    print(f'  {name}: mean={np.nanmean(data):.3f}, std={np.nanstd(data):.3f}, range=[{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]')

print('\\n✓ Ready for SPI calculation')
print('\\n' + '='*80)
print('NEXT STEPS:')
print('1. Compute SPI using weighted formula')
print('2. Compute Accessibility Index (AI)')
print('3. Generate zonal statistics by tehsil')
print('4. Create final visualizations and reports')
print('='*80)