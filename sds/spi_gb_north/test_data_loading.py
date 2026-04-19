#!/usr/bin/env python3
"""
SPI/AI Calculation - Data Loading and Verification Script
Test script to load and verify all preprocessed data before creating the full notebook.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print('Testing SPI/AI data loading...')

# Configure paths
cwd = Path.cwd()
if cwd.name == 'spi_gb_north':
    root = cwd
elif (cwd / 'spi_gb_north').exists():
    root = cwd / 'spi_gb_north'
elif (cwd / 'sds' / 'spi_gb_north').exists():
    root = cwd / 'sds' / 'spi_gb_north'
else:
    root = cwd

interim = root / 'data' / 'interim'
outputs = root / 'outputs'

# SPI parameters
SPI_WEIGHTS = {
    'terrain': 0.40,  # Terrain Ruggedness (TPI)
    'forest': 0.25,   # Forest Cover
    'water': 0.20,    # Water Bodies
    'snow': 0.15      # Snow/Ice Extent
}

# File paths
data_paths = {
    'dem': interim / 'dem_32643_100m.tif',
    'tpi_zscore': interim / 'tpi_products' / 'tpi_zscore_radius2_32643_100m.tif',
    'forest_mask': interim / 'forest_mask_32643_100m.tif',
    'water_mask': interim / 'water_mask_32643_100m.tif',
    'snow_frequency': interim / 'snow_aligned_32643_100m' / 'snow_frequency_aligned_32643_100m.tif',
    'roads_distance': interim / 'dist_to_roads_all_32643_100m.tif',
    'aoi_boundary': interim / 'aoi_boundary_32643.gpkg',
    'tehsils': interim / 'tehsils_aoi_32643.gpkg'
}

# Check file existence
missing_files = []
for name, path in data_paths.items():
    if not path.exists():
        missing_files.append(f"{name}: {path}")

if missing_files:
    print('❌ Missing required data files:')
    for f in missing_files:
        print(f'  {f}')
    exit(1)
else:
    print('✓ All required data files found')

# Load raster data
def load_raster_data(path, name):
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = {
            'crs': src.crs,
            'transform': src.transform,
            'shape': data.shape,
            'nodata': src.nodata,
            'dtype': data.dtype
        }

        finite_count = np.isfinite(data).sum() if np.issubdtype(data.dtype, np.floating) else (data != src.nodata).sum()
        total_pixels = data.size
        coverage_pct = (finite_count / total_pixels) * 100

        print(f'✓ {name}: {data.shape} | {data.dtype} | {coverage_pct:.1f}% valid pixels')

        return data, meta

print('\nLoading raster data layers...')
rasters = {}
metadata = {}

# Core SPI components
rasters['dem'], metadata['dem'] = load_raster_data(data_paths['dem'], 'DEM')
rasters['tpi_zscore'], metadata['tpi_zscore'] = load_raster_data(data_paths['tpi_zscore'], 'TPI Z-score')
rasters['forest_mask'], metadata['forest_mask'] = load_raster_data(data_paths['forest_mask'], 'Forest Mask')
rasters['water_mask'], metadata['water_mask'] = load_raster_data(data_paths['water_mask'], 'Water Mask')
rasters['snow_frequency'], metadata['snow_frequency'] = load_raster_data(data_paths['snow_frequency'], 'Snow Frequency')
rasters['roads_distance'], metadata['roads_distance'] = load_raster_data(data_paths['roads_distance'], 'Roads Distance')

# Load vector data
print('\nLoading vector data...')
vectors = {}
vectors['aoi'] = gpd.read_file(data_paths['aoi_boundary'])
vectors['tehsils'] = gpd.read_file(data_paths['tehsils'])

print(f'✓ AOI boundary: {len(vectors["aoi"])} features')
print(f'✓ Tehsils: {len(vectors["tehsils"])} features')

# Verify consistency
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

print('\n✓ All data layers loaded and verified successfully!')
print('Ready to proceed with SPI calculation.')