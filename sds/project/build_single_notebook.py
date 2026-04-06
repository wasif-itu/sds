import json

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "sds", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12.12"}}, "nbformat": 4, "nbformat_minor": 5}

def add_md(text):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    # Split text into true newlines, not the literal characters `\n`
    lines = [line + "\n" for line in text.split("\n")]
    if lines and lines[-1] == "\n":
        lines = lines[:-1]
    nb["cells"].append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": lines})

add_md("# Proof of Concept: Single Tehsil Analysis (Gilgit)\\nThis notebook runs an end-to-end data acquisition test for a single Tehsil.")

add_md("## Phase 1: Data Acquisition")
code_phase1 = """import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio.mask import mask
import requests
import matplotlib.pyplot as plt

print("1. Fetching GeoBoundaries (Pakistan ADM3/Tehsil)...")
gb_url = "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM3/"
r = requests.get(gb_url)
geo_url = r.json()['gjDownloadURL']
tehsils = gpd.read_file(geo_url)

# Select 'GILGIT' tehsil
tehsil = tehsils[tehsils['shapeName'].str.upper() == 'GILGIT'].iloc[0:1]
print(f"Target Tehsil: {tehsil['shapeName'].values[0]}")

fig, ax = plt.subplots(figsize=(6,6))
tehsil.plot(ax=ax, color='lightgreen', edgecolor='black')
plt.title(f"{tehsil['shapeName'].values[0]} Boundary")
plt.show()

print("\\n2. Fetching OSM Roads (Drive Network)...")
try:
    G = ox.graph_from_polygon(tehsil.geometry.values[0], network_type="drive", simplify=True)
    nodes, edges = ox.graph_to_gdfs(G)
    print(f"Downloaded {len(edges)} road segments for the tehsil.")
    
    fig, ax = plt.subplots(figsize=(8,8))
    tehsil.plot(ax=ax, color='none', edgecolor='black')
    edges.plot(ax=ax, linewidth=0.5, color='red')
    plt.title(f"Road Network in {tehsil['shapeName'].values[0]}")
    plt.show()
except Exception as e:
    print(f"OSMnx Extraction Failed: {e}")
"""
add_code(code_phase1)

add_md("## Phase 2: Processing and Metric Calculation Feasibility")
code_phase2 = """# Calculate Road Density
utm_crs = 32643
tehsil_proj = tehsil.to_crs(epsg=utm_crs)

if 'edges' in locals() and len(edges) > 0:
    edges_proj = edges.to_crs(epsg=utm_crs)
    total_road_length_km = edges_proj.geometry.length.sum() / 1000
    tehsil_area_sqkm = tehsil_proj.geometry.area.sum() / 1e6
    road_density = total_road_length_km / tehsil_area_sqkm
    
    print(f"Total Road Length: {total_road_length_km:.2f} km")
    print(f"Tehsil Area: {tehsil_area_sqkm:.2f} sq km")
    print(f"Calculated Road Density: {road_density:.4f} km/sq.km")
else:
    print("Could not calculate road density due to missing road data.")
"""
add_code(code_phase2)

add_md("## Phase 3: Querying and Visualizing Raster Data (STAC)")
code_phase3 = """import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import numpy as np

print("Querying Raster Data via STAC...")
bounds = tehsil.total_bounds
bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

# Elevation
print("\\nFetching Copernicus DEM (GLO-30)...")
r_dem = requests.post(stac_url, json={"collections": ["cop-dem-glo-30"], "bbox": bbox})
items_dem = r_dem.json().get('features', [])
if items_dem:
    dem_url = items_dem[0]['assets']['data']['href']
    print(f"Loading DEM from STAC URL...")
    with rasterio.open(dem_url) as src:
        out_image_dem, out_transform_dem = mask(src, tehsil.geometry, crop=True)
        out_meta_dem = src.meta
        
    fig, ax = plt.subplots(figsize=(10,10))
    im = show(out_image_dem, transform=out_transform_dem, ax=ax, cmap='terrain', title="Elevation (DEM) - Gilgit Tehsil")
    M = im.get_images()[0]
    fig.colorbar(M, ax=ax, fraction=0.046, pad=0.04, label='Elevation (meters)')
    plt.show()

# Land Cover
print("\\nFetching ESA WorldCover (10m)...")
r_esa = requests.post(stac_url, json={"collections": ["esa-worldcover"], "bbox": bbox})
items_esa = r_esa.json().get('features', [])
if items_esa:
    map_url = items_esa[0]['assets']['map']['href']
    print(f"Loading Land Cover from STAC URL...")
    with rasterio.open(map_url) as src:
        out_image_lc, out_transform_lc = mask(src, tehsil.geometry, crop=True)
    
    fig, ax = plt.subplots(figsize=(10,10))
    im = show(out_image_lc, transform=out_transform_lc, ax=ax, cmap='tab20', title="ESA WorldCover - Gilgit Tehsil")
    plt.show()
"""
add_code(code_phase3)

with open('project_poc.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("Updated Notebook 'project_poc.ipynb' specifically for Gilgit Tehsil, now with raster plots!")
