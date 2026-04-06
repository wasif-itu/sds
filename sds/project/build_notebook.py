import json

# Define the notebook structure
nb = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "sds",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" for line in text.strip().split("\\n")]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\\n" for line in text.split("\\n")]
    })

md_intro = """# Proof of Concept: Spatial Disparity Analysis in Northern Pakistan
This notebook implements a Proof of Concept (PoC) to verify the feasibility of acquiring and processing the datasets necessary to calculate the Scenic Potential Index (SPI) and Accessibility Index (AI) at the Tehsil level."""
add_md(md_intro)

md_phase1 = """## Phase 1: Data Acquisition
Here we will attempt to programmatically acquire:
1. Tehsil Administrative Boundaries (GeoBoundaries)
2. OpenStreetMap Roads (OSMnx)
3. Bounding Boxes for Raster Data (ESA WorldCover, Copernicus DEM via Planetary Computer STAC)"""
add_md(md_phase1)

code_phase1 = """import geopandas as gpd
import osmnx as ox
import requests
import matplotlib.pyplot as plt

# 1. GeoBoundaries Download
print("Fetching GeoBoundaries (Pakistan ADM3/Tehsil)...")
gb_url = "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM3/"
r = requests.get(gb_url)
data = r.json()
geojson_url = data['gjDownloadURL']
pak_adm3 = gpd.read_file(geojson_url)
print(f"Loaded {len(pak_adm3)} Tehsils for Pakistan.")

# Filter for a sample: e.g., index 0
sample_tehsil = pak_adm3.iloc[[0]] 
print("Sample Tehsil selected:", sample_tehsil['shapeGroup'].values[0] if 'shapeGroup' in sample_tehsil else "Unknown")
fig, ax = plt.subplots(figsize=(5,5))
sample_tehsil.plot(ax=ax)
plt.title("Sample Tehsil Boundary")
plt.show()

# 2. Roads Download using OSMnx
print("\\nFetching OSM Roads for Sample Tehsil Bounding Box...")
bounds = sample_tehsil.total_bounds
west, south, east, north = bounds
print(f"Bounding Box: West={west}, South={south}, East={east}, North={north}")

try:
    try:
        G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type="drive", simplify=True)
    except TypeError:
        G = ox.graph_from_bbox(north, south, east, west, network_type="drive", simplify=True)
    
    nodes, edges = ox.graph_to_gdfs(G)
    print(f"Successfully downloaded {len(edges)} road segments within bounding box.")
    
    fig, ax = plt.subplots(figsize=(5,5))
    edges.plot(ax=ax, linewidth=0.5)
    plt.title("OSM Roads")
    plt.show()
except Exception as e:
    print(f"OSMnx fetch failed: {e}")

# 3. Check STAC API for Rasters
print("\\nChecking Planetary Computer STAC API for raster coverage...")
stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
bbox = [west, south, east, north]

payload1 = {"collections": ["esa-worldcover"], "bbox": bbox}
r1 = requests.post(stac_url, json=payload1)
items1 = r1.json().get('features', [])
print(f"ESA WorldCover STAC check: Found {len(items1)} items.")

payload2 = {"collections": ["cop-dem-glo-30"], "bbox": bbox}
r2 = requests.post(stac_url, json=payload2)
items2 = r2.json().get('features', [])
print(f"Copernicus DEM (GLO-30) STAC check: Found {len(items2)} items.")
"""
add_code(code_phase1)

md_phase2 = """## Phase 2: Processing and Metric Calculation Feasibility
In this phase, we calculate basic metrics simulating the Accessibility Index and Scenic Potential metrics."""
add_md(md_phase2)

code_phase2 = """# Simulate road density
if 'edges' in locals() and len(edges) > 0:
    # Estimate total road length using projected CRS
    edges_proj = edges.to_crs(epsg=3857)
    sample_tehsil_proj = sample_tehsil.to_crs(epsg=3857)
    
    total_length_m = edges_proj.geometry.length.sum()
    area_sq_m = sample_tehsil_proj.geometry.area.sum()
    
    road_density = (total_length_m / 1000) / (area_sq_m / 1e6)
    print(f"Calculated Road Density: {road_density:.4f} km/sq.km")
else:
    print("Could not calculate road density due to missing road data.")
"""
add_code(code_phase2)

md_phase3 = """## Phase 3: Spatial Analysis Workflow Feasibility
Testing geospatial libraries for spatial regression compatibility."""
add_md(md_phase3)

code_phase3 = """import libpysal
from libpysal.weights import Queen

print("Testing Queen Contiguity Spatial Weights Matrix Initialization...")
test_subset = pak_adm3.head(10).copy()
wq = Queen.from_dataframe(test_subset)
print(f"Spatial Weights successfully initialized for {test_subset.shape[0]} entities.")
print("Mean neighbors:", wq.mean_neighbors)
"""
add_code(code_phase3)

with open('project_poc.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook 'project_poc.ipynb' generated successfully.")
