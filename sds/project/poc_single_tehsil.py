import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio.mask import mask
import requests
import json
import shapely.geometry
import matplotlib.pyplot as plt
import numpy as np
import pprint

print("Starting end-to-end PoC on 'Gilgit' Tehsil...")

# 1. Fetch ADM3 
print("1. Fetching GeoBoundaries...")
gb_url = "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM3/"
r = requests.get(gb_url)
geo_url = r.json()['gjDownloadURL']
tehsils = gpd.read_file(geo_url)

# Select 'GILGIT' tehsil
tehsil = tehsils[tehsils['shapeName'].str.upper() == 'GILGIT'].iloc[0:1]
print(f"Loaded Tehsil: {tehsil['shapeName'].values[0]}")

# Save the polygon to a geojson for reference
tehsil.to_file("tehsil_gilgit.geojson", driver="GeoJSON")

# 2. Extract OSM Roads
print("\\n2. Fetching OSM Roads...")
try:
    # Use standard place query based on boundary polygon
    # This avoids generic bbox queries that might capture too much
    # Fallback to polygon if place name is ambiguous
    G = ox.graph_from_polygon(tehsil.geometry.values[0], network_type="drive", simplify=True)
    nodes, edges = ox.graph_to_gdfs(G)
    
    # Calculate Road Density (km / km^2)
    # Project to a metric CRS (e.g., EPSG:3857 for simplicity, or UTM 43N (EPSG:32643) for Gilgit)
    utm_crs = 32643
    tehsil_proj = tehsil.to_crs(epsg=utm_crs)
    edges_proj = edges.to_crs(epsg=utm_crs)
    
    total_road_length_km = edges_proj.geometry.length.sum() / 1000
    tehsil_area_sqkm = tehsil_proj.geometry.area.sum() / 1e6
    road_density = total_road_length_km / tehsil_area_sqkm
    
    print(f"Total Road Length: {total_road_length_km:.2f} km")
    print(f"Tehsil Area: {tehsil_area_sqkm:.2f} sq km")
    print(f"**Road Density (AI proxy): {road_density:.4f} km/sq.km**")
    
    edges.to_file("gilgit_roads.geojson", driver="GeoJSON")
except Exception as e:
    print(f"OSMnx Extraction Failed: {e}")

# STAC Request for Raster data
print("\\n3. Querying Raster Data (STAC)...")

# Bounding box of tehsil
bounds = tehsil.total_bounds
bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]

stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

# Elevation
payload_dem = {"collections": ["cop-dem-glo-30"], "bbox": bbox}
r_dem = requests.post(stac_url, json=payload_dem)
items_dem = r_dem.json().get('features', [])
if items_dem:
    print(f"Found {len(items_dem)} DEM tiles.")
    # Extract asset link (usually named 'data')
    dem_url = items_dem[0]['assets']['data']['href']
    print(f"DEM Download URL: {dem_url[:80]}...")
else:
    print("No DEM data found.")

# ESA WorldCover
payload_esa = {"collections": ["esa-worldcover"], "bbox": bbox}
r_esa = requests.post(stac_url, json=payload_esa)
items_esa = r_esa.json().get('features', [])
if items_esa:
    print(f"Found {len(items_esa)} ESA WorldCover tiles.")
    map_url = items_esa[0]['assets']['map']['href']
    print(f"WorldCover Map URL: {map_url[:80]}...")
else:
    print("No WorldCover data found.")

print("\\nPoC for single Tehsil execution finished successfully.")
