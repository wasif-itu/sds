import geopandas as gpd
import osmnx as ox
import requests
import os

print("--- Phase 1 Data Acquisition Test ---")

# 1. GeoBoundaries (Tehsil level - ADM3) for Pakistan
print("\n1. Testing GeoBoundaries Download...")
gb_url = "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM3/"
try:
    r = requests.get(gb_url)
    data = r.json()
    geojson_url = data['gjDownloadURL']
    print(f"GeoJSON URL found: {geojson_url}")
    print("GeoBoundaries API check passed.")
except Exception as e:
    print(f"GeoBoundaries API failed: {e}")

# 2. OSM Roads
print("\n2. Testing OSMnx Roads Download...")
try:
    place_name = "Gilgit, Pakistan"
    print(f"Fetching road network for '{place_name}'...")
    G = ox.graph_from_place(place_name, network_type="drive", simplify=True)
    nodes, edges = ox.graph_to_gdfs(G)
    print(f"Successfully downloaded {len(edges)} road segments for {place_name}.")
except Exception as e:
    print(f"OSMnx failed: {e}")

# 3. Raster Data (SRTM / ESA WorldCover) via Planetary Computer STAC
print("\n3. Testing STAC APIs for Raster Data (Planetary Computer)...")
try:
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    bbox = [74.0, 35.0, 74.5, 35.5]
    
    # Test ESA WorldCover
    payload1 = {
        "collections": ["esa-worldcover"],
        "bbox": bbox
    }
    r1 = requests.post(stac_url, json=payload1)
    items1 = r1.json().get('features', [])
    print(f"ESA WorldCover STAC check: Found {len(items1)} items for bounding box {bbox}.")
    
    # Test Copernicus DEM (SRTM alternative)
    payload2 = {
        "collections": ["cop-dem-glo-30"],
        "bbox": bbox
    }
    r2 = requests.post(stac_url, json=payload2)
    items2 = r2.json().get('features', [])
    print(f"Copernicus DEM (GLO-30) STAC check: Found {len(items2)} items for bounding box {bbox}.")
except Exception as e:
    print(f"STAC API failed: {e}")
