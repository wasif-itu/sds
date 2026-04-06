import geopandas as gpd
import osmnx as ox
import requests

print("--- Testing PoC Code Logic ---")
gb_url = "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM3/"
r = requests.get(gb_url)
geojson_url = r.json()['gjDownloadURL']
pak_adm3 = gpd.read_file(geojson_url)

sample_tehsil = pak_adm3.iloc[[0]] 
bounds = sample_tehsil.total_bounds
west, south, east, north = bounds

print(f"Sample area bounds: {west}, {south}, {east}, {north}")
try:
    G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type="drive", simplify=True)
    nodes, edges = ox.graph_to_gdfs(G)
    
    edges_proj = edges.to_crs(epsg=3857)
    tehsil_proj = sample_tehsil.to_crs(epsg=3857)
    
    road_density = (edges_proj.geometry.length.sum() / 1000) / (tehsil_proj.geometry.area.sum() / 1e6)
    print(f"Road density phase 2 successful: {road_density:.4f}")
except Exception as e:
    print(f"OSMnx Phase 2 Failed: {e}")

import libpysal
from libpysal.weights import Queen
wq = Queen.from_dataframe(pak_adm3.head(10))
print(f"Phase 3 Queen Contiguity successful: {wq.n} entities.")
