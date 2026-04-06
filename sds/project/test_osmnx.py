import osmnx as ox
print("Testing OSMnx on a small bounding box...")
# Latitude: 35.92, Longitude: 74.30 (approximate near Gilgit)
north, south, east, west = 35.93, 35.91, 74.32, 74.28
try:
    G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type="drive", simplify=True)
    nodes, edges = ox.graph_to_gdfs(G)
    print(f"Successfully downloaded {len(edges)} road segments from bounding box.")
except Exception as e:
    print(f"OSMnx failed: {e}")
