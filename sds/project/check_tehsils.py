import geopandas as gpd
import requests
import pandas as pd

print("Fetching GeoBoundaries (Pakistan ADM3/Tehsil)...")
gb_url = "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM3/"
r = requests.get(gb_url)
geojson_url = r.json()['gjDownloadURL']
pak_adm3 = gpd.read_file(geojson_url)

print("Columns available:")
print(pak_adm3.columns.tolist())

print("\\nSample rows:")
print(pak_adm3[['shapeName', 'shapeGroup']].head())

print("\\nUnique shapeGroups:")
print(pak_adm3['shapeGroup'].unique())
