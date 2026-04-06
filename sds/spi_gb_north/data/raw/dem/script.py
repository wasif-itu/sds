import os
import requests

# ===== CONFIG =====
LAT_MIN = 30
LAT_MAX = 38
LON_MIN = 70
LON_MAX = 78

SAVE_DIR = "srtm_tiles"
BASE_URL = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"

os.makedirs(SAVE_DIR, exist_ok=True)

# ===== GENERATE TILE NAMES =====
def generate_tile_names(lat_min, lat_max, lon_min, lon_max):
    tiles = []
    for lat in range(lat_min, lat_max):
        for lon in range(lon_min, lon_max):
            lat_prefix = "N" if lat >= 0 else "S"
            lon_prefix = "E" if lon >= 0 else "W"

            tile = f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}.SRTMGL1.hgt.zip"
            tiles.append(tile)
    return tiles

tiles = generate_tile_names(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

# ===== DOWNLOAD FUNCTION =====
def download_tile(tile):
    url = BASE_URL + tile
    save_path = os.path.join(SAVE_DIR, tile)

    if os.path.exists(save_path):
        print(f"[✓] Already exists: {tile}")
        return

    try:
        print(f"[↓] Downloading: {tile}")
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"[✔] Saved: {tile}")
        else:
            print(f"[✗] Failed: {tile} ({response.status_code})")

    except Exception as e:
        print(f"[ERROR] {tile}: {e}")

# ===== DOWNLOAD ALL =====
for tile in tiles:
    download_tile(tile)

print("\n🎉 Done downloading all tiles!")