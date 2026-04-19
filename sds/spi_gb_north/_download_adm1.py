import requests, json

api_url = 'https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM1/'
r = requests.get(api_url)
dl_url = r.json()['gjDownloadURL']
gj = requests.get(dl_url).json()

with open('data/raw/admin_boundaries/geoBoundaries-PAK-ADM1.geojson', 'w') as f:
    json.dump(gj, f)

print('Downloaded ADM1')
