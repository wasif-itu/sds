import geopandas as gpd
import requests

print("Fetching ADM1 (Provinces)...")
r_adm1 = requests.get("https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM1/")
url_adm1 = r_adm1.json()['gjDownloadURL']
adm1 = gpd.read_file(url_adm1)
print("ADM1 regions:", adm1['shapeName'].unique())

print("\\nFetching ADM3 (Tehsils)...")
r_adm3 = requests.get("https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM3/")
url_adm3 = r_adm3.json()['gjDownloadURL']
adm3 = gpd.read_file(url_adm3)

# Filter ADM1 for Gilgit-Baltistan and Khyber Pakhtunkhwa
# Let's inspect the names directly because spellings might vary
northern_provinces = adm1[adm1['shapeName'].str.contains('Gilgit|Khyber|KPK', case=False, na=False)]
print("\\nMatched Provinces:")
print(northern_provinces['shapeName'].tolist())

# It mentions "Upper Khyber Pakhtunkhwa". Since we only have ADM1, we might just get all of KP for now 
# and then filter by some y-coordinate (latitude) or just report for GB and all KP.
# For exact "Upper KP", we might need ADM2 (Districts).
print("\\nFetching ADM2 (Districts)...")
r_adm2 = requests.get("https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM2/")
url_adm2 = r_adm2.json()['gjDownloadURL']
adm2 = gpd.read_file(url_adm2)

# Spatial join ADM3 with ADM1 to get province for each Tehsil
# use point on surface to avoid boundary overlap issues
adm3_points = adm3.copy()
adm3_points['geometry'] = adm3_points['geometry'].representative_point()
tehsils_with_province = gpd.sjoin(adm3_points, adm1[['shapeName', 'geometry']], how='inner', predicate='intersects')

gb_tehsils = tehsils_with_province[tehsils_with_province['shapeName_right'].str.contains('Gilgit', case=False, na=False)]
kp_tehsils = tehsils_with_province[tehsils_with_province['shapeName_right'].str.contains('Khyber', case=False, na=False)]

print(f"\\nTotal Tehsils in Gilgit-Baltistan: {len(gb_tehsils)}")
print(f"Total Tehsils in Khyber Pakhtunkhwa: {len(kp_tehsils)}")

# To approximate "Upper KP", we can filter KP tehsils that are located above a certain latitude
# For example, latitude > 34.5 roughly covers Upper KP (Chitral, Dir, Swat, Kohistan, etc.)
upper_kp_tehsils = kp_tehsils[kp_tehsils.geometry.y > 34.5]
print(f"Total Tehsils in 'Upper' Khyber Pakhtunkhwa (Lat > 34.5): {len(upper_kp_tehsils)}")

combined_count = len(gb_tehsils) + len(upper_kp_tehsils)
print(f"\\nEstimated Tehsils in Area of Interest (GB + Upper KP): {combined_count}")

# Extract names into a single list
aoi_tehsil_names = gb_tehsils['shapeName_left'].tolist() + upper_kp_tehsils['shapeName_left'].tolist()
aoi_tehsil_names.sort()

# Write to text file
output_file = "aoi_tehsils_list.txt"
with open(output_file, 'w') as f:
    f.write(f"Names of {combined_count} Tehsils in Gilgit-Baltistan and Upper KP (Lat > 34.5):\\n\\n")
    for name in aoi_tehsil_names:
        f.write(name + "\\n")
print(f"\\nSuccessfully wrote Tehsil names to {output_file}")
