from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

stations_df = pd.read_csv("data/processed/stations.csv", encoding="utf-8-sig")
schools_df = pd.read_csv("data/processed/schools.csv", encoding="cp949")

selected_year = 2025
schools_year = schools_df[schools_df["year"] == selected_year].copy()
valid_stations = stations_df.dropna(subset = ["latitude", "longitude"]).copy()
valid_stations = valid_stations.drop_duplicates(subset="station_match_key")

print("School records:", len(schools_year))
print("Available subway stations:", len(valid_stations))


plt.figure(figsize=(10, 9))
sns.scatterplot(
    data=valid_stations,
    x="longitude",
    y="latitude",
    color="darkblue",
    s=45,
    alpha=0.4,
    label="Subway Stations")

sns.scatterplot(
    data=schools_year,
    x="longitude",
    y="latitude",
    color="darkorange",
    marker="^",
    s=25,
    alpha=0.8,
    label="Schools")

plt.title("School and Subway Station Locations in Seoul (2025)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

mean_latitude = schools_year["latitude"].mean()
plt.gca().set_aspect(1 / np.cos(np.radians(mean_latitude)))

plt.legend()
plt.tight_layout()
plt.savefig(project_dir / "output/school_and_metro_locations.png",dpi=300)
plt.show()

#now zoom of most concentrated area (cause its sooo tiny on big map)

district_counts = schools_year["district"].value_counts()

most_concentrated_district = district_counts.idxmax()

district_schools = schools_year[schools_year["district"] == most_concentrated_district].copy()

print("Number of school records:", len(district_schools))

lat_min = district_schools["latitude"].min()
lat_max = district_schools["latitude"].max()
lon_min = district_schools["longitude"].min()
lon_max = district_schools["longitude"].max()

lat_padding = max((lat_max - lat_min) * 0.15, 0.005)
lon_padding = max((lon_max - lon_min) * 0.15, 0.005)
lat_min -= lat_padding
lat_max += lat_padding
lon_min -= lon_padding
lon_max += lon_padding


nearby_stations = valid_stations[
    (valid_stations["latitude"] >= lat_min) &
    (valid_stations["latitude"] <= lat_max) &
    (valid_stations["longitude"] >= lon_min) &
    (valid_stations["longitude"] <= lon_max)].copy()


plt.figure(figsize=(10, 9))

sns.scatterplot(
    data=nearby_stations,
    x="longitude",
    y="latitude",
    color="darkblue",
    s=45,
    alpha=0.4,
    label="Subway Stations")

sns.scatterplot(
    data=district_schools,
    x="longitude",
    y="latitude",
    color="darkorange",
    marker="^",
    s=25,
    alpha=0.8,
    label="Schools")

plt.title(f"School and Subway Station Locations - {most_concentrated_district} (most concentrated district)")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim(lon_min, lon_max)
plt.ylim(lat_min, lat_max)

mean_latitude = district_schools["latitude"].mean()

plt.gca().set_aspect(1 / np.cos(np.radians(mean_latitude)))
plt.legend()
plt.tight_layout()
plt.savefig(project_dir / "school_and_metro_locations_zoom.png",dpi=300,)
plt.show()
