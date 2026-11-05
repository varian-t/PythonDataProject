import pandas as pd
import matplotlib.pyplot as plt


metro_df = pd.read_csv("data/processed/metro.csv",
    usecols=["hour_start", "passenger_count"],
    encoding="utf-8-sig")

stations_df = pd.read_csv("data/processed/stations.csv",
    encoding="utf-8-sig")

schools_df = pd.read_csv("data/processed/schools.csv",
    encoding="cp949")

station_matching = pd.read_csv("data/integrated/station_matching.csv",
    encoding="utf-8-sig")

hourly_average = (
    metro_df.groupby("hour_start")["passenger_count"].mean().sort_index())

plt.figure(figsize=(10, 5))
plt.bar( hourly_average.index, hourly_average.values)
plt.title("Average Metro Passenger Activity by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Passenger Count")
plt.xticks(range(24))
plt.tight_layout()
plt.show()
