import os
import pandas as pd
import matplotlib.pyplot as plt
from data_cleaning import normalize_metro_data

# 1. Load and clean the raw dataset directly
raw_df = pd.read_csv("data/raw/metrodata_translated.csv", encoding="cp949")
df, metro_conflicts = normalize_metro_data(raw_df)

def get_segment(hour):
    if 4 <= hour <= 5:
        return "Early morning (04-06h)"
    elif 6 <= hour <= 8:
        return "Morning rush (06-09h)"
    elif 9 <= hour <= 15:
        return "Midday (09-16h)"
    elif 16 <= hour <= 19:
        return "Evening rush (16-20h)"
    elif 20 <= hour <= 23:
        return "Evening (20-24h)"
    else:
        return "Night (00-04h)"


segment_files = {
    "Early morning (04-06h)": "early_morning",
    "Morning rush (06-09h)": "morning_rush",
    "Midday (09-16h)": "midday",
    "Evening rush (16-20h)": "evening_rush",
    "Evening (20-24h)": "evening",
    "Night (00-04h)": "night"
}

month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


df["month"] = pd.to_datetime(df["month"])
df["station"] = df["station_names"] + " (" + df["metro_lines"] + ")"


hourly = df.groupby(["station", "month", "hour_start"], as_index=False)["passenger_count"].sum()
hourly["daily_passengers"] = hourly["passenger_count"] / hourly["month"].dt.days_in_month


month_total = hourly.groupby(["station", "month"])["daily_passengers"].transform("sum")
hourly = hourly[month_total > 0].copy()


total_per_station = hourly.groupby("station")["daily_passengers"].sum()
months_per_station = hourly.groupby("station")["month"].nunique()
avg_daily = total_per_station / months_per_station

good_stations = avg_daily[avg_daily >= 500].index
hourly = hourly[hourly["station"].isin(good_stations)].copy()


hourly["segment"] = hourly["hour_start"].apply(get_segment)


def plot_segment_charts(result, segment, title, filename):
    seg = result[result["segment"] == segment]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))


    top_peak = seg.nlargest(10, "peak_hour")
    axes[0, 0].barh(top_peak["station"], top_peak["peak_hour"], color="red")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("MOST crowded - busiest hour")
    axes[0, 0].set_xlabel("Passengers per day")

    top_avg = seg.nlargest(10, "average_hour")
    axes[0, 1].barh(top_avg["station"], top_avg["average_hour"], color="red")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title("MOST crowded - average per hour")
    axes[0, 1].set_xlabel("Passengers per day")

    low_peak = seg.nsmallest(10, "peak_hour")
    axes[1, 0].barh(low_peak["station"], low_peak["peak_hour"], color="green")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("LEAST crowded - busiest hour")
    axes[1, 0].set_xlabel("Passengers per day")

    low_avg = seg.nsmallest(10, "average_hour")
    axes[1, 1].barh(low_avg["station"], low_avg["average_hour"], color="green")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("LEAST crowded - average per hour")
    axes[1, 1].set_xlabel("Passengers per day")

    fig.suptitle(segment + " - " + title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


all_results = []

for m in range(0, 13):
    if m == 0:
        title = "All months"
        folder = "output/figures/all_months"
        data = hourly
    else:
        title = month_names[m - 1]
        if m < 10:
            month_str = "0" + str(m)
        else:
            month_str = str(m)
        folder = "output/figures/" + month_str + "_" + title.lower()
        data = hourly[hourly["month"].dt.month == m]

    os.makedirs(folder, exist_ok=True)
    print("Making charts for:", title, "(" + str(m + 1) + "/13)")

    profile = data.groupby(["station", "segment", "hour_start"], as_index=False)["daily_passengers"].mean()
    peak_df = profile.groupby(["station", "segment"], as_index=False)["daily_passengers"].max()
    avg_df = profile.groupby(["station", "segment"], as_index=False)["daily_passengers"].mean()

    result = peak_df.rename(columns={"daily_passengers": "peak_hour"})
    result["average_hour"] = avg_df["daily_passengers"]


    result = result[result["peak_hour"] >= 10].copy()


    for segment in segment_files:
        file_path = folder + "/" + segment_files[segment] + ".png"
        plot_segment_charts(result, segment, title, file_path)

    result["scope"] = title
    all_results.append(result)

final_table = pd.concat(all_results, ignore_index=True)
os.makedirs("output/tables", exist_ok=True)
final_table.to_csv("output/tables/crowd_rankings.csv", index=False, encoding="utf-8-sig")
