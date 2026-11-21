import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_FILE = "data/raw/metrodata_translated.csv"
PROCESSED_FILE = "data/processed/metro.csv"
COORD_FILE = "data/raw/station_coordinate.csv"
OUT = "output"
OUT_MONTHS = "output_by_month"
TOP_N = 30
YEARS_MONTHLY = [2022, 2023, 2024, 2025, 2026]

hours = list(range(4, 24)) + [0, 1, 2, 3]

segments = {
    "Early morning (04-07)": [4, 5, 6],
    "Morning rush (07-10)": [7, 8, 9],
    "Midday (10-17)": [10, 11, 12, 13, 14, 15, 16],
    "Evening rush (17-20)": [17, 18, 19],
    "Night (20-04)": [20, 21, 22, 23, 0, 1, 2, 3],
}

month_names = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def load_data():
    """Returns the data in long format: month, year, mon, line, station, hour, passengers."""
    if os.path.exists(PROCESSED_FILE):
        df = pd.read_csv(PROCESSED_FILE, encoding="utf-8-sig")
        df = df.rename(columns={"metro_lines": "line", "station_names": "station",
                                "hour_start": "hour", "passenger_count": "passengers"})
    else:
        raw = pd.read_csv(RAW_FILE, encoding="cp949")
        raw = raw.drop_duplicates()
        hour_columns = [c for c in raw.columns if "passengers" in c]
        df = pd.melt(raw, id_vars=["months", "metro_lines", "station_names"],
                     value_vars=hour_columns, var_name="column", value_name="passengers")
        df["hour"] = df["column"].str[:2].astype(int)     # "04..." -> 4
        df = df.rename(columns={"months": "month", "metro_lines": "line",
                                "station_names": "station"})
        df = df.groupby(["month", "line", "station", "hour", "column"])["passengers"].max().reset_index()

    text = df["month"].astype(str).str.replace("-", "").str.replace("/", "")
    df["year"] = text.str[:4].astype(int)
    df["mon"] = text.str[4:6].astype(int)
    return df[["month", "year", "mon", "line", "station", "hour", "passengers"]]

def make_table(data):
    """Average passengers (boarding + alighting) for each station and hour."""
    monthly = data.groupby(["month", "line", "station", "hour"])["passengers"].sum().reset_index()
    avg = monthly.groupby(["line", "station", "hour"])["passengers"].mean().reset_index()
    avg["label"] = avg["station"] + " (" + avg["line"] + ")"

    table = avg.pivot_table(index="label", columns="hour", values="passengers", fill_value=0)
    table = table[hours]

    total = table.sum(axis=1)
    table = table[total > 0.01 * total.median()]
    return table


def make_summary(table):
    """For each station and segment: average, busiest hour and quietest hour."""
    rows = []
    for station in table.index:
        for name, segment_hours in segments.items():
            values = table.loc[station, segment_hours]
            rows.append({"station": station,
                         "segment": name,
                         "avg_passengers": values.mean(),
                         "busiest_hour": values.idxmax(),
                         "busiest_value": values.max(),
                         "quietest_hour": values.idxmin(),
                         "quietest_value": values.min()})
    return pd.DataFrame(rows)


def make_top_bottom(summary):
    """Most and least crowded station in each segment (questions 3 and 4)."""
    rows = []
    for name in segments:
        part = summary[summary["segment"] == name]
        best = part.loc[part["avg_passengers"].idxmax()]
        worst = part.loc[part["avg_passengers"].idxmin()]
        rows.append({"segment": name,
                     "most_crowded": best["station"],
                     "most_crowded_avg": best["avg_passengers"],
                     "least_crowded": worst["station"],
                     "least_crowded_avg": worst["avg_passengers"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 3. heatmaps
def relative(table):
    """Each row divided by its own maximum: 1 = peak hour of that station."""
    return table.div(table.max(axis=1), axis=0)


def heatmap_top_stations(table, path, suffix):
    top = table.loc[table.sum(axis=1).nlargest(TOP_N).index]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(top, cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Avg passengers per hour slot (monthly total)"})
    ax.set_title(f"Top {TOP_N} busiest stations: average passengers by hour" + suffix)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Station (line)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap_relative(table, path, suffix):
    top = table.loc[table.sum(axis=1).nlargest(TOP_N).index]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(relative(top), cmap="coolwarm", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Crowdedness relative to own peak (1 = peak)"})
    ax.set_title("Relative crowdedness per station (each row scaled to its own peak)" + suffix)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Station (line)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap_all_stations(table, path, suffix):
    rel = relative(table)
    peak_hour = rel.idxmax(axis=1)
    order = peak_hour.map(hours.index).sort_values().index
    rel = rel.loc[order]

    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(rel, cmap="coolwarm", vmin=0, vmax=1, yticklabels=False, ax=ax,
                cbar_kws={"label": "Crowdedness relative to own peak (1 = peak)"})
    ax.set_title(f"All {len(rel)} stations: when does each one peak? (sorted by peak hour)" + suffix)
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Stations (one row each)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap_by_segment(table, path, suffix):
    """One heatmap for each segment of the day (top stations)."""
    top = table.loc[table.sum(axis=1).nlargest(TOP_N).index]
    fig, axes = plt.subplots(1, 5, figsize=(22, 10))
    for ax, (name, segment_hours) in zip(axes, segments.items()):
        sns.heatmap(top[segment_hours], cmap="YlOrRd", vmin=0, vmax=top.values.max(),
                    cbar=False, yticklabels=(name == "Early morning (04-07)"), ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Hour")
        ax.set_ylabel("")
    fig.suptitle("Station crowdedness by segment of the day" + suffix)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap_segment_avg(summary, path, suffix):
    pv = summary.pivot_table(index="station", columns="segment", values="avg_passengers")
    pv = pv[list(segments)]
    pv = pv.loc[pv.sum(axis=1).nlargest(TOP_N).index]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(pv, cmap="YlOrRd", annot=True, fmt=".0f", ax=ax)
    ax.set_title("Average passengers per hour by segment (top stations)" + suffix)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmaps_by_line(table, folder, suffix):
    """One relative heatmap for each metro line, with the station names."""
    os.makedirs(folder, exist_ok=True)
    line_of = pd.Series([label.rsplit(" (", 1)[1][:-1] for label in table.index],
                        index=table.index)
    for line in line_of.unique():
        part = table[line_of == line]
        if len(part) < 2:
            continue
        rel = relative(part)
        order = rel.idxmax(axis=1).map(hours.index).sort_values().index
        rel = rel.loc[order]
        rel.index = [label.rsplit(" (", 1)[0] for label in rel.index]

        fig, ax = plt.subplots(figsize=(12, max(4, 0.22 * len(rel) + 2)))
        sns.heatmap(rel, cmap="coolwarm", vmin=0, vmax=1, ax=ax,
                    cbar_kws={"label": "Relative to own peak (1 = peak)"})
        ax.set_title(f"{line}: station crowdedness by hour (sorted by peak hour)" + suffix)
        ax.set_xlabel("Hour of the day")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(f"{folder}/heatmap_{line}.png", dpi=120)
        plt.close(fig)

def clean_name(names):
    """Same name format in the metro data and in the coordinates file."""
    return names.str.split("(").str[0].str.strip().str.lower().str.replace(" ", "")


def map_by_segment(data, path, suffix):
    """Map of Seoul: one panel per segment, colour = crowdedness."""
    if not os.path.exists(COORD_FILE):
        print("coordinates file not found -> map skipped")
        return

    coords = pd.read_csv(COORD_FILE)
    coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
    coords["lng"] = pd.to_numeric(coords["lng"], errors="coerce")
    coords = coords.dropna(subset=["lat", "lng"])
    coords["key"] = clean_name(coords["name"])
    coords = coords.groupby("key")[["lat", "lng"]].mean().reset_index()

    monthly = data.groupby(["month", "station", "hour"])["passengers"].sum().reset_index()
    avg = monthly.groupby(["station", "hour"])["passengers"].mean().reset_index()
    station_table = avg.pivot_table(index="station", columns="hour", values="passengers", fill_value=0)

    result = pd.DataFrame()
    result["station"] = station_table.index
    for name, segment_hours in segments.items():
        result[name] = station_table[segment_hours].mean(axis=1).values
    result["key"] = clean_name(result["station"])
    result = result.merge(coords, on="key", how="inner")
    print("stations on the map:", len(result), "of", len(station_table))

    result = result[result["lat"].between(37.40, 37.72) & result["lng"].between(126.76, 127.20)]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.ravel()
    for ax, name in zip(axes, segments):
        part = result.sort_values(name)
        vmax = part[name].quantile(0.95)
        points = ax.scatter(part["lng"], part["lat"], c=part[name], cmap="YlOrRd",
                            vmin=0, vmax=vmax, s=10 + 80 * part[name] / part[name].max(),
                            edgecolor="black", linewidth=0.2)
        fig.colorbar(points, ax=ax, label="Avg passengers per hour slot")
        for i in part.tail(3).index:
            ax.text(part.loc[i, "lng"], part.loc[i, "lat"], part.loc[i, "station"], fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    axes[-1].axis("off")
    fig.suptitle("Seoul subway: station crowdedness by segment of the day" + suffix)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def make_all(data, folder, suffix=""):
    os.makedirs(folder, exist_ok=True)
    table = make_table(data)
    summary = make_summary(table)
    top_bottom = make_top_bottom(summary)

    summary.to_csv(f"{folder}/segment_summary.csv", index=False, encoding="utf-8-sig")
    top_bottom.to_csv(f"{folder}/top_bottom_per_segment.csv", index=False, encoding="utf-8-sig")
    heatmap_top_stations(table, f"{folder}/heatmap_top_stations.png", suffix)
    heatmap_relative(table, f"{folder}/heatmap_relative.png", suffix)
    heatmap_all_stations(table, f"{folder}/heatmap_all_stations.png", suffix)
    heatmap_by_segment(table, f"{folder}/heatmap_by_segment.png", suffix)
    heatmap_segment_avg(summary, f"{folder}/heatmap_segment_avg.png", suffix)
    heatmaps_by_line(table, f"{folder}/lines", suffix)
    map_by_segment(data, f"{folder}/map_by_segment.png", suffix)
    return top_bottom

def compare_months(data, folder):
    """Whole network: how the 12 months compare with each other."""
    os.makedirs(folder, exist_ok=True)

    net = data.groupby(["year", "mon", "hour"])["passengers"].sum().reset_index()
    net = net.groupby(["mon", "hour"])["passengers"].mean().reset_index()
    net = net.pivot_table(index="mon", columns="hour", values="passengers")[hours]
    net.index = [name[:3] for name in month_names]

    fig, ax = plt.subplots(figsize=(13, 6))
    for month in net.index:
        ax.plot([str(h) for h in hours], net.loc[month], label=month, linewidth=2)
    ax.set_title("Network passengers by hour, one line per month")
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Passengers per hour slot (all stations, monthly total)")
    ax.legend(ncol=6)
    fig.tight_layout()
    fig.savefig(f"{folder}/lines_hourly_profile_by_month.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(relative(net), cmap="coolwarm", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Relative to the month's own peak hour"})
    ax.set_title("When does the network peak in each month?")
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(f"{folder}/heatmap_month_by_hour.png", dpi=150)
    plt.close(fig)

    seg = pd.DataFrame()
    for name, segment_hours in segments.items():
        seg[name] = net[segment_hours].mean(axis=1)
    fig, ax = plt.subplots(figsize=(13, 6))
    seg.plot(kind="bar", ax=ax, width=0.85)
    ax.set_title("Network passengers per hour slot, by segment and month")
    ax.set_ylabel("Avg passengers per hour slot (all stations, monthly total)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.tight_layout()
    fig.savefig(f"{folder}/bars_segments_by_month.png", dpi=150)
    plt.close(fig)

    seg.round(0).to_csv(f"{folder}/network_segments_by_month.csv", encoding="utf-8-sig")

data = load_data()
result = make_all(data, OUT)
print(result.to_string(index=False))
recent = data[data["year"].isin(YEARS_MONTHLY)]
all_results = []
for m in range(1, 13):
    name = month_names[m - 1]
    part = recent[recent["mon"] == m]
    years = part["year"].nunique()
    print("month:", name, "-", years, "years of data")
    suffix = f"  |  {name} (avg. of {years} years)"
    result = make_all(part, f"{OUT_MONTHS}/{m:02d}_{name}", suffix)
    result.insert(0, "month", name)
    all_results.append(result)

compare_months(recent, f"{OUT_MONTHS}/comparison")
pd.concat(all_results).to_csv(f"{OUT_MONTHS}/comparison/most_least_crowded_by_month.csv",
                              index=False, encoding="utf-8-sig")
print("done")
