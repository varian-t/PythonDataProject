"""
Identifies metro stations within 1 km of at least one school, then measures
how much passenger traffic those stations see during the two main commute
windows:
  - Morning  : 06:00–09:00
  - Afternoon: 16:00–18:00
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # non-interactive backend – saves files without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Paths
BASE_DIR        = Path(__file__).resolve().parent
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
INTEGRATED_DIR  = BASE_DIR / "data" / "integrated"
VIS_DIR         = BASE_DIR / "visualisations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

SCHOOL_STATION_CSV = INTEGRATED_DIR / "school_station_comparison.csv"
METRO_CSV          = PROCESSED_DIR  / "metro.csv"
OUTPUT_CSV         = PROCESSED_DIR  / "commute_station_summary.csv"

# Distance threshold
DISTANCE_THRESHOLD_M = 1000 # only schools within this distance count
THRESHOLD_LABEL      = "1 km" # used in chart titles and console output

# Commute hour definitions
MORNING_HOURS   = [6, 7, 8]        # 06:00–09:00
AFTERNOON_HOURS = [16, 17]         # 16:00–18:00
COMMUTE_HOURS   = MORNING_HOURS + AFTERNOON_HOURS

# Styling
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {
    "morning":   "#E07B39",   # warm orange
    "afternoon": "#3A86C8",   # cool blue
    "total":     "#5B4D8E",   # purple
    "schools":   "#2BAE66",   # green
}

# Load & filter school-station comparison data
print("Loading school-station comparison data …")
sc = pd.read_csv(SCHOOL_STATION_CSV, encoding="utf-8-sig")

# Keep each school only once using its latest available year
sc_latest = (
    sc.sort_values("year", ascending=False)
      .drop_duplicates(subset=["school_name", "address"], keep="first")
)

# Filter to schools within the distance threshold
within_threshold = sc_latest[
    sc_latest["distance_to_nearest_station_m"] <= DISTANCE_THRESHOLD_M
].copy()
print(f"  Schools within {THRESHOLD_LABEL} of a metro station : {len(within_threshold):,}")
print(f"  Unique metro stations in that set         : {within_threshold['station_match_key'].nunique():,}")

# Count schools per station
station_school_counts = (
    within_threshold
    .groupby(["station_match_key", "nearest_station_name", "nearest_station_lines"])
    .agg(
        school_count      = ("school_name",                   "nunique"),
        schools_in_500m   = ("within_500m",                   "sum"),
        avg_distance_m    = ("distance_to_nearest_station_m", "mean"),
        min_distance_m    = ("distance_to_nearest_station_m", "min"),
    )
    .reset_index()
)
station_school_counts["avg_distance_m"] = station_school_counts["avg_distance_m"].round(1)
station_school_counts["min_distance_m"] = station_school_counts["min_distance_m"].round(1)

# Exclude any stations with zero schools
station_school_counts = station_school_counts[station_school_counts["school_count"] >= 1].copy()

target_stations = set(station_school_counts["station_match_key"])

# Load metro traffic data
print("Loading metro passenger data (commute hours) …")
metro = pd.read_csv(
    METRO_CSV,
    encoding="utf-8-sig",
    usecols=["station_match_key", "hour_start", "passenger_type", "passenger_count", "month"],
)
metro["month"] = pd.to_datetime(metro["month"])

# Filter to target stations and commute hours
metro_commute = metro[
    metro["station_match_key"].isin(target_stations) &
    metro["hour_start"].isin(COMMUTE_HOURS)
].copy()

# Label commute period
metro_commute["commute_period"] = metro_commute["hour_start"].apply(
    lambda h: "morning" if h in MORNING_HOURS else "afternoon"
)

print(f"  Commute-hour records loaded: {len(metro_commute):,}")

# Aggregate traffic — average monthly passengers per station per period
print("Aggregating traffic by station and commute period …")

# Monthly total per station per period
monthly = (
    metro_commute
    .groupby(["station_match_key", "month", "commute_period"])["passenger_count"]
    .sum()
    .reset_index()
)

# Average across all months
avg_traffic = (
    monthly
    .groupby(["station_match_key", "commute_period"])["passenger_count"]
    .mean()
    .round()
    .astype(int)
    .reset_index()
    .rename(columns={"passenger_count": "avg_monthly_passengers"})
)

# Pivot wide: one row per station
traffic_wide = avg_traffic.pivot(
    index="station_match_key",
    columns="commute_period",
    values="avg_monthly_passengers"
).reset_index()
traffic_wide.columns.name = None
traffic_wide = traffic_wide.rename(columns={
    "morning":   "avg_monthly_morning",
    "afternoon": "avg_monthly_afternoon",
})
traffic_wide["avg_monthly_total_commute"] = (
    traffic_wide["avg_monthly_morning"].fillna(0) +
    traffic_wide["avg_monthly_afternoon"].fillna(0)
).astype(int)

# Build master summary table
summary = station_school_counts.merge(traffic_wide, on="station_match_key", how="left")
# Drop any stations that have no commute traffic data
summary = summary[summary["avg_monthly_total_commute"] > 0].copy()
summary = summary.sort_values("avg_monthly_total_commute", ascending=False).reset_index(drop=True)
summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"  Summary saved → {OUTPUT_CSV}")
print(f"  Stations in summary: {len(summary)}")

# Visualisation helpers
def fmt_k(x, pos=None):
    """Tick formatter: 120000 → '120 k'"""
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{int(x/1_000)} k"
    return str(int(x))


# top 25 stations by total commute traffic
TOP_N  = 25
top25  = summary.head(TOP_N).copy()

# CHART 1: 4-panel overview
print("Generating commute_traffic_overview.png …")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle(
    f"Metro Stations Within {THRESHOLD_LABEL} of Schools — Commute Traffic Overview",
    fontsize=16, fontweight="bold", y=1.01
)

# Panel A: Top 25 stations — stacked bar (morning + afternoon)
ax = axes[0, 0]
labels = top25["nearest_station_name"]
x      = np.arange(len(labels))
width  = 0.6

bars_m = ax.bar(x, top25["avg_monthly_morning"],   width, label="Morning  06–09",   color=PALETTE["morning"])
bars_a = ax.bar(x, top25["avg_monthly_afternoon"],  width,
                bottom=top25["avg_monthly_morning"], label="Afternoon 16–18", color=PALETTE["afternoon"])
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.set_title(f"Top {TOP_N} Stations: Average Monthly Commute Passengers")
ax.set_ylabel("Avg monthly passengers")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(-0.5, len(labels) - 0.5)

# Panel B: School count per station (top 25 by school count)
ax = axes[0, 1]
top_schools = summary.nlargest(TOP_N, "school_count").sort_values("school_count")
colors_sc   = [PALETTE["schools"] if v >= 10 else "#7AC99F" for v in top_schools["school_count"]]
bars_sc     = ax.barh(top_schools["nearest_station_name"], top_schools["school_count"],
                      color=colors_sc, edgecolor="white", linewidth=0.4)
ax.bar_label(bars_sc, padding=3, fontsize=8)
ax.set_title(f"Top {TOP_N} Stations: Schools Within {THRESHOLD_LABEL}")
ax.set_xlabel("Number of schools")
ax.set_xlim(0, top_schools["school_count"].max() + 3)

# Panel C: Histogram — distribution of school counts per station
ax = axes[1, 0]
ax.hist(summary["school_count"], bins=range(1, summary["school_count"].max() + 2),
        color=PALETTE["schools"], edgecolor="white", linewidth=0.5, alpha=0.9)
ax.axvline(summary["school_count"].median(), color="#E03E3E", linestyle="--",
           linewidth=1.5, label=f"Median = {summary['school_count'].median():.0f}")
ax.set_title(f"Distribution: Schools per Station ({THRESHOLD_LABEL} radius)")
ax.set_xlabel("Number of schools")
ax.set_ylabel("Number of stations")
ax.legend()

# Panel D: Scatter — school count vs total commute traffic
ax = axes[1, 1]
scatter_data = summary.dropna(subset=["avg_monthly_total_commute", "school_count"])
sc_plot = ax.scatter(
    scatter_data["school_count"],
    scatter_data["avg_monthly_total_commute"],
    c=scatter_data["avg_monthly_total_commute"],
    cmap="YlOrRd", s=60, alpha=0.75, edgecolors="grey", linewidths=0.3
)
# Annotate top 10 busiest
for _, row in scatter_data.head(10).iterrows():
    ax.annotate(
        row["nearest_station_name"],
        xy=(row["school_count"], row["avg_monthly_total_commute"]),
        xytext=(5, 3), textcoords="offset points", fontsize=7, color="#333333"
    )
plt.colorbar(sc_plot, ax=ax, label="Avg monthly commute passengers")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.set_title("School Count vs. Total Commute Traffic")
ax.set_xlabel(f"Schools within {THRESHOLD_LABEL}")
ax.set_ylabel("Avg monthly commute passengers")

fig.tight_layout()
out_path = VIS_DIR / "commute_traffic_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out_path}")

# CHART 2: Bubble chart — schools vs traffic, bubble = % in 500 m
print("Generating top_stations_bubble.png …")

bubble_data = summary.dropna(subset=["avg_monthly_total_commute"]).head(60).copy()
bubble_data["pct_500m"] = (bubble_data["schools_in_500m"] / bubble_data["school_count"]).clip(0, 1)
bubble_size  = bubble_data["school_count"] * 40

fig, ax = plt.subplots(figsize=(16, 10))
sc_bub = ax.scatter(
    bubble_data["avg_monthly_morning"],
    bubble_data["avg_monthly_afternoon"],
    s=bubble_size,
    c=bubble_data["pct_500m"],
    cmap="RdYlGn", alpha=0.78, edgecolors="grey", linewidths=0.4
)
cbar = plt.colorbar(sc_bub, ax=ax)
cbar.set_label("Fraction of schools within 500 m", fontsize=10)

# Annotate the busiest 20
for _, row in bubble_data.head(20).iterrows():
    ax.annotate(
        row["nearest_station_name"],
        xy=(row["avg_monthly_morning"], row["avg_monthly_afternoon"]),
        xytext=(6, 4), textcoords="offset points", fontsize=8, color="#222222"
    )

# Legend for bubble size
for n in [5, 10, 15, 20]:
    ax.scatter([], [], s=n * 40, label=f"{n} schools",
               color="grey", alpha=0.5, edgecolors="grey")
ax.legend(title="Bubble = school count", loc="lower right", fontsize=9)

ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
ax.set_title(
    f"School-Adjacent Metro Stations: Morning vs Afternoon Commute Traffic\n"
    f"(bubble size = schools within {THRESHOLD_LABEL}, colour = fraction within 500 m)",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Avg monthly passengers — Morning 06:00–09:00")
ax.set_ylabel("Avg monthly passengers — Afternoon 16:00–18:00")
fig.tight_layout()
out_path = VIS_DIR / "top_stations_bubble.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out_path}")

# CHART 3: Heatmap — top 30 stations × commute hour (boarding + alighting)
print("Generating heatmap_commute.png …")

TOP_HM = 30
top30_keys = summary.head(TOP_HM)["station_match_key"].tolist()
top30_names = dict(zip(summary["station_match_key"], summary["nearest_station_name"]))

hm_data = (
    metro_commute[metro_commute["station_match_key"].isin(top30_keys)]
    .groupby(["station_match_key", "hour_start"])["passenger_count"]
    .mean()
    .round()
    .reset_index()
)

hm_pivot = hm_data.pivot(index="station_match_key", columns="hour_start", values="passenger_count")
hm_pivot.index = [top30_names.get(k, k) for k in hm_pivot.index]

# Sort rows by total commute traffic (highest at top)
hm_pivot = hm_pivot.loc[[top30_names[k] for k in top30_keys if top30_names.get(k) in hm_pivot.index]]
hm_pivot.columns = [f"{h:02d}:00" for h in hm_pivot.columns]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    hm_pivot / 1_000,
    ax=ax,
    cmap="YlOrRd",
    annot=True, fmt=".0f",
    linewidths=0.4, linecolor="white",
    cbar_kws={"label": "Avg monthly passengers (thousands)"}
)
ax.set_title(
    f"Commute-Hour Traffic — Top {TOP_HM} Busiest School-Adjacent Stations\n"
    "(average monthly passengers per hour slot, boarding + alighting)",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Hour of day")
ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=9)

# Draw divider between morning and afternoon groups
# morning hours are cols 0–2 (6,7,8) | afternoon 3–4 (16,17)
n_morning = len(MORNING_HOURS)
ax.axvline(n_morning, color="black", linewidth=2)
ax.text(n_morning / 2, -0.6, "Morning 06–09",
        ha="center", va="center", fontsize=10, color=PALETTE["morning"],
        transform=ax.get_xaxis_transform())
ax.text(n_morning + len(AFTERNOON_HOURS) / 2, -0.6, "Afternoon 16–18",
        ha="center", va="center", fontsize=10, color=PALETTE["afternoon"],
        transform=ax.get_xaxis_transform())

fig.tight_layout()
out_path = VIS_DIR / "heatmap_commute.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out_path}")

# 6. Console summary
print()
print("=" * 70)
print(f"COMMUTE TRAFFIC SUMMARY — SCHOOL-ADJACENT METRO STATIONS ({THRESHOLD_LABEL})")
print("=" * 70)
print(f"\nTotal qualifying stations : {len(summary)}")
print(f"Total qualifying schools  : {within_threshold['school_name'].nunique():,}")
print()
print(f"{'Station':<35} {'Schools':>7} {'Morning':>12} {'Afternoon':>12} {'Total':>12}")
print("-" * 80)
for _, row in summary.head(20).iterrows():
    name = row["nearest_station_name"][:34]
    print(
        f"{name:<35} {int(row['school_count']):>7} "
        f"{int(row.get('avg_monthly_morning',   0) or 0):>12,} "
        f"{int(row.get('avg_monthly_afternoon', 0) or 0):>12,} "
        f"{int(row['avg_monthly_total_commute']):>12,}"
    )
print("-" * 80)
print()
print("Visualisations saved in:", VIS_DIR)
print("Summary CSV saved in   :", OUTPUT_CSV)
