import pandas as pd
import re

def create_station_match_key(name):

    if pd.isna(name):
        return pd.NA
    #remove parasethesis like extra text
    name = re.sub(r"\s*\([^)]*\)", "", str(name))
    #remove whitespace
    name = re.sub(r"\s+", " ", name).strip()
    #all lowercase
    return name.casefold()


def normalize_metro_data(metro_df):

    metro = metro_df.copy()

    for col in ["metro_lines", "station_names"]:
        metro[col] = metro[col].astype("string").str.strip()
    metro["station_match_key"] = (metro["station_names"].apply(create_station_match_key))
    passenger_columns = [col for col in metro.columns
            if "boarding passengers" in col
            or "alighting passengers" in col]

    if len(passenger_columns) != 48:raise ValueError( f"Expected 48 passenger columns")

    #month into datetime
    metro["months"] = pd.to_datetime(metro["months"].astype("string"),format="%Y%m",errors="raise")

    #upload date into datetime
    metro["data_upload_date"] = pd.to_datetime( metro["data_upload_date"].astype("string"),format="%Y%m%d",errors="raise")

    #passenger counts numeric
    for col in passenger_columns:metro[col] = pd.to_numeric( metro[col],errors="raise")

    metro = metro.rename(columns={"months": "month","data_upload_date": "upload_date"})
    original_rows = len(metro)
    metro = metro.drop_duplicates().copy()
    removed_rows = original_rows - len(metro)

    key_columns = ["month", "metro_lines", "station_names"]
    metro["record_total_activity"] = metro[passenger_columns].sum(axis=1)
    metro["nonzero_passenger_columns"] = (metro[passenger_columns] != 0).sum(axis=1)

    #groups with different passenger counts
    conflicting_mask = (metro.groupby(key_columns)[passenger_columns].transform("nunique").gt(1).any(axis=1))

    #records belonging to conflicting groups
    metro["has_conflicting_counts"] = conflicting_mask

    max_activity = (metro.groupby(key_columns)["record_total_activity"].transform("max"))
    is_highest_activity = metro["record_total_activity"].eq(max_activity)

    #check that there is exactly one highest-activity record per conflict group
    winner_counts = (
        is_highest_activity[conflicting_mask]
        .groupby(
            [
                metro.loc[conflicting_mask, col]
                for col in key_columns
            ]
        )
        .sum()
    )

    if (winner_counts != 1).any():
        raise ValueError("Some conflicting groups have tied activity totals ")
    metro_conflicts = metro.loc[conflicting_mask].copy()
    metro_conflicts["kept_for_analysis"] = is_highest_activity[conflicting_mask]

    #explain the decision for each record
    metro_conflicts["resolution"] = (
        metro_conflicts["kept_for_analysis"]
        .map({
            True: "kept_highest_activity",
            False: "excluded_lower_activity"
        })
    )

    #keep non-conflicting records and the highest-activity conflict records
    metro = metro.loc[~conflicting_mask | is_highest_activity].copy()

    print("Conflicting records excluded:",len(metro_conflicts) - metro_conflicts["kept_for_analysis"].sum())
    metro = metro.drop(columns=["record_total_activity","nonzero_passenger_columns"])

    #reshape hourly passenger columns into rows
    metro_long = metro.melt(
    id_vars=[
        "month",
        "metro_lines",
        "station_names",
        "station_match_key",
        "upload_date",
        "has_conflicting_counts"
    ],
    value_vars=passenger_columns,
    var_name="time_and_type",
    value_name="passenger_count"
)

    time_parts = metro_long["time_and_type"].str.extract(
        r"(?P<hour_interval>\d{2}시-\d{2}시) "
        r"(?P<passenger_type>boarding|alighting) passengers")

    #check all hourly column names were successfully interpreted
    if time_parts.isnull().any().any():
        raise ValueError("Some hourly column names could not be interpreted.")

    metro_long["hour_interval"] = time_parts["hour_interval"]
    metro_long["hour_start"] = (time_parts["hour_interval"].str[:2].astype("Int64"))
    metro_long["passenger_type"] = time_parts["passenger_type"]

    metro_long = metro_long.drop(columns=["time_and_type"])

    #arrange columns
    metro_long = metro_long[
    [
        "month",
        "metro_lines",
        "station_names",
        "station_match_key",
        "hour_start",
        "hour_interval",
        "passenger_type",
        "passenger_count",
        "upload_date",
        "has_conflicting_counts"
    ]
]
    #sort records
    metro_long = metro_long.sort_values(
        by=[
            "month",
            "metro_lines",
            "station_names",
            "hour_start",
            "passenger_type"
        ]
    ).reset_index(drop=True)

    return metro_long, metro_conflicts


def school_data_normalization(school_df):
    
    schools = school_df.copy()

    schools = schools.rename(columns={
        "city": "city_source_label",
        "school distinction": "school_type",
        "owned": "ownership",
        "name": "school_name",
        "postal code": "postal_code",
        "lat": "latitude",
        "lng": "longitude"
    })

    text_columns = [
        "city_source_label",
        "district",
        "school_type",
        "ownership",
        "school_name",
        "postal_code",
        "address"
    ]

    for col in text_columns:
        schools[col] = schools[col].astype("string").str.strip()

    schools["city"] = "Seoul"

    schools["year"] = pd.to_numeric(
        schools["year"],
        errors="coerce"
    ).astype("Int64")

    schools["postal_code"] = schools["postal_code"].str.zfill(5)


    schools["latitude"] = pd.to_numeric(
        schools["latitude"],
        errors="coerce")
    schools["longitude"] = pd.to_numeric(
        schools["longitude"],
        errors="coerce")

    schools = schools[
        [
            "year",
            "city",
            "district",
            "school_type",
            "ownership",
            "school_name",
            "postal_code",
            "address",
            "latitude",
            "longitude",
            "city_source_label"
        ]
    ]

    return schools


def normalize_station_data(station_df):

    #copy of the og
    stations = station_df.copy()

    #renaming so we keep consistent naming
    stations = stations.rename(columns={
        "line": "metro_lines",
        "name": "station_names",
        "code": "station_code",
        "lat": "latitude",
        "lng": "longitude"
    })

    #removing empty space
    stations["metro_lines"] = stations["metro_lines"].str.strip()
    stations["station_names"] = stations["station_names"].str.strip()
    stations["station_match_key"] = (
    stations["station_names"].apply(create_station_match_key))

    #station codes to integers
    stations["station_code"] = pd.to_numeric(
        stations["station_code"],
        errors="coerce"
    ).astype("Int64")

    #coordinates to numbers
    stations["latitude"] = pd.to_numeric(
        stations["latitude"],
        errors="coerce")
    stations["longitude"] = pd.to_numeric(
        stations["longitude"],
        errors="coerce")

    #rearranging columns
    stations = stations[
    ["metro_lines","station_names","station_match_key","station_code","latitude","longitude"]]

    return stations