
import pandas as pd
from difflib import SequenceMatcher
import re

def inspect_dataset(df, dataset_name):
    print(f"DATASET: {dataset_name}")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())


def investigate_metro_duplicates(metro_df):

    key_columns = ["months", "metro_lines", "station_names"]

    # Identify passenger count columns
    passenger_columns = [
        col for col in metro_df.columns
        if "boarding passengers" in col
        or "alighting passengers" in col
    ]

    # Remove exact duplicates
    metro_unique = metro_df.drop_duplicates().copy()

    # Find repeated station-month records
    duplicate_stations = metro_unique[
        metro_unique.duplicated(
            subset=key_columns,
            keep=False
        )
    ]

    # Identify groups with different passenger counts
    passenger_differences = (
        duplicate_stations
        .groupby(key_columns)[passenger_columns]
        .nunique()
    )

    conflicting_groups = passenger_differences[
        passenger_differences.gt(1).any(axis=1)
    ]

    # Get the keys of the conflicting groups
    conflicting_keys = conflicting_groups.reset_index()[key_columns]

    # Retrieve all records belonging to those groups
    conflicting_records = duplicate_stations.merge(
        conflicting_keys,
        on=key_columns,
        how="inner"
    )

    # Calculate the number of non-zero passenger columns per record
    conflicting_records["nonzero_columns"] = (
        conflicting_records[passenger_columns] != 0
    ).sum(axis=1)

    # Calculate total passengers per record
    conflicting_records["total_passengers"] = (
        conflicting_records[passenger_columns].sum(axis=1)
    )

    # Summarize the records in each conflicting group
    conflict_summary = (
        conflicting_records
        .groupby(key_columns)
        .agg(
            record_count=("data_upload_date", "size"),
            upload_date_count=("data_upload_date", "nunique"),
            min_nonzero_columns=("nonzero_columns", "min"),
            max_nonzero_columns=("nonzero_columns", "max"),
            min_total_passengers=("total_passengers", "min"),
            max_total_passengers=("total_passengers", "max")
        )
        .reset_index()
    )

    print("\n" + "=" * 50)
    print("METRO CONFLICT SUMMARY")
    print("=" * 50)

    print("\nNumber of conflicting groups:")
    print(len(conflict_summary))

    print("\nGroups containing exactly two records:")
    print((conflict_summary["record_count"] == 2).sum())

    print("\nGroups with the same upload date for all records:")
    print((conflict_summary["upload_date_count"] == 1).sum())

    print("\nGroups where the smallest record has at most 2 non-zero columns:")
    print((conflict_summary["min_nonzero_columns"] <= 2).sum())

    print("\nGroups where the smallest record has a total count of 2 or less:")
    print((conflict_summary["min_total_passengers"] <= 2).sum())

    # Display summary for all conflicting groups
    print("\nSummary of all conflicting groups:")

    print(
        conflict_summary.sort_values(
            by=["months", "metro_lines", "station_names"]
        ).to_string(index=False)
    )

def investigate_station_coordinates(station_df):

    print("STATION COORDINATES")
    print("\nMissing codes:")
    print(station_df[station_df["code"].isnull()])

    for col in ["lat", "lng"]:

        invalid_coordinates = pd.to_numeric(station_df[col],errors="coerce").isnull()

        print(f"\nInvalid values in {col}:")
        print(station_df.loc[invalid_coordinates, col].value_counts())

        print(f"Number of invalid values: {invalid_coordinates.sum()}")

def investigate_school_records(school_df):

    print("SCHOOL DATA")

    print("\nNumber of school records per year:")
    print(school_df["year"].value_counts().sort_index())

    print("\nSchool classifications:")
    print(school_df["school distinction"].value_counts())

    print("\nSchool ownership:")
    print(school_df["owned"].value_counts())

    print("\nRepeated school names:")
    print(school_df["name"].value_counts().head(15))


def investigate_station_matching(metro_df, station_df):

    key_columns = ["metro_lines", "station_names"]
    metro_stations = metro_df[key_columns].drop_duplicates()

    coordinate_stations = station_df[key_columns].drop_duplicates()

    station_comparison = metro_stations.merge(coordinate_stations,on=key_columns,how="left",indicator=True)

    exact_matches = station_comparison[station_comparison["_merge"] == "both"]
    unmatched_stations = station_comparison[station_comparison["_merge"] == "left_only"][key_columns].copy()
    coordinate_names = coordinate_stations["station_names"].drop_duplicates()
    unmatched_stations["name_exists"] = (unmatched_stations["station_names"].isin(coordinate_names))
    different_line_matches = unmatched_stations[unmatched_stations["name_exists"]]
    missing_station_names = unmatched_stations[ ~unmatched_stations["name_exists"]]
    possible_matches = different_line_matches.merge(coordinate_stations,on="station_names",how="inner",suffixes=("_metro", "_coordinates"))


    line_mapping_candidates = (possible_matches .groupby([
            "metro_lines_metro",
            "metro_lines_coordinates"
        ])
        .agg(
            matching_station_names=("station_names", "nunique"),
            example_stations=(
                "station_names",
                lambda x: ", ".join(sorted(x.unique())[:8])
            )
        )
        .reset_index()
        .sort_values(
            by=["metro_lines_metro", "matching_station_names"],
            ascending=[True, False]
        )
    )

    print("STATION MATCHING INVESTIGATION")

    print("\nUnique metro station-line combinations:")
    print(len(metro_stations))

    print("\nUnique coordinate station-line combinations:")
    print(len(coordinate_stations))

    print("\nExact matches (same name and line):")
    print(len(exact_matches))

    print("\nUnmatched station-line combinations:")
    print(len(unmatched_stations))

    print("\nUnmatched combinations with a matching station name:")
    print(len(different_line_matches))

    print("\nStation names not found in coordinate dataset:")
    print(missing_station_names["station_names"].nunique())

    print("\nStation names completely missing:")
    print(
        missing_station_names["station_names"]
        .drop_duplicates()
        .sort_values()
        .to_string(index=False)
    )

    print("POSSIBLE METRO LINE MAPPINGS")
    print(line_mapping_candidates.to_string(index=False))



def validate_station_matches(station_matching):

    print("STATION MATCHING VALIDATION")

    print("\nTotal station-line combinations:")
    print(len(station_matching))

    print("\nCoordinate matching results:")
    print(station_matching["match_status"].value_counts())

    print("\nName matching results:")
    print(station_matching["name_match_type"].value_counts())

    duplicates = station_matching.duplicated(subset=["metro_lines", "station_names"]).sum()

    print("\nDuplicate station and line combinations:")
    print(duplicates)


    matched = (station_matching["match_status"] == "matched").sum()

    total = len(station_matching)

    print("\nUsable coordinate coverage:")
    print(f"{matched / total * 100:.2f}%")





