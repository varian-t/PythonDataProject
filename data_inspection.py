
import pandas as pd


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

    duplicate_stations = metro_df[
        metro_df.duplicated(
            subset=["months", "metro_lines", "station_names"],
            keep=False
        )
    ]
    passenger_columns = [
        col for col in metro_df.columns
        if "boarding passengers" in col
        or "alighting passengers" in col
    ]
    conflicting_groups = (duplicate_stations
        .groupby(["months", "metro_lines", "station_names"])[passenger_columns]
        .nunique()
        .gt(1)
        .any(axis=1)
    )

    print("METRO DUPLICATES")

    print("\nNumber of repeated station groups:")
    print(duplicate_stations.groupby(["months", "metro_lines", "station_names"]).ngroups)

    print("\nDifferent passenger counts:")
    print(conflicting_groups.sum())

    conflicting_keys = conflicting_groups[conflicting_groups].reset_index()[ ["months", "metro_lines", "station_names"]]

    conflicting_records = duplicate_stations.merge(
        conflicting_keys,
        on=["months", "metro_lines", "station_names"],
        how="inner")

    print("\nConflicting records:")

    print(
        conflicting_records[
            ["months", "metro_lines", "station_names",
             "data_upload_date"]
        ].head(20).to_string(index=False)
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
