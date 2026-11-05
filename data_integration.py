import pandas as pd
import numpy as np

def create_station_matching_table(metro_df, station_df):

    #unique station-line combinations from metro data
    metro_stations = metro_df[ ["metro_lines", "station_names", "station_match_key"]].drop_duplicates().copy()

    #coordinate pairs with valid coordinates
    coordinate_pairs = (station_df.dropna(subset=["latitude", "longitude"])[["station_match_key", "latitude", "longitude"]].drop_duplicates())

    # Check if station  has multiple coordinate 
    coordinate_pair_counts = (coordinate_pairs.groupby("station_match_key").size())
    ambiguous_stations = coordinate_pair_counts[coordinate_pair_counts > 1]

    if not ambiguous_stations.empty:
        raise ValueError("Station has multipl coordinates: "f"{ambiguous_stations.index.tolist()}")

    #one coordinate lookup record per station name
    coordinate_lookup = (station_df
        .groupby("station_match_key", as_index=False)
        .agg(
            coordinate_station_names=(
                "station_names",
                lambda x: " / ".join(sorted(set(x.dropna())))),
            coordinate_lines=(
                "metro_lines",
                lambda x: ", ".join(sorted(set(x.dropna())))),
            latitude=("latitude", "first"),
            longitude=("longitude", "first")
        ))

    # Match metro stations with coordinate records
    station_matching = metro_stations.merge(
        coordinate_lookup,
        on="station_match_key",
        how="left",
        indicator=True,
        validate="many_to_one")


    name_found = station_matching["_merge"] == "both"
    coordinates_found = (station_matching["latitude"].notna()& station_matching["longitude"].notna())

    station_matching["match_status"] = "station_name_not_found"
    station_matching.loc[name_found & coordinates_found,"match_status"] = "matched"
    station_matching.loc[name_found & ~coordinates_found,"match_status"] = "coordinates_unavailable"

    #check if exact match
    exact_name_pairs = (station_df[["station_match_key", "station_names"]].drop_duplicates().assign(exact_name_found=True))
    station_matching = station_matching.merge(
        exact_name_pairs,
        on=["station_match_key", "station_names"],
        how="left",
        validate="many_to_one"
    )

    station_matching["name_match_type"] = "unmatched"
    station_matching.loc[name_found & station_matching["exact_name_found"].eq(True),"name_match_type"] = "exact"
    station_matching.loc[ name_found & station_matching["exact_name_found"].ne(True),"name_match_type"] = "standardized_alias"

    #remove temp columns
    station_matching = station_matching.drop(columns=["_merge", "exact_name_found"])

    #arrange columns
    station_matching = station_matching[
        [
            "metro_lines",
            "station_names",
            "station_match_key",
            "coordinate_station_names",
            "coordinate_lines",
            "latitude",
            "longitude",
            "name_match_type",
            "match_status"
        ]]

    #sort records
    station_matching = station_matching.sort_values(by=["metro_lines", "station_names"]).reset_index(drop=True)
    return station_matching


def create_school_station_comparison(schools_df,station_matching,metro_df):

    schools = schools_df.copy().reset_index(drop=True)

    #we only keep ones w valid coordinates
    valid_stations = station_matching[station_matching["match_status"] == "matched"].copy()

    #one record per physical location
    station_locations = (valid_stations
        .groupby("station_match_key", as_index=False)
        .agg(nearest_station_name=(
                "coordinate_station_names",
                "first"
            ),
            nearest_station_lines=(
                "metro_lines",
                lambda x: ", ".join(sorted(set(x.dropna())))
            ),
            coordinate_lines=(
                "coordinate_lines",
                "first"
            ),
            station_latitude=("latitude", "first"),
            station_longitude=("longitude", "first")
        )
    )

    # Check that every station has only one coordinate pair
    coordinate_counts = (valid_stations[ ["station_match_key", "latitude", "longitude"]]
        .drop_duplicates()
        .groupby("station_match_key")
        .size()
    )

    if (coordinate_counts > 1).any():
        raise ValueError(
            "Multiple coordinate pairs"
        )

    school_latitudes = schools["latitude"].to_numpy(dtype=float)
    school_longitudes = schools["longitude"].to_numpy(dtype=float)
    station_latitudes = np.radians( station_locations["station_latitude"].to_numpy(dtype=float))
    station_longitudes = np.radians(station_locations["station_longitude"].to_numpy(dtype=float))

    #we keep the nearest stations
    nearest_indices = []
    nearest_distances = []

    #calculating distances in batches cause we got a lot to go thru
    batch_size = 1000

    for start in range(0, len(schools), batch_size):
        end = start + batch_size
        #get schoool coords
        school_lat = np.radians( school_latitudes[start:end])[:, np.newaxis]
        school_lon = np.radians( school_longitudes[start:end])[:, np.newaxis]

        #calculate diff
        delta_lat = station_latitudes[np.newaxis, :] - school_lat
        delta_lon = station_longitudes[np.newaxis, :] - school_lon

        #Haversine formula
        a = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(school_lat)
            * np.cos(station_latitudes[np.newaxis, :])
            * np.sin(delta_lon / 2) ** 2 )

        earth_radius = 6371000
        distances = (2 * earth_radius* np.arcsin(np.sqrt(np.clip(a, 0, 1))))
        batch_nearest_indices = distances.argmin(axis=1)

        batch_nearest_distances = distances[ np.arange(len(distances)),batch_nearest_indices]
        nearest_indices.extend(batch_nearest_indices.tolist())
        nearest_distances.extend(batch_nearest_distances.tolist())

    #nearest station information
    nearest_stations = (station_locations.iloc[nearest_indices].reset_index(drop=True))

    #add nearest station information to school records
    schools["nearest_station_name"] = (nearest_stations["nearest_station_name"].to_numpy())
    schools["nearest_station_lines"] = (nearest_stations["nearest_station_lines"].to_numpy())
    schools["station_match_key"] = (nearest_stations["station_match_key"].to_numpy())
    schools["station_latitude"] = (nearest_stations["station_latitude"].to_numpy())
    schools["station_longitude"] = (nearest_stations["station_longitude"].to_numpy())

    #distance in meters
    schools["distance_to_nearest_station_m"] = np.round(nearest_distances,1)

    #groups based on distance
    schools["within_500m"] = (schools["distance_to_nearest_station_m"] <= 500)
    schools["within_1000m"] = ( schools["distance_to_nearest_station_m"] <= 1000)

    #years where passenger data is availabl
    metro_years = set(pd.to_datetime(metro_df["month"]).dt.year.unique())
    schools["metro_data_available_for_year"] = (schools["year"].isin(metro_years))

    return schools

