from pathlib import Path
import pandas as pd
import numpy as np

from data_integration import (create_station_matching_table, create_school_station_comparison)
from data_inspection import (validate_station_matches, inspect_dataset, investigate_metro_duplicates, investigate_station_coordinates,investigate_school_records, investigate_station_matching)
from data_cleaning import (normalize_station_data, school_data_normalization, normalize_metro_data)


metro_csv_filepath = "data/raw/metrodata_translated.csv"
station_csv_filepath = "data/raw/station_coordinate.csv"
school_csv_filepath = "data/raw/school coordinate_translated.csv"
metro_df = pd.read_csv(metro_csv_filepath, encoding="cp949")
station_df = pd.read_csv(station_csv_filepath, encoding="utf-8-sig")
school_df = pd.read_csv(school_csv_filepath,encoding="utf-8-sig",dtype={"postal code": "string"})



#function to make an overall analysis w multiple commands

# uwu main code uwu


#################DATA INSPECTION CODE 
#inspect_dataset(metro_df, "Metro Passenger Data")
#inspect_dataset(station_df, "Station Coordinate Data")
#inspect_dataset(school_df, "School Coordinate Data")

#duplicates analysis
#investigate_metro_duplicates(metro_df)
#investigate_station_coordinates(station_df)
#investigate_school_records(school_df)

#normalization (functions in data_cleaning.py file)
stations_clean = normalize_station_data(station_df)
stations_clean.to_csv("data/processed/stations.csv", index=False, encoding="utf-8-sig")

schools_clean = school_data_normalization(school_df)
schools_clean.to_csv("data/processed/schools.csv", index=False, encoding="cp949")

metro_passengers_clean, metro_conflicts = normalize_metro_data(metro_df)
metro_conflicts.to_csv("data/processed/metro_duplicate_conflicts.csv",index=False,encoding="utf-8-sig")


validate_station_matches( metro_passengers_clean,stations_clean)


#data integration parts so like merging some tables
station_matching = create_station_matching_table(metro_passengers_clean, stations_clean)
station_matching.to_csv("data/integrated/station_matching.csv", index=False, encoding="utf-8-sig")

school_station_comparison = create_school_station_comparison(schools_clean,station_matching,metro_passengers_clean)
school_station_comparison.to_csv("data/integrated/station_matching.csv",index=False,encoding="utf-8-sig")
