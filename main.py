import pandas as pd
import numpy as np

from data_inspection import (inspect_dataset, investigate_metro_duplicates, investigate_station_coordinates,investigate_school_records)

metro_csv_filepath = "data/raw/metrodata_translated.csv"
station_csv_filepath = "data/raw/station_coordinate.csv"
school_csv_filepath = "data/raw/school_coordinate.csv"
metro_df = pd.read_csv(metro_csv_filepath, encoding="cp949")
station_df = pd.read_csv(station_csv_filepath, encoding="utf-8-sig")
school_df = pd.read_csv(school_csv_filepath, encoding="cp949")



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