import os
import pandas as pd
import glob
import numpy as np

MEASUREMENTS_FOLDER = os.fsencode("measurements")
PRESSURE_FILE_PREFIX = "pressure_"
WIFI_FILE_PREFIX = "wifi_"


def get_file_in_directory_with_prefix(directory, prefix):
    directory_decoded = os.fsdecode(directory)
    for filename in glob.glob(directory_decoded + "/" + prefix + "*"):
        return filename


def turn_wifi_readings_into_master_form(wifi_df):
    result=wifi_df.pivot(index='timestamp', columns='bssid', values='dbm').reset_index().rename_axis(None, axis=1)
    #Source: https://stackoverflow.com/questions/60501660/im-trying-to-reshape-my-data-from-a-long-format-to-something-that-isnt-quite-t?noredirect=1#comment107032085_60501660
    return result


def combine_pressure_readings_and_wifi_data(pressure_df, wifi_df):
    def get_pressure_closest_to_timestamp(timestamp):
        return pressure_df.iloc[(pressure_df['timestamp']-timestamp).abs().argsort()[:1]]
        # Source https://codereview.stackexchange.com/questions/204549/lookup-closest-value-in-pandas-dataframe
    #wifi_df['pressure'] = get_pressure_closest_to_timestamp(wifi_df['timestamp'])
    return wifi_df

master_df = pd.DataFrame(columns=['timestamp', 'nodeId', 'pressure'])

for measurement_series_name in os.listdir(MEASUREMENTS_FOLDER):
    # TODO: check if file, if yes skip
    measurement_series_directory = os.path.join(MEASUREMENTS_FOLDER, os.fsencode(measurement_series_name))
    for measurement_name in os.listdir(measurement_series_directory):
        measurement_path = os.path.join(measurement_series_directory, measurement_name)
        measurement_name = os.fsdecode(measurement_name)
        node_name, cardinal_direction = measurement_name.rsplit(".", 1)
        wifi_filepath = get_file_in_directory_with_prefix(measurement_path, WIFI_FILE_PREFIX)
        pressure_filepath = get_file_in_directory_with_prefix(measurement_path, PRESSURE_FILE_PREFIX)
        wifi_raw_df = pd.read_csv(wifi_filepath, sep=";")
        pressure_raw_df = pd.read_csv(pressure_filepath, sep=";")
        wifi_master_form_df = turn_wifi_readings_into_master_form(wifi_raw_df)
        pd.set_option('display.max_columns', None)
        #print(wifi_master_form_df)
        wifi_master_form_df = combine_pressure_readings_and_wifi_data(pressure_raw_df, wifi_master_form_df)
        exit(0)