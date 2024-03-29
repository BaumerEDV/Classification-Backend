import os
import pandas as pd
import glob
from SETTINGS import MEASUREMENTS_FOLDER, PRESSURE_FILE_PREFIX, WIFI_FILE_PREFIX, COMBINED_DATA_EXPORT_FILE_NAME, \
    UNI_WIFI_BSSID_REGEX_PATTERN, SSID_EXCLUDE_REGEX_PATTERN


def get_file_in_directory_with_prefix(directory, prefix):
    directory_decoded = os.fsdecode(directory)
    for filename in glob.glob(directory_decoded + "/" + prefix + "*"):
        return filename


def turn_wifi_readings_into_master_form(wifi_df):
    result = wifi_df.pivot(index='timestamp', columns='bssid', values='dbm').reset_index().rename_axis(None, axis=1)
    # Source: https://stackoverflow.com/questions/60501660/im-trying-to-reshape-my-data-from-a-long-format-to-something-that-isnt-quite-t?noredirect=1#comment107032085_60501660
    return result


def remove_bssids_not_from_university(wifi_df):
    filter = wifi_df['bssid'].str.contains(UNI_WIFI_BSSID_REGEX_PATTERN)
    return wifi_df[filter]


def remove_unwanted_ssids(wifi_df):
    filter = wifi_df['ssid'].str.contains(SSID_EXCLUDE_REGEX_PATTERN)
    return wifi_df[~filter]


def combine_pressure_readings_and_wifi_data(pressure_df, wifi_df):
    def get_pressure_for_data_row(row):
        result = pressure_df.iloc[(pressure_df['timestamp'] - row['timestamp']).abs().argsort()[:1]]['pressure']
        result = result.iloc[0]
        return result
        # Source https://codereview.stackexchange.com/questions/204549/lookup-closest-value-in-pandas-dataframe

    wifi_df['pressure'] = wifi_df.apply(lambda row: get_pressure_for_data_row(row), axis=1)
    return wifi_df


def apply_nodeId_to_df(nodeId, df):
    df['nodeId'] = nodeId


if __name__ == "__main__":
    master_df = pd.DataFrame(columns=['timestamp', 'nodeId', 'pressure'])

    for measurement_series_name in os.listdir(MEASUREMENTS_FOLDER):
        measurement_series_directory = os.path.join(MEASUREMENTS_FOLDER, os.fsencode(measurement_series_name))
        for measurement_name in os.listdir(measurement_series_directory):
            measurement_path = os.path.join(measurement_series_directory, measurement_name)
            node_name, cardinal_direction = os.fsdecode(measurement_name).rsplit(".", 1)
            wifi_filepath = get_file_in_directory_with_prefix(measurement_path, WIFI_FILE_PREFIX)
            pressure_filepath = get_file_in_directory_with_prefix(measurement_path, PRESSURE_FILE_PREFIX)
            wifi_raw_df = pd.read_csv(wifi_filepath, sep=";")
            wifi_raw_df = remove_bssids_not_from_university(wifi_raw_df)
            wifi_raw_df = remove_unwanted_ssids(wifi_raw_df)
            pressure_raw_df = pd.read_csv(pressure_filepath, sep=";")
            wifi_master_form_df = turn_wifi_readings_into_master_form(wifi_raw_df)
            wifi_master_form_df = combine_pressure_readings_and_wifi_data(pressure_raw_df, wifi_master_form_df)
            apply_nodeId_to_df(node_name, wifi_master_form_df)
            master_df = pd.merge(master_df, wifi_master_form_df, how='outer')

    master_df.to_csv(COMBINED_DATA_EXPORT_FILE_NAME)
