import os

MEASUREMENTS_FOLDER = os.fsencode("measurements")
PRESSURE_FILE_PREFIX = "pressure_"
WIFI_FILE_PREFIX = "wifi_"
COMBINED_DATA_EXPORT_FILE_NAME = "combined_data.csv"
UNI_WIFI_BSSID_REGEX_PATTERN = "\w\w:a0:57:\w\w:\w\w:\w\w"
SSID_EXCLUDE_REGEX_PATTERN = "conference\.uni-regensburg\.de"
EXPORT_FEATURE_VECTOR_FILE_NAME = "feature_vector_head.csv"
CLASSIFIER_JOBLIB_FILE_NAME = "classifier.joblib"
SCALER_JOBLIB_FILE_NAME = "scaler.joblib"
DBM_NA_FILL_VALUE = -100
CONFIRMATION_MEASUREMENT_EXPORT_FILE_NAME = "confirmation_data.csv"
CONFIRMATION_MEASUREMENTS_FOLDER = os.fsencode("confirmation_measurements")
