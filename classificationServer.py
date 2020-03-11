# based on: https://gist.github.com/fahadysf/d80b99685ea3cfe3de4631f60e0136cc
# and https://gist.github.com/gnilchee/246474141cbe588eb9fb

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import json
import cgi
import random
import pandas as pd
import numpy as np
from SETTINGS import DBM_NA_FILL_VALUE, EXPORT_FEATURE_VECTOR_FILE_NAME, CLASSIFIER_JOBLIB_FILE_NAME, \
    SCALER_JOBLIB_FILE_NAME

STATUS_CODE_NOT_IMPLEMENTED = 501
STATUS_CODE_OK = 200
STATUS_CODE_BAD_REQUEST = 400
SERVER_PORT = 8111
PRESSURE_NA_FILL_VALUE = 974


def get_classification_result_as_dict(measurement_dict):
    measurement_df = pd.DataFrame([measurement_dict])
    # TODO: if you were to transform the timestamp into more sophisticated forms of data, you'd do it here
    feature_vector = pd.concat(
        [FEATURES_HEAD, measurement_df[FEATURES_HEAD.columns.intersection(measurement_df.columns)]], sort=False)
    # feature_vector["pressure"].fillna(PRESSURE_NA_FILL_VALUE, inplace=True)
    feature_vector.fillna(DBM_NA_FILL_VALUE, inplace=True)
    feature_vector = SCALER.transform(feature_vector)
    predictions = CLASSIFIER.predict_proba(feature_vector)
    result = dict(zip(CLASSIFIER.classes_, predictions[0].tolist()))
    for nodeId, probability in result.items():
        if probability == 0:
            result.pop(nodeId)
    index_of_highest_likelihood_prediction = np.where(predictions[0] == np.amax(predictions[0]))[0][0]
    result["prediction"] = CLASSIFIER.classes_[index_of_highest_likelihood_prediction]
    return result


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass


# This is the HTTP Handler which provides a simple JSON REST API
class MyRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(STATUS_CODE_OK)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    # this API doesn't handle GET responses
    def do_GET(self):
        self.send_response(STATUS_CODE_NOT_IMPLEMENTED)
        self.end_headers()
        return

    # POST answers the request for wifi data location prediction
    def do_POST(self):
        content_type, pdict = cgi.parse_header(self.headers['content-type'])

        # refuse to receive non-json content
        if content_type != 'application/json':
            self.send_response(STATUS_CODE_BAD_REQUEST)
            self.end_headers()
            return

        # read the wifi data and convert it into a python dictionary
        length = int(self.headers['content-length'])
        try:
            measurement = json.loads(self.rfile.read(length))
            # TODO: you might have to add .decode("utf-8") after length) depending on your operating system (will raise an error that bytearray isn't a string if you do not)
        except json.decoder.JSONDecodeError:
            self.send_response(STATUS_CODE_BAD_REQUEST)
            self.end_headers()
            return

        result = get_classification_result_as_dict(measurement)

        # send the answer as json
        self._set_headers()
        self.wfile.write(json.dumps(result).encode())


def run(server_class=ThreadingSimpleServer, handler_class=MyRequestHandler, port=SERVER_PORT):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd on port %d...' % port)
    httpd.serve_forever()


"""
curl: expected prediction is VielberthGebaeude.1.148
            curl --data "{\"timestamp\": 1582889652737, \"00:a0:57:2d:87:8d\": -74.0, \"00:a0:57:2d:dc:c9\": -89.0, \"00:a0:57:30:bd:c8\": -70.0, \"00:a0:57:30:be:8d\": -78.0, \"00:a0:57:30:bf:72\": -61.0, \"00:a0:57:30:bf:b2\": -67.0, \"00:a0:57:30:bf:f7\": -69.0, \"00:a0:57:30:c0:66\": -81.0, \"00:a0:57:30:ff:13\": -77.0, \"00:a0:57:31:02:6c\": -70.0, \"00:a0:57:31:03:66\": -71.0, \"00:a0:57:31:05:6c\": -67.0, \"00:a0:57:35:25:5c\": -63.0, \"00:a0:57:35:25:81\": -51.0, \"00:a0:57:35:25:9f\": -86.0, \"00:a0:57:35:27:b4\": -69.0, \"00:a0:57:35:2b:89\": -56.0, \"00:a0:57:35:2b:95\": -71.0, \"00:a0:57:35:2b:fa\": -53.0, \"00:a0:57:35:2c:68\": -67.0, \"00:a0:57:35:2c:92\": -53.0, \"00:a0:57:35:2d:63\": -55.0, \"00:a0:57:35:51:d2\": -60.0, \"00:a0:57:35:51:d5\": -64.0, \"00:a0:57:35:52:5f\": -65.0, \"00:a0:57:35:53:0b\": -62.0, \"00:a0:57:35:53:0f\": -74.0, \"00:a0:57:35:53:17\": -69.0, \"00:a0:57:35:91:af\": -66.0, \"00:a0:57:35:a0:45\": -76.0, \"02:a0:57:2d:87:8d\": -74.0, \"02:a0:57:2d:dc:c9\": -87.0, \"02:a0:57:30:bd:c8\": -68.0, \"02:a0:57:30:be:8d\": -78.0, \"02:a0:57:30:bf:72\": -56.0, \"02:a0:57:30:bf:b2\": -67.0, \"02:a0:57:30:c0:66\": -82.0, \"02:a0:57:31:02:6c\": -72.0, \"02:a0:57:31:03:66\": -73.0, \"02:a0:57:31:05:6c\": -68.0, \"02:a0:57:35:25:5c\": -61.0, \"02:a0:57:35:25:81\": -50.0, \"02:a0:57:35:25:9f\": -85.0, \"02:a0:57:35:27:b4\": -66.0, \"02:a0:57:35:2b:89\": -60.0, \"02:a0:57:35:2b:95\": -60.0, \"02:a0:57:35:2b:fa\": -52.0, \"02:a0:57:35:2c:92\": -51.0, \"02:a0:57:35:51:d2\": -63.0, \"02:a0:57:35:52:5f\": -64.0, \"02:a0:57:35:53:0b\": -64.0, \"02:a0:57:35:53:0f\": -73.0, \"02:a0:57:35:53:17\": -69.0, \"02:a0:57:35:91:af\": -67.0, \"02:a0:57:35:a0:45\": -78.0, \"06:a0:57:2d:87:8d\": -74.0, \"06:a0:57:30:bd:c8\": -70.0, \"06:a0:57:30:bf:72\": -55.0, \"06:a0:57:30:bf:b2\": -66.0, \"06:a0:57:30:c0:66\": -82.0, \"06:a0:57:30:ff:13\": -77.0, \"06:a0:57:31:02:6c\": -71.0, \"06:a0:57:31:03:66\": -73.0, \"06:a0:57:31:05:6c\": -68.0, \"06:a0:57:35:25:5c\": -70.0, \"06:a0:57:35:25:81\": -50.0, \"06:a0:57:35:25:9f\": -86.0, \"06:a0:57:35:27:b4\": -64.0, \"06:a0:57:35:2b:89\": -57.0, \"06:a0:57:35:2b:fa\": -51.0, \"06:a0:57:35:2c:92\": -52.0, \"06:a0:57:35:2d:63\": -51.0, \"06:a0:57:35:51:d2\": -58.0, \"06:a0:57:35:51:d5\": -81.0, \"06:a0:57:35:52:5f\": -64.0, \"06:a0:57:35:53:0b\": -63.0, \"06:a0:57:35:53:0f\": -74.0, \"06:a0:57:35:53:17\": -70.0, \"06:a0:57:35:91:af\": -67.0, \"06:a0:57:35:a0:45\": -66.0, \"0a:a0:57:2d:87:8d\": -85.0, \"0a:a0:57:30:bd:c8\": -69.0, \"0a:a0:57:30:be:8d\": -78.0, \"0a:a0:57:30:bf:72\": -55.0, \"0a:a0:57:30:bf:b2\": -66.0, \"0a:a0:57:30:bf:f7\": -72.0, \"0a:a0:57:30:c0:66\": -82.0, \"0a:a0:57:30:ff:13\": -77.0, \"0a:a0:57:31:02:6c\": -73.0, \"0a:a0:57:31:03:66\": -73.0, \"0a:a0:57:31:05:6c\": -69.0, \"0a:a0:57:35:25:5c\": -70.0, \"0a:a0:57:35:25:81\": -48.0, \"0a:a0:57:35:25:9f\": -86.0, \"0a:a0:57:35:27:b4\": -64.0, \"0a:a0:57:35:2b:89\": -60.0, \"0a:a0:57:35:2b:8e\": -87.0, \"0a:a0:57:35:2b:fa\": -51.0, \"0a:a0:57:35:2c:68\": -66.0, \"0a:a0:57:35:2c:92\": -51.0, \"0a:a0:57:35:2d:63\": -52.0, \"0a:a0:57:35:51:d2\": -57.0, \"0a:a0:57:35:51:d5\": -82.0, \"0a:a0:57:35:52:5f\": -64.0, \"0a:a0:57:35:53:0b\": -64.0, \"0a:a0:57:35:53:0f\": -75.0, \"0a:a0:57:35:53:17\": -71.0, \"0a:a0:57:35:91:af\": -66.0, \"0a:a0:57:35:a0:45\": -78.0, \"pressure\": 974.481}" --header "Content-Type: application/json" http://localhost:8111
"""

if __name__ == "__main__":
    from joblib import load

    # Initializing our global resources.
    FEATURES_HEAD = pd.read_csv(EXPORT_FEATURE_VECTOR_FILE_NAME)
    CLASSIFIER = load(CLASSIFIER_JOBLIB_FILE_NAME)
    SCALER = load(SCALER_JOBLIB_FILE_NAME)
    run()
