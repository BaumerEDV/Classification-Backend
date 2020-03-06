#based on: https://gist.github.com/fahadysf/d80b99685ea3cfe3de4631f60e0136cc
#and https://gist.github.com/gnilchee/246474141cbe588eb9fb

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import json
import cgi
import random
import pandas as pd
import numpy as np

STATUS_CODE_NOT_IMPLEMENTED = 501
STATUS_CODE_OK = 200
STATUS_CODE_BAD_REQUEST = 400
EXPORT_FEATURE_VECTOR_FILE_NAME = "feature_vector_head.csv"

# Resources to read
#
# http://stackoverflow.com/a/1239252/603280
# http://stackoverflow.com/questions/13689927/how-to-get-the-amount-of-work-left-to-be-done-by-a-python-multiprocessing-pool
#


def get_classification_result_as_dict(measurement_dict):
    #time.sleep(random.random()*2.0)
    result = {"prediction": random.random()}
    measurement_df = pd.DataFrame([measurement_dict])
    feature_vector = pd.concat([features_head, measurement_df[features_head.columns.intersection(measurement_df.columns)]], sort=False)
    feature_vector["pressure"].fillna(974, inplace=True)
    feature_vector.fillna(-100, inplace=True)
    feature_vector = scaler.transform(feature_vector)
    prediction = classifier.predict_proba(feature_vector)
    #feature_vector.to_csv("temp.csv")
    result = dict(zip(classifier.classes_, prediction[0].tolist()))
    index_of_highest_likelihood_prediction = np.where(prediction[0] == np.amax(prediction[0]))[0][0]
    result["prediction"] = classifier.classes_[index_of_highest_likelihood_prediction]
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
        measurement = json.loads(self.rfile.read(length))

        result = get_classification_result_as_dict(measurement)

        # send the answer as json
        self._set_headers()
        self.wfile.write(json.dumps(result).encode())


def run(server_class=ThreadingSimpleServer, handler_class=MyRequestHandler, port=8111):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)

    print('Starting httpd on port %d...' % port)
    httpd.serve_forever()


"""
            To run a new task simply send the following JSON as POST:
            {"runtask": true, "sessionid": "ANY-UNIQUE-NAME-FOR-YOUR-TASK", 'arg1', 'repeatcount'}
            Curl Syntax:
            curl --data "{\"model\":\"MD-5\", \"00:a0:57:30:bd:c8\":\"-55\" }" --header "Content-Type: application/json" http://localhost:8111
"""

"""
better curl: expected prediction is VielberthGebaeude.1.148
            curl --data "{\"timestamp\": 1582889652737, \"00:a0:57:2d:87:8d\": -74.0, \"00:a0:57:2d:dc:c9\": -89.0, \"00:a0:57:30:bd:c8\": -70.0, \"00:a0:57:30:be:8d\": -78.0, \"00:a0:57:30:bf:72\": -61.0, \"00:a0:57:30:bf:b2\": -67.0, \"00:a0:57:30:bf:f7\": -69.0, \"00:a0:57:30:c0:66\": -81.0, \"00:a0:57:30:ff:13\": -77.0, \"00:a0:57:31:02:6c\": -70.0, \"00:a0:57:31:03:66\": -71.0, \"00:a0:57:31:05:6c\": -67.0, \"00:a0:57:35:25:5c\": -63.0, \"00:a0:57:35:25:81\": -51.0, \"00:a0:57:35:25:9f\": -86.0, \"00:a0:57:35:27:b4\": -69.0, \"00:a0:57:35:2b:89\": -56.0, \"00:a0:57:35:2b:95\": -71.0, \"00:a0:57:35:2b:fa\": -53.0, \"00:a0:57:35:2c:68\": -67.0, \"00:a0:57:35:2c:92\": -53.0, \"00:a0:57:35:2d:63\": -55.0, \"00:a0:57:35:51:d2\": -60.0, \"00:a0:57:35:51:d5\": -64.0, \"00:a0:57:35:52:5f\": -65.0, \"00:a0:57:35:53:0b\": -62.0, \"00:a0:57:35:53:0f\": -74.0, \"00:a0:57:35:53:17\": -69.0, \"00:a0:57:35:91:af\": -66.0, \"00:a0:57:35:a0:45\": -76.0, \"02:a0:57:2d:87:8d\": -74.0, \"02:a0:57:2d:dc:c9\": -87.0, \"02:a0:57:30:bd:c8\": -68.0, \"02:a0:57:30:be:8d\": -78.0, \"02:a0:57:30:bf:72\": -56.0, \"02:a0:57:30:bf:b2\": -67.0, \"02:a0:57:30:c0:66\": -82.0, \"02:a0:57:31:02:6c\": -72.0, \"02:a0:57:31:03:66\": -73.0, \"02:a0:57:31:05:6c\": -68.0, \"02:a0:57:35:25:5c\": -61.0, \"02:a0:57:35:25:81\": -50.0, \"02:a0:57:35:25:9f\": -85.0, \"02:a0:57:35:27:b4\": -66.0, \"02:a0:57:35:2b:89\": -60.0, \"02:a0:57:35:2b:95\": -60.0, \"02:a0:57:35:2b:fa\": -52.0, \"02:a0:57:35:2c:92\": -51.0, \"02:a0:57:35:51:d2\": -63.0, \"02:a0:57:35:52:5f\": -64.0, \"02:a0:57:35:53:0b\": -64.0, \"02:a0:57:35:53:0f\": -73.0, \"02:a0:57:35:53:17\": -69.0, \"02:a0:57:35:91:af\": -67.0, \"02:a0:57:35:a0:45\": -78.0, \"06:a0:57:2d:87:8d\": -74.0, \"06:a0:57:30:bd:c8\": -70.0, \"06:a0:57:30:bf:72\": -55.0, \"06:a0:57:30:bf:b2\": -66.0, \"06:a0:57:30:c0:66\": -82.0, \"06:a0:57:30:ff:13\": -77.0, \"06:a0:57:31:02:6c\": -71.0, \"06:a0:57:31:03:66\": -73.0, \"06:a0:57:31:05:6c\": -68.0, \"06:a0:57:35:25:5c\": -70.0, \"06:a0:57:35:25:81\": -50.0, \"06:a0:57:35:25:9f\": -86.0, \"06:a0:57:35:27:b4\": -64.0, \"06:a0:57:35:2b:89\": -57.0, \"06:a0:57:35:2b:fa\": -51.0, \"06:a0:57:35:2c:92\": -52.0, \"06:a0:57:35:2d:63\": -51.0, \"06:a0:57:35:51:d2\": -58.0, \"06:a0:57:35:51:d5\": -81.0, \"06:a0:57:35:52:5f\": -64.0, \"06:a0:57:35:53:0b\": -63.0, \"06:a0:57:35:53:0f\": -74.0, \"06:a0:57:35:53:17\": -70.0, \"06:a0:57:35:91:af\": -67.0, \"06:a0:57:35:a0:45\": -66.0, \"0a:a0:57:2d:87:8d\": -85.0, \"0a:a0:57:30:bd:c8\": -69.0, \"0a:a0:57:30:be:8d\": -78.0, \"0a:a0:57:30:bf:72\": -55.0, \"0a:a0:57:30:bf:b2\": -66.0, \"0a:a0:57:30:bf:f7\": -72.0, \"0a:a0:57:30:c0:66\": -82.0, \"0a:a0:57:30:ff:13\": -77.0, \"0a:a0:57:31:02:6c\": -73.0, \"0a:a0:57:31:03:66\": -73.0, \"0a:a0:57:31:05:6c\": -69.0, \"0a:a0:57:35:25:5c\": -70.0, \"0a:a0:57:35:25:81\": -48.0, \"0a:a0:57:35:25:9f\": -86.0, \"0a:a0:57:35:27:b4\": -64.0, \"0a:a0:57:35:2b:89\": -60.0, \"0a:a0:57:35:2b:8e\": -87.0, \"0a:a0:57:35:2b:fa\": -51.0, \"0a:a0:57:35:2c:68\": -66.0, \"0a:a0:57:35:2c:92\": -51.0, \"0a:a0:57:35:2d:63\": -52.0, \"0a:a0:57:35:51:d2\": -57.0, \"0a:a0:57:35:51:d5\": -82.0, \"0a:a0:57:35:52:5f\": -64.0, \"0a:a0:57:35:53:0b\": -64.0, \"0a:a0:57:35:53:0f\": -75.0, \"0a:a0:57:35:53:17\": -71.0, \"0a:a0:57:35:91:af\": -66.0, \"0a:a0:57:35:a0:45\": -78.0, \"pressure\": 974.481}" --header "Content-Type: application/json" http://localhost:8111
"""

if __name__ == "__main__":
    # Initializing our global resources.
    global classifier, scaler, features_head
    features_head = pd.read_csv(EXPORT_FEATURE_VECTOR_FILE_NAME)
    from joblib import load
    classifier = load("classifier.joblib")
    scaler = load("scaler.joblib")
    # Run the task broker.
    run()
