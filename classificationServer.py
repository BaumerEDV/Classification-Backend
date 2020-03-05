#based on: https://gist.github.com/fahadysf/d80b99685ea3cfe3de4631f60e0136cc
#and https://gist.github.com/gnilchee/246474141cbe588eb9fb

from multiprocessing import Pool, Manager

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import json
import cgi
import random, time

STATUS_CODE_NOT_IMPLEMENTED = 501
STATUS_CODE_OK = 200
STATUS_CODE_BAD_REQUEST = 400

# Resources to read
#
# http://stackoverflow.com/a/1239252/603280
# http://stackoverflow.com/questions/13689927/how-to-get-the-amount-of-work-left-to-be-done-by-a-python-multiprocessing-pool
#


def get_classification_result_as_dict(measurement_json):
    time.sleep(random.random()*10.0)
    result = {"prediction": random.random()}
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
            curl --data "{\"runtask\":\"true\", \"sessionid\":\"session-5\", \"number\":\"+\", \"repeatcount\": 100 }" \
            --header "Content-Type: application/json" http://localhost:8111
"""

if __name__ == "__main__":
    # Initializing our global resources.

    # Run the task broker.
    run()
