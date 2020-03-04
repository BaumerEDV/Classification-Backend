#based on: https://gist.github.com/fahadysf/d80b99685ea3cfe3de4631f60e0136cc
#and https://gist.github.com/gnilchee/246474141cbe588eb9fb

from multiprocessing import Pool, Manager

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import json
import cgi
import random, time

# Resources to read
#
# http://stackoverflow.com/a/1239252/603280
# http://stackoverflow.com/questions/13689927/how-to-get-the-amount-of-work-left-to-be-done-by-a-python-multiprocessing-pool
#


def another_task():
    time.sleep(random.random()*10.0)
    return "Task Result"


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass


# This is the HTTP Server which provides a simple JSON REST API
class MyRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    # GET sends back the complete contents of the manager dictionary 'd' as JSON.
    # This can be modified to any desired response (should be JSON)
    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps(d))

    # POST echoes the message adding a JSON field
    def do_POST(self):
        #ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
        ctype, pdict = cgi.parse_header(self.headers['content-type'])

        # refuse to receive non-json content
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return

        # read the message and convert it into a python dictionary
        #length = int(self.headers.getheader('content-length'))
        length = int(self.headers['content-length'])
        message = json.loads(self.rfile.read(length))

        #if message.has_key('runtask'):
        if 'runtask' in message:
            """
            To run a new task simply send the following JSON as POST:
            {"runtask": true, "sessionid": "ANY-UNIQUE-NAME-FOR-YOUR-TASK", 'arg1', 'repeatcount'}
            Curl Syntax:
            curl --data "{\"runtask\":\"true\", \"sessionid\":\"session-5\", \"number\":\"+\", \"repeatcount\": 100 }" \
            --header "Content-Type: application/json" http://localhost:8111
            """
            print("Starting task with %s, %s, %s" % (message['sessionid'], message['number'], message['repeatcount']))
            #result = p.apply_async(task, (d, message['sessionid'], message['number'], message['repeatcount']))
            result = another_task()
            message['task-result'] = result
        #elif message.has_key('sessionid'):
        elif 'sessionid' in message:
            """
            To see the status of a currently running task (or completed task) simpley POST the following JSON
            {"sessionid": "THE-UNIQUE-NAME-FOR-YOUR-TASK"}
            Curl Syntax:
            curl --data "{\"sessionid\":\"session-5\"}" --header "Content-Type: application/json" http://localhost:8111
            """
            message['status'] = d[message['sessionid']]

        # send the message back
        self._set_headers()
        self.wfile.write(json.dumps(message).encode())


def run(server_class=ThreadingSimpleServer, handler_class=MyRequestHandler, port=8111):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)

    print('Starting httpd on port %d...' % port)
    httpd.serve_forever()


if __name__ == "__main__":
    # Run the task broker.
    # Initializing our global resources.
    global p, m, d
    p = Pool()
    m = Manager()
    d = m.dict()
    run()