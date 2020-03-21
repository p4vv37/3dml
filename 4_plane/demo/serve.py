""" Serve files from diirectory

Source: https://stackoverflow.com/a/21957017/2919764
User: poke

Use this to serve models locally, so Tnesorflow can access them  

Example:
    From inside of location where models directory with models
    converted for tf.js are run:

        $ python serve.py
    
    And open http://0.0.0.0:8003/ in browser.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(CORSRequestHandler, self).end_headers()
        
httpd = HTTPServer(('localhost', 8003), CORSRequestHandler)
httpd.serve_forever()
