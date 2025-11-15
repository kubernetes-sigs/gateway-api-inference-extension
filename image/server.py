import threading
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

class MultiPortHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._handle_request()

    # FIX: Add this method to handle the test's POST requests
    def do_POST(self):
        # We need to read the body to keep the socket clean, even if we ignore it
        content_length = int(self.headers.get('Content-Length', 0))
        self.rfile.read(content_length)
        self._handle_request()

    def _handle_request(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        response = f"Handled by port: {self.server.server_port}\n"
        self.wfile.write(response.encode('utf-8'))
    
    def log_message(self, format, *args):
        pass

def start_server(port):
    print(f"Starting server on port {port}...")
    server = HTTPServer(('0.0.0.0', port), MultiPortHandler)
    server.serve_forever()

if __name__ == "__main__":
    start_port = 8000
    port_count = 1
    if len(sys.argv) >= 3:
        start_port = int(sys.argv[1])
        port_count = int(sys.argv[2])

    threads = []
    for i in range(port_count):
        port = start_port + i
        thread = threading.Thread(target=start_server, args=(port,))
        thread.daemon = True
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()