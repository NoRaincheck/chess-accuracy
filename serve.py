#!/usr/bin/env python3
"""Dev server for docs/ with correct WASM MIME type and COOP/COEP headers.

Usage:
    python serve.py          # default port 8000
    PORT=3000 python serve.py
"""

import http.server
import os
import sys

PORT = int(os.environ.get("PORT", 8000))
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


class WasmHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    with http.server.HTTPServer(("", PORT), WasmHandler) as httpd:
        print(f"Serving {DIRECTORY} on http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
