#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend.
Run: python serve.py
Then open http://127.0.0.1:8080 in your browser.
"""

import http.server
import socketserver
from pathlib import Path

PORT = 8080
FRONTEND_DIR = Path(__file__).parent / "frontend"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"✨ Frontend server running at http://127.0.0.1:{PORT}")
        print(f"📁 Serving from: {FRONTEND_DIR}")
        print(f"⚠️  Make sure backend is running: uvicorn backend.app:app --reload")
        print("Press Ctrl+C to stop.")
        httpd.serve_forever()
