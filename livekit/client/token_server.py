#!/usr/bin/env python3
"""
Simple HTTP server that generates LiveKit tokens and serves static files.
This is for DEVELOPMENT ONLY - in production, tokens should come from your backend!
"""

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from livekit import api

# LiveKit credentials (dev only!)
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "devsecret")

# Get the project root directory (parent of livekit_client)
PROJECT_ROOT = Path(__file__).parent.parent


class TokenHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler that generates tokens and serves files."""

    def __init__(self, *args, **kwargs):
        # Set the directory to the project root to serve files
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self):
        """Handle GET requests for serving files."""
        # Redirect /simple_client.html to /client/simple_client.html
        if self.path == "/simple_client.html":
            self.path = "/client/simple_client.html"
        # Let the parent class handle the file serving
        return super().do_GET()

    def do_POST(self):
        """Handle POST requests for token generation."""
        if self.path == "/token":
            # Read request body
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            room = data.get("room", "my-avatar-test-room")
            identity = data.get("identity", "user")

            # Generate token
            token = (
                api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                .with_identity(identity)
                .with_name(identity)
                .with_grants(
                    api.VideoGrants(
                        room_join=True, room=room, can_publish=True, can_subscribe=True
                    )
                )
                .to_jwt()
            )

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            response = json.dumps({"token": token})
            self.wfile.write(response.encode())

            print(f"✅ Generated token for {identity} in room {room}")
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header("Access-Control-Allow-Origin", "*")
        SimpleHTTPRequestHandler.end_headers(self)


def run(port=8000):
    """Start the server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, TokenHTTPRequestHandler)

    print(f"🚀 Token server running on http://localhost:{port}")
    print(f"📝 Serving files from: {PROJECT_ROOT}")
    print(f"🎫 Token endpoint: http://localhost:{port}/token")
    print(f"🌐 Client page: http://localhost:{port}/simple_client.html")
    print(f"   (or: http://localhost:{port}/client/simple_client.html)")
    print()
    print("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


if __name__ == "__main__":
    run()
