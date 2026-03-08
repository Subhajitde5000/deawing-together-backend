# Backend deployment notes

- The server binds to `HOST` and `PORT` environment variables (defaults to `0.0.0.0:8000`).
- On hosted platforms (Render, Heroku, etc.), you should set:
  - `PORT` (provided by platform)
  - `ALLOWED_ORIGINS` to a comma-separated list including the URL of your frontend (e.g. `https://drawing-togethers.netlify.app`).
    This value is used for CORS on HTTP endpoints and is also checked against the `Origin` header on WebSocket handshakes.
- For TLS support use `wss://` in the front-end WebSocket URL; the hosting provider will terminate HTTPS/TLS for you.
