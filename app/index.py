"""
Vercel ASGI entry: export `app` here only.
`app/main.py` uses `fastapi_app` so Vercel does not see two `app` instances.
"""
from app.main import fastapi_app as app

__all__ = ["app"]
