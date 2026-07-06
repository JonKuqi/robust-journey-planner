from __future__ import annotations

"""Trino connection helpers."""

from urllib.parse import urlparse

from .settings import ProjectSettings, get_settings


def create_trino_connection(settings: ProjectSettings | None = None):
    """Create an authenticated Trino connection from project settings."""
    settings = settings or get_settings()
    if not settings.trino_token:
        raise RuntimeError("EPFL_COM490_TOKEN is not set; cannot authenticate to Trino.")
    if not settings.trino_url:
        raise RuntimeError("TRINO_URL is not set; cannot connect to Trino.")

    try:
        from trino.auth import JWTAuthentication
        from trino.dbapi import connect
    except ImportError as exc:
        raise RuntimeError("The 'trino' package is required for CSA table preparation.") from exc

    trino_url = urlparse(settings.trino_url)
    return connect(
        host=trino_url.hostname,
        port=trino_url.port,
        auth=JWTAuthentication(settings.trino_token),
        http_scheme=trino_url.scheme,
        verify=True,
    )

