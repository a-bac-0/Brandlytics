"""
ASGI entrypoint for the FastAPI app.
Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import yaml

# Reuse the existing FastAPI application defined in api/app.py
from api.app import app  # noqa: F401


def configure_logging() -> None:
    """Configure logging based on config/logging settings, with safe fallbacks."""
    try:
        cfg_path = Path("config/config.yaml")
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
        log_cfg = (cfg or {}).get("logging", {})
        level_name = str(log_cfg.get("level", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)
        fmt = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logging.basicConfig(level=level, format=fmt)
    except Exception:
        # Fallback
        logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    configure_logging()

    # Load API host/port from config + allow env overrides
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    api_cfg = (cfg or {}).get("api", {})

    host = os.getenv("API_HOST", api_cfg.get("host", "0.0.0.0"))
    port = int(os.getenv("API_PORT", api_cfg.get("port", 8000)))
    reload_flag = bool(api_cfg.get("debug", False))

    # Start Uvicorn using the import string to support --reload
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload_flag,
    )
