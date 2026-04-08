# server/app.py
# OpenEnv-compliant server entry point

import uvicorn
from app.server import app


def main():
    """Main entry point for OpenEnv server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
    )


if __name__ == "__main__":
    main()