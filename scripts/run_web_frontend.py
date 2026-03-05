"""
Run the Flask frontend locally.

Usage:
    python scripts/run_web_frontend.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.web_frontend.app import create_app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=True)

