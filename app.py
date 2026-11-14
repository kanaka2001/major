"""
VTU Major Project: Intelligent Medicinal Plant Recognition
Root Application Entrypoint: app.py

Launches the interactive Web Dashboard:
- Runs Gradio Blocks when 'gradio' is installed
- Runs standalone interactive web server on port 7860 as zero-dependency fallback

Usage:
  .venv/bin/python3 app.py
"""

import sys
import os

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.ui.app import launch_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    launch_app(port=port)

