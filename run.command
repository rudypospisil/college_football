#!/bin/sh
# Start the menu with the project virtual environment.
cd "$(dirname "$0")" && exec .venv/bin/python app.py "$@"
