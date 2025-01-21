#!/bin/bash

# Run Python script to create the virtual environment
python3 create_venv.py

# Activate the virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error while trying to activate the venv!"
    exit 1
fi

# Run Python script to download language packs
python3 download_language_packs.py