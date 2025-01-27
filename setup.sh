#!/bin/bash

# Run Python script to create the virtual environment
python3 create_venv.py

# Activate the virtual environment
echo "Activating virtual environment 'venv'..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error while trying to activate the virtual environment!"
    exit 1
fi
echo -e "Virtual environment 'venv' is now active.\n"

# Run Python script to download language packs
python3 download_language_packs.py