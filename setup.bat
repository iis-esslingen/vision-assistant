@echo off

REM Run Python script to create the virtual environment
python create_venv.py
if errorlevel 1 (
    echo Error while running create_venv.py!
    exit /b 1
)

REM Activate the virtual environment
call venv\Scripts\activate
if errorlevel 1 (
    echo Error while trying to activate the venv!
    exit /b 1
)

REM Run Python script to download language packs
python download_language_packs.py
if errorlevel 1 (
    echo Error while running download_language_packs.py!
    exit /b 1
)

echo All tasks completed successfully!
