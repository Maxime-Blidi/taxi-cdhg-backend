@echo off
REM Go up one directory
cd ..

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Go into the app folder
cd app

REM Run the Python script
python test3.py solve show

REM Pause to see output
pause
