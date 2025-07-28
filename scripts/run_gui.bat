@echo off
TITLE AI Gate - Knowledge Base Builder

REM --- Go to the script's directory ---
cd /d "%~dp0"

REM --- Activate the virtual environment ---
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

REM --- Check if activation was successful ---
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate the Python virtual environment.
    echo Please ensure the 'venv' directory exists in the parent folder.
    pause
    exit /b
)

REM --- Run the Python GUI script ---
echo Starting the Knowledge Base Builder GUI...
echo A new window will open in your web browser.
python knowledge_base_gui.py

REM --- Deactivate environment and exit ---
echo.
echo GUI has been closed. Press any key to exit this window.
call ..\venv\Scripts\deactivate.bat
pause