@echo off
echo ========================================
echo  PCB Defect Detection Web App
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "backend\venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run: cd backend ^&^& python -m venv venv
    pause
    exit /b 1
)

echo Activating virtual environment...
call backend\venv\Scripts\activate.bat

echo.
echo Installing/Updating dependencies...
pip install -q -r backend\requirements.txt

echo.
echo ========================================
echo  Starting Streamlit App...
echo ========================================
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run frontend\app.py

pause
