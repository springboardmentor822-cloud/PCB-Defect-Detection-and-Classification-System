#!/bin/bash

echo "========================================"
echo " PCB Defect Detection Web App"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "backend/venv/bin/activate" ]; then
    echo "Virtual environment not found!"
    echo "Please run: cd backend && python -m venv venv"
    exit 1
fi

echo "Activating virtual environment..."
source backend/venv/bin/activate

echo ""
echo "Installing/Updating dependencies..."
pip install -q -r backend/requirements.txt

echo ""
echo "========================================"
echo " Starting Streamlit App..."
echo "========================================"
echo ""
echo "The app will open in your browser at:"
echo "http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

streamlit run frontend/app.py
