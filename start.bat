@echo off
cd /d "%~dp0"
if not exist .venv-pytorch\Scripts\python.exe (
  echo Create a Python 3.12 virtual environment and install requirements.txt first.
  pause
  exit /b 1
)
.venv-pytorch\Scripts\python.exe -m streamlit run streamlit_app\app.py --server.address 127.0.0.1 --server.port 8501
pause
