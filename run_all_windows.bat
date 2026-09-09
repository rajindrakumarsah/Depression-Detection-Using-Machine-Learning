@echo off
python prepare_data.py
if errorlevel 1 exit /b %errorlevel%
python train_baseline.py
if errorlevel 1 exit /b %errorlevel%
python train_model.py
if errorlevel 1 exit /b %errorlevel%
streamlit run app.py
