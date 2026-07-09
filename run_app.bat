@echo off
cd /d "%~dp0"
echo Installing libraries into Python 3.13...
py -3.13 -m pip install streamlit pandas numpy scikit-learn matplotlib
echo.
echo Starting FC Barcelona Scouting Tool...
py -3.13 -m streamlit run barca_app.py
pause
