@echo off
echo ========================================
echo LANCEMENT INTERFACE STREAMLIT
echo ========================================
echo.

REM Activer l'environnement virtuel si nécessaire
REM call venv\Scripts\activate

echo Lancement de l'application Streamlit...
streamlit run app_unsw.py

pause