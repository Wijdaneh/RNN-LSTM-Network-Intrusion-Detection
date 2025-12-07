@echo off
echo ========================================
echo 🚀 ENTRAÎNEMENT DU MODÈLE LSTM UNSW-NB15
echo ========================================

REM Créer les dossiers nécessaires
if not exist "CSV Files" mkdir "CSV Files"
if not exist "models" mkdir "models"
if not exist "results" mkdir "results"

REM Vérifier l'existence des fichiers CSV
echo.
echo 🔍 Vérification des fichiers CSV...
dir "CSV Files\*.csv" > nul 2>&1
if errorlevel 1 (
    echo ❌ Aucun fichier CSV trouvé dans CSV Files/
    echo.
    echo 📥 Téléchargez les fichiers depuis:
    echo https://research.unsw.edu.au/projects/unsw-nb15-dataset
    echo.
    echo 📁 Placez-les dans le dossier CSV Files/
    pause
    exit /b 1
)

REM Installer les dépendances si nécessaire
echo.
echo 📦 Vérification des dépendances...
pip install tensorflow pandas scikit-learn matplotlib -q

REM Exécuter l'entraînement
echo.
echo 🏃 Démarrage de l'entraînement...
python train_unsw_final.py

REM Tester le modèle
echo.
echo 🔮 Test du modèle...
if exist "models\lstm_unsw_model.h5" (
    python predictor.py
)

echo.
echo ✅ Opération terminée!
pause