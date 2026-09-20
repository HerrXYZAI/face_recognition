@echo off
REM ============================================================
REM  run.bat - face_pipeline starten (Windows / Anaconda)
REM
REM  Richtet beim ersten Start automatisch eine Conda-Umgebung
REM  "face_pipeline" ein (Python 3.11 + Abhaengigkeiten), oeffnet
REM  danach ein Menue fuer die vier Schritte:
REM    1) Gesichter aus Lightroom exportieren
REM    2) Klassifikator trainieren
REM    3) Gesichtserkennung auf die Fotobibliothek anwenden
REM    4) Ergebnisse als XMP nach Lightroom zurueckschreiben
REM
REM  Kann direkt per Doppelklick gestartet werden - eine bereits
REM  geoeffnete Anaconda Prompt ist NICHT erforderlich.
REM
REM  Lieber Docker statt Conda? Siehe README.md /
REM  docker\docker-compose.yml.
REM ============================================================

setlocal enabledelayedexpansion

set ENV_NAME=face_pipeline
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

REM Python soll keine .pyc-Bytecode-Caches (__pycache__) mehr schreiben.
REM Auf manchen Laufwerken (v.a. externe/USB-Platten mit FAT32/exFAT, nur
REM 2-Sekunden-Zeitstempel-Aufloesung) kann Python sonst faelschlich eine
REM veraltete .pyc-Datei fuer aktuell halten, obwohl die .py-Datei laengst
REM aktualisiert wurde - das fuehrt zu "der Fix wirkt nicht"-Symptomen.
set PYTHONDONTWRITEBYTECODE=1

echo.
echo ============================================================
echo  face_pipeline - Setup
echo ============================================================
echo.

REM --- 1. Conda finden und noetigenfalls in dieser Konsole bootstrappen ---
set "CONDA_BASE="
where conda >nul 2>nul
if not errorlevel 1 goto :conda_on_path

echo [INFO] 'conda' ist in dieser Konsole noch nicht bekannt - suche Anaconda/Miniconda ...
set CONDA_CANDIDATES=%USERPROFILE%\anaconda3;%USERPROFILE%\miniconda3;%USERPROFILE%\Anaconda3;%USERPROFILE%\Miniconda3;%ProgramData%\anaconda3;%ProgramData%\miniconda3;C:\ProgramData\Anaconda3;C:\ProgramData\Miniconda3

for %%P in ("%CONDA_CANDIDATES:;=" "%") do (
    if exist "%%~P\Scripts\activate.bat" (
        set "CONDA_BASE=%%~P"
        goto :conda_bootstrap
    )
)
goto :no_conda

:conda_bootstrap
echo [OK] Anaconda/Miniconda gefunden unter: %CONDA_BASE%
call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_BASE%"
if errorlevel 1 goto :no_conda
goto :conda_ready

:conda_on_path
echo [OK] conda bereits im PATH gefunden.
for /f "delims=" %%B in ('conda info --base 2^>nul') do set "CONDA_BASE=%%B"
goto :conda_ready

REM --- 2. Conda-Umgebung "face_pipeline" finden oder anlegen --------------
:conda_ready
if not defined CONDA_BASE goto :no_conda

set "PYEXE="
if exist "%CONDA_BASE%\envs\%ENV_NAME%\python.exe" (
    set "PYEXE=%CONDA_BASE%\envs\%ENV_NAME%\python.exe"
    echo [OK] Conda-Umgebung "%ENV_NAME%" gefunden.
    goto :env_ready
)

echo [INFO] Conda gefunden, aber Umgebung "%ENV_NAME%" existiert noch nicht.
echo        Erstelle Conda-Umgebung "%ENV_NAME%" mit Python 3.11 ...
call conda create -n %ENV_NAME% python=3.11 -y
if errorlevel 1 (
    echo [FEHLER] Konnte Conda-Umgebung "%ENV_NAME%" nicht erstellen.
    pause
    exit /b 1
)
set "PYEXE=%CONDA_BASE%\envs\%ENV_NAME%\python.exe"
set "JUST_CREATED=1"
goto :env_ready

REM --- 2b. Kein Conda gefunden -- Python auf PATH oder .venv verwenden ----
:no_conda
echo [INFO] Keine Conda-Installation gefunden -- verwende Python/venv als Alternative.
REM "where python" reicht nicht: ohne aktivierte Umgebung zeigt Windows oft
REM auf den Microsoft-Store-Alias, der existiert, aber nicht lauffaehig ist.
python --version >nul 2>nul
if not errorlevel 1 (
    set "PYEXE=python"
    goto :env_ready
)
if exist ".venv\Scripts\python.exe" (
    set "PYEXE=.venv\Scripts\python.exe"
    goto :env_ready
)
where py >nul 2>nul
if errorlevel 1 (
    echo [FEHLER] Kein Python gefunden ^(weder conda noch py/python^).
    echo Bitte Python 3.10+ installieren: https://www.python.org/downloads/
    echo   - beim Installieren "Add python.exe to PATH" ankreuzen
    echo oder eine passende Conda-Umgebung aktivieren, bevor run.bat
    echo gestartet wird.
    pause
    exit /b 1
)
echo [INFO] Erstelle lokale virtuelle Umgebung .venv ...
py -m venv .venv
if errorlevel 1 (
    echo [FEHLER] Konnte .venv nicht erstellen.
    pause
    exit /b 1
)
set "PYEXE=.venv\Scripts\python.exe"
set "JUST_CREATED=1"

REM --- 3. Abhaengigkeiten installieren (nur beim allerersten Start) -------
:env_ready
echo [OK] Verwende Python: %PYEXE%
"%PYEXE%" --version

if defined JUST_CREATED (
    echo.
    echo [INFO] Installiere Python-Pakete, das kann ein paar Minuten dauern ...
    "%PYEXE%" -m pip install --upgrade pip >nul
    "%PYEXE%" -m pip install -e .
    if errorlevel 1 (
        echo [FEHLER] pip install fehlgeschlagen, siehe Ausgabe oben.
        pause
        exit /b 1
    )
    call :install_onnxruntime
) else (
    echo [OK] Umgebung bereits eingerichtet, ueberspringe Paketinstallation.
    echo      ^(Menuepunkt "Abhaengigkeiten neu installieren" fuer ein Update.^)
)

REM --- 4. exiftool pruefen (nur fuer Schritt 4/write-xmp noetig) ----------
where exiftool >nul 2>nul
if not errorlevel 1 (
    echo [OK] exiftool im PATH gefunden.
    goto :exiftool_done
)
if exist "%PROJECT_DIR%exiftool.exe" (
    echo [OK] exiftool.exe im Projektordner gefunden.
    set "PATH=%PROJECT_DIR%;%PATH%"
    goto :exiftool_done
)
echo.
echo [WARNUNG] exiftool wurde nicht gefunden ^(nur fuer Schritt 4 noetig^).
echo Bitte manuell herunterladen von https://exiftool.org
echo   1. Die Windows-ZIP-Datei herunterladen und KOMPLETT entpacken
echo      ^(nicht nur die .exe herausziehen!^)
echo   2. Die Datei "exiftool(-k).exe" UND den Ordner "exiftool_files"
echo      zusammen in diesen Projektordner kopieren:
echo      %PROJECT_DIR%
echo   3. Nur die Datei in "exiftool.exe" umbenennen.
echo      "exiftool_files" bleibt unveraendert direkt daneben liegen.
echo   4. Testen mit: exiftool.exe -ver
:exiftool_done

REM --- 5. config.yaml anlegen, falls noch nicht vorhanden -----------------
if not exist "config.yaml" (
    echo.
    echo [INFO] Keine config.yaml gefunden -- erstelle aus config.example.yaml.
    copy /y config.example.yaml config.yaml >nul
    echo.
    echo WICHTIG: catalog_copy_path ^(Kopie deiner .lrcat-Datei^) und
    echo images.root ^(dein Fotoordner^) in config.yaml eintragen.
    notepad config.yaml
)

echo.
echo [OK] Setup abgeschlossen.
echo.
pause

REM ============================================================
REM  Hauptmenue
REM ============================================================
:menu
cls
echo ============================================================
echo  face_pipeline - Gesichtserkennung fuer Lightroom Classic
echo ============================================================
echo   1) Lightroom-Katalog-Schema anzeigen (Diagnose)
echo   2) Schritt 1: Gesichter aus Lightroom exportieren
echo   3) Export-Stichproben pruefen (Zuschnitte ansehen)
echo   4) Schritt 2: Klassifikator trainieren
echo   5) Schritt 3: Gesichtserkennung auf Fotobibliothek anwenden
echo   6) Status anzeigen (erkannte Gesichter je Person)
echo   7) Schritt 4: Ergebnisse als XMP nach Lightroom schreiben
echo   8) config.yaml bearbeiten
echo   9) Abhaengigkeiten neu installieren/aktualisieren
echo  10) Beenden
echo ============================================================
set "CHOICE="
set /p CHOICE="Auswahl (1-10, Enter = 2): "
if "%CHOICE%"=="" set "CHOICE=2"

if "%CHOICE%"=="1" goto :do_inspect
if "%CHOICE%"=="2" goto :do_export
if "%CHOICE%"=="3" goto :do_verify
if "%CHOICE%"=="4" goto :do_train
if "%CHOICE%"=="5" goto :do_infer
if "%CHOICE%"=="6" goto :do_status
if "%CHOICE%"=="7" goto :do_xmp
if "%CHOICE%"=="8" goto :do_editconfig
if "%CHOICE%"=="9" goto :do_reinstall
if "%CHOICE%"=="10" goto :end_ok
echo Ungueltige Auswahl, bitte 1-10 eingeben.
echo.
goto :menu

:do_inspect
echo.
echo --- Lightroom-Katalog-Schema ---
"%PYEXE%" -m face_pipeline.cli inspect-catalog
goto :done

:do_export
echo.
echo --- Schritt 1: Gesichter aus Lightroom exportieren ---
"%PYEXE%" -m face_pipeline.cli export-faces
goto :done

:do_verify
echo.
echo --- Export-Stichproben pruefen ---
"%PYEXE%" -m face_pipeline.cli verify-crops
echo.
echo Oeffne den Ordner data\verify_crops und pruefe, ob die Zuschnitte
echo jeweils die richtige Person zeigen, bevor du trainierst.
goto :done

:do_train
echo.
echo --- Schritt 2: Klassifikator trainieren ---
"%PYEXE%" -m face_pipeline.cli train-classifier
goto :done

:do_infer
echo.
echo --- Schritt 3: Gesichtserkennung auf Fotobibliothek anwenden ---
set "INFER_LIMIT="
set /p INFER_LIMIT="Nur Testlauf mit wie vielen Bildern? (Enter = komplette Bibliothek): "
if "%INFER_LIMIT%"=="" (
    "%PYEXE%" -m face_pipeline.cli run-inference
) else (
    "%PYEXE%" -m face_pipeline.cli run-inference --limit %INFER_LIMIT%
)
"%PYEXE%" -m face_pipeline.cli status
goto :done

:do_status
echo.
"%PYEXE%" -m face_pipeline.cli status
goto :done

:do_xmp
echo.
echo --- Schritt 4: Ergebnisse als XMP nach Lightroom schreiben ---
where exiftool >nul 2>nul
if errorlevel 1 (
    echo exiftool ist nicht verfuegbar -- siehe Hinweis beim Setup oben.
    goto :done
)
echo Zuerst ein Probelauf ^(dry-run^), es wird noch nichts geschrieben:
"%PYEXE%" -m face_pipeline.cli write-xmp --dry-run
echo.
set /p XMP_CONFIRM="Jetzt wirklich schreiben? (j/n): "
if /i "%XMP_CONFIRM%"=="j" (
    "%PYEXE%" -m face_pipeline.cli write-xmp
    echo.
    echo Fertig. In Lightroom: betroffene Fotos auswaehlen, dann
    echo   Metadaten ^> Metadaten aus Datei lesen
    echo damit die neuen Gesichtsbereiche/Namen erscheinen.
) else (
    echo Abgebrochen, es wurde nichts geschrieben.
)
goto :done

:do_editconfig
notepad config.yaml
goto :done

:do_reinstall
echo.
echo --- Abhaengigkeiten neu installieren/aktualisieren ---
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -e .
call :install_onnxruntime
goto :done

:done
echo.
pause
goto :menu

:end_ok
echo.
echo Auf Wiedersehen!
exit /b 0

REM ============================================================
REM  Hilfsroutinen
REM ============================================================

REM Installiert onnxruntime-gpu (falls eine NVIDIA-GPU erkannt wird)
REM oder sonst das CPU-only onnxruntime, jeweils in %PYEXE%.
:install_onnxruntime
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo NVIDIA-GPU erkannt ^(nvidia-smi^) -- installiere onnxruntime-gpu.
    echo Hinweis: onnxruntime-gpu unter Windows braucht eine passende
    echo CUDA/cuDNN-Installation. Falls die GPU nicht genutzt werden kann,
    echo in config.yaml recognition.ctx_id auf -1 setzen und stattdessen
    echo   "%PYEXE%" -m pip install onnxruntime
    echo ausfuehren ^(statt onnxruntime-gpu^).
    "%PYEXE%" -m pip install onnxruntime-gpu
) else (
    echo Keine NVIDIA-GPU erkannt -- installiere onnxruntime ^(CPU^).
    "%PYEXE%" -m pip install onnxruntime
)
exit /b 0
