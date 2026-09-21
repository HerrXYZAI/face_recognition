@echo off
REM ============================================================
REM  run-docker.bat - face_pipeline in Docker starten (Windows)
REM
REM  Docker-Pendant zu run.bat: statt einer lokalen Conda-Umgebung
REM  wird das Docker-Image aus docker\docker-compose.yml gebaut und
REM  jeder Schritt in einem Container ausgefuehrt. Dasselbe Menue
REM  fuer die fuenf Schritte wie in run.bat:
REM    1) Gesichter aus Lightroom exportieren
REM    2) Klassifikator trainieren
REM    3) Gesichtserkennung auf die Fotobibliothek anwenden
REM    4) Erkannte Gesichter im Browser pruefen (bevor irgendetwas
REM       geschrieben wird)
REM    5) Ergebnisse als XMP nach Lightroom zurueckschreiben
REM
REM  Voraussetzung: Docker Desktop (https://docs.docker.com/get-docker/),
REM  fuer GPU-Beschleunigung zusaetzlich der NVIDIA Container Toolkit
REM  auf dem Host. Siehe auch README.md, Abschnitt "Setup (Docker)".
REM
REM  Kann direkt per Doppelklick gestartet werden.
REM ============================================================

setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"
set "COMPOSE_FILE=docker\docker-compose.yml"

echo.
echo ============================================================
echo  face_pipeline - Docker-Setup
echo ============================================================
echo.

REM --- 1. Docker pruefen ---------------------------------------------------
where docker >nul 2>nul
if errorlevel 1 (
    echo [FEHLER] Docker wurde nicht gefunden.
    echo Bitte Docker Desktop installieren: https://docs.docker.com/get-docker/
    pause
    exit /b 1
)
docker compose version >nul 2>nul
if errorlevel 1 (
    echo [FEHLER] "docker compose" ist nicht verfuegbar. Bitte Docker Desktop aktualisieren.
    pause
    exit /b 1
)
echo [OK] Docker gefunden.
docker info >nul 2>nul
if errorlevel 1 (
    echo [FEHLER] Docker-Daemon laeuft nicht. Bitte Docker Desktop starten.
    pause
    exit /b 1
)

REM --- 2. GPU erkennen (NVIDIA), sonst CPU-Profil verwenden ----------------
set "PROFILE=cpu"
set "SERVICE=pipeline-cpu"
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo [OK] NVIDIA-GPU erkannt ^(nvidia-smi^) -- verwende GPU-Profil.
    echo      Voraussetzung: NVIDIA Container Toolkit ist auf diesem Host installiert.
    set "PROFILE=gpu"
    set "SERVICE=pipeline-gpu"
) else (
    echo [INFO] Keine NVIDIA-GPU erkannt -- verwende CPU-Profil.
)

REM --- 3. docker\.env anlegen, falls noch nicht vorhanden ------------------
if not exist "docker\.env" (
    echo.
    echo [INFO] Keine docker\.env gefunden -- erstelle aus docker\.env.example.
    copy /y docker\.env.example docker\.env >nul
    echo.
    echo WICHTIG: IMAGES_ROOT_HOST und CATALOG_COPY_DIR_HOST in docker\.env
    echo auf absolute Windows-Pfade setzen ^(z.B. D:/photos^).
    notepad docker\.env
)

REM --- 4. config.yaml anlegen, falls noch nicht vorhanden ------------------
if not exist "config.yaml" (
    echo.
    echo [INFO] Keine config.yaml gefunden -- erstelle aus config.example.yaml.
    copy /y config.example.yaml config.yaml >nul
    echo.
    echo WICHTIG: Im Docker-Betrieb muessen catalog_copy_path und images.root
    echo auf die Container-Pfade zeigen, nicht auf die echten Windows-Pfade:
    echo   lightroom.catalog_copy_path: /data/lightroom/catalog.lrcat
    echo   images.root: /data/images
    notepad config.yaml
)

REM --- 5. Image bauen (nutzt Layer-Cache, dauert nur beim ersten Mal) ------
echo.
echo [INFO] Baue Docker-Image fuer Profil "%PROFILE%" ...
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% build
if errorlevel 1 (
    echo [FEHLER] Docker-Build fehlgeschlagen, siehe Ausgabe oben.
    pause
    exit /b 1
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
echo  face_pipeline - Gesichtserkennung fuer Lightroom Classic ^(Docker, %PROFILE%^)
echo ============================================================
echo   1) Lightroom-Katalog-Schema anzeigen (Diagnose)
echo   2) Schritt 1: Gesichter aus Lightroom exportieren
echo   3) Export-Uebersicht anzeigen (exportierte Gesichter je Person)
echo   4) Export-Stichproben pruefen (Zuschnitte ansehen)
echo   5) Schritt 2: Klassifikator trainieren
echo   6) Schritt 3: Gesichtserkennung auf Fotobibliothek anwenden
echo   7) Status anzeigen (erkannte Gesichter je Person)
echo   8) Schritt 4: Erkannte Gesichter im Browser pruefen
echo   9) Schritt 5: Ergebnisse als XMP nach Lightroom schreiben
echo  10) config.yaml bearbeiten
echo  11) docker\.env bearbeiten (Host-Pfade)
echo  12) Docker-Image neu bauen (nach Code-/Dependency-Aenderungen)
echo  13) Beenden
echo ============================================================
set "CHOICE="
set /p CHOICE="Auswahl (1-13, Enter = 2): "
if "%CHOICE%"=="" set "CHOICE=2"

if "%CHOICE%"=="1" goto :do_inspect
if "%CHOICE%"=="2" goto :do_export
if "%CHOICE%"=="3" goto :do_export_status
if "%CHOICE%"=="4" goto :do_verify
if "%CHOICE%"=="5" goto :do_train
if "%CHOICE%"=="6" goto :do_infer
if "%CHOICE%"=="7" goto :do_status
if "%CHOICE%"=="8" goto :do_review
if "%CHOICE%"=="9" goto :do_xmp
if "%CHOICE%"=="10" goto :do_editconfig
if "%CHOICE%"=="11" goto :do_editenv
if "%CHOICE%"=="12" goto :do_rebuild
if "%CHOICE%"=="13" goto :end_ok
echo Ungueltige Auswahl, bitte 1-13 eingeben.
echo.
goto :menu

:do_inspect
echo.
echo --- Lightroom-Katalog-Schema ---
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% inspect-catalog
goto :done

:do_export
echo.
echo --- Schritt 1: Gesichter aus Lightroom exportieren ---
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% export-faces
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% export-status
goto :done

:do_export_status
echo.
echo --- Export-Uebersicht (exportierte Gesichter je Person) ---
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% export-status
goto :done

:do_verify
echo.
echo --- Export-Stichproben pruefen ---
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% verify-crops
echo.
echo Oeffne den Ordner data\verify_crops und pruefe, ob die Zuschnitte
echo jeweils die richtige Person zeigen, bevor du trainierst.
goto :done

:do_train
echo.
echo --- Schritt 2: Klassifikator trainieren ---
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% train-classifier
goto :done

:do_infer
echo.
echo --- Schritt 3: Gesichtserkennung auf Fotobibliothek anwenden ---
set "INFER_LIMIT="
set /p INFER_LIMIT="Nur Testlauf mit wie vielen Bildern? (Enter = komplette Bibliothek): "
if "%INFER_LIMIT%"=="" (
    docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% run-inference
) else (
    docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% run-inference --limit %INFER_LIMIT%
)
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% status
goto :done

:do_status
echo.
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% status
goto :done

:do_review
echo.
echo --- Schritt 4: Erkannte Gesichter im Browser pruefen ---
echo Oeffnet gleich http://localhost:7860 -- Browser manuell oeffnen und die
echo vorgeschlagenen Namen bestaetigen/korrigieren/ablehnen.
echo Zum Beenden hier im Fenster Strg+C druecken.
echo.
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm --service-ports %SERVICE% review-faces --host 0.0.0.0
goto :done

:do_xmp
echo.
echo --- Schritt 5: Ergebnisse als XMP nach Lightroom schreiben ---
echo Zuerst ein Probelauf ^(dry-run^), es wird noch nichts geschrieben:
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% write-xmp --dry-run
echo.
set /p XMP_CONFIRM="Jetzt wirklich schreiben? (j/n): "
if /i "%XMP_CONFIRM%"=="j" (
    docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% run --rm %SERVICE% write-xmp
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

:do_editenv
notepad docker\.env
goto :done

:do_rebuild
echo.
echo --- Docker-Image neu bauen ---
docker compose -f "%COMPOSE_FILE%" --profile %PROFILE% build --no-cache
goto :done

:done
echo.
pause
goto :menu

:end_ok
echo.
echo Auf Wiedersehen!
exit /b 0
