@echo off
rem Opens the viewer, ray traced, after compiling what changed.
rem
rem   gui                    Yennefer in the gold studio (studio-model)
rem   gui <model file>       that model in the gold studio
rem   gui <scene>            a scene by name: cornell-final, final-week, ...
rem   gui ... -Drt.x=y       any other -D switch goes on to the renderer
rem
rem From PowerShell: .\gui
rem GUI_DRY=1 prints the command instead of running it.
setlocal
cd /d "%~dp0"

set "YEN=C:\Users\szymondomeracki\Documents\projects\StlViewer\models-to-render\models\003_yennefer\single_color_19.stl"
set "WHAT=-Drt.scene=studio-model"
set "REST=%*"

set "FIRST=%~1"
if not defined FIRST goto yennefer
if "%FIRST:~0,1%"=="-" goto yennefer

rem The first word is a model file or a scene; the rest of the line goes on unchanged.
set "ALL=%*"
call set "REST=%%ALL:*%1=%%"
if exist "%FIRST%" (
    set WHAT=-Drt.scene=studio-model "-Drt.model=%~f1"
) else (
    set "WHAT=-Drt.scene=%FIRST%"
)
goto run

:yennefer
if exist "%YEN%" set WHAT=%WHAT% "-Drt.model=%YEN%"

:run
rem PowerShell hands an unquoted -Drt.samples=64 over as "-Drt .samples=64": join it again.
if defined REST set "REST=%REST:-Drt .=-Drt.%"
set CMD=mvn -q -o compile exec:java -Drt.gui -Drt.gui.raytrace %WHAT% %REST%
if defined GUI_DRY (
    echo %CMD%
    exit /b 0
)
call %CMD%
