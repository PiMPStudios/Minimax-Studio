@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

rem MiniMax Studio runs on exactly one Python: .python-version is the source of
rem truth (pyproject's requires-python agrees, CI runs only that version). Find
rem it, build .venv on it, and move aside a .venv built on anything else.
set /p want=<.python-version
for /f "usebackq tokens=1,2 delims=." %%a in (".python-version") do set "maj=%%a" & set "min=%%b"

rem 1. Which interpreter is Python %want%? MINIMAX_STUDIO_PYTHON overrides.
if defined MINIMAX_STUDIO_PYTHON (
  set "PYPREF=%MINIMAX_STUDIO_PYTHON%"
  goto :venv
)
py -%want% -V >nul 2>nul
if not errorlevel 1 (
  set "PYPREF=py -%want%"
  goto :venv
)
python%want% -V >nul 2>nul
if not errorlevel 1 (
  set "PYPREF=python%want%"
  goto :venv
)
echo MiniMax Studio needs Python %want% - not found.
echo Install it from python.org and re-run, or set MINIMAX_STUDIO_PYTHON
echo to the full path of a %want% interpreter.
exit /b 1

:venv
rem 2. A .venv on the wrong Python looks ready and silently cannot install the
rem    pinned [train] extra, so it is moved aside, never deleted.
if not exist ".venv\Scripts\python.exe" goto :create
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(%maj%,%min%) else 1)" >nul 2>nul
if not errorlevel 1 goto :run
echo .venv was built with a different Python - moving it to .venv.stale and
echo rebuilding on Python %want%. Delete .venv.stale when you are done with it.
if exist .venv.stale rmdir /s /q .venv.stale
ren .venv .venv.stale

:create
%PYPREF% -m venv .venv
if errorlevel 1 exit /b 1

:run
call .venv\Scripts\activate.bat
pip install -e ".[dev]" >nul 2>nul || pip install -e .
python -m minimax_studio %*
