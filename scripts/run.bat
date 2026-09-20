@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

rem MiniMax Studio runs on exactly one Python: .python-version is the source of
rem truth (pyproject's requires-python agrees, CI runs only that version). Since
rem 0.2.67 this launcher *provides* that interpreter instead of demanding it: uv
rem reads the pin, reuses a system Python 3.12 when one exists, and otherwise
rem downloads a standalone build. Nobody installs Python 3.12 to run Studio any
rem more - the pin itself did not loosen, because simpletuner==4.8.0 ships
rem nothing outside >=3.12,<3.14 (that is the 0.2.28 incident).
rem
rem MINIMAX_STUDIO_PYTHON=\path\to\python.exe still bypasses uv entirely.
rem Unlike run.sh this never prompts: a double-clicked .bat has no console to
rem answer, so it prints the fix and exits. Pass --install-uv to mean it.

set /p want=<.python-version
for /f "usebackq tokens=1,2 delims=." %%a in (".python-version") do set "maj=%%a" & set "min=%%b"
set "uv_install=powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex""
set "venv_python=.venv\Scripts\python.exe"

set "print_plan=0"
set "install_uv=0"
set "app_args="
:argloop
if "%~1"=="" goto argdone
if "%~1"=="--print-runtime" (set "print_plan=1" & shift & goto argloop)
if "%~1"=="--install-uv" (set "install_uv=1" & shift & goto argloop)
set "app_args=!app_args! %~1"
shift
goto argloop
:argdone

rem 1. Where is uv? MINIMAX_STUDIO_UV_BIN wins, then PATH, then where the
rem    official installer drops it (before the shell path refreshes).
set "uv="
if defined MINIMAX_STUDIO_UV_BIN set "uv=%MINIMAX_STUDIO_UV_BIN%"
if not defined uv (
  for /f "usebackq delims=" %%u in (`where uv 2^>nul`) do (
    if not defined uv set "uv=%%u"
  )
)
if not defined uv if exist "%USERPROFILE%\.local\bin\uv.exe" set "uv=%USERPROFILE%\.local\bin\uv.exe"
if not defined uv if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "uv=%USERPROFILE%\.cargo\bin\uv.exe"
if not defined uv set "uv_display=not found (set MINIMAX_STUDIO_UV_BIN or install uv)"
if defined uv set "uv_display=!uv!"
set "override=none"
if defined MINIMAX_STUDIO_PYTHON set "override=%MINIMAX_STUDIO_PYTHON%"
set "venvver=none"
if exist "%venv_python%" (
  for /f "usebackq delims=" %%v in (`""%venv_python%" -c "import sys; print(sys.version.split()[0])" 2^>nul`) do set "venvver=%%v"
)

rem --print-runtime answers "what will Studio actually use?" without building
rem anything, so a support thread can be settled from one paste.
if "%print_plan%"=="1" (
  echo wants python   %want%  (.python-version^)
  echo uv             !uv_display!
  echo override       !override!
  echo .venv python   !venvver!
  exit /b 0
)

rem 2. The documented opt-out: an interpreter you chose yourself, uv never required.
if defined MINIMAX_STUDIO_PYTHON (
  "%MINIMAX_STUDIO_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(%maj%,%min%) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo MINIMAX_STUDIO_PYTHON=%MINIMAX_STUDIO_PYTHON% is not Python %want%.
    exit /b 1
  )
  call :retire
  if not exist "%venv_python%" "%MINIMAX_STUDIO_PYTHON%" -m venv .venv
  if errorlevel 1 exit /b 1
  call .venv\Scripts\activate.bat
  python -m pip install -e ".[dev]" >nul 2>nul || python -m pip install -e .
  python -m minimax_studio !app_args!
  exit /b
)

rem 3. The default path: uv resolves the pinned interpreter.
if not defined uv (
  if "%install_uv%"=="1" (
    echo Installing uv: %uv_install%
    call %uv_install%
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "uv=%USERPROFILE%\.local\bin\uv.exe"
  )
)
if not defined uv (
  echo MiniMax Studio needs uv - or set MINIMAX_STUDIO_PYTHON to a Python %want%.
  echo Install uv:
  echo     %uv_install%
  echo Then re-run scripts\run.bat.
  exit /b 1
)

call :retire
rem --seed so .venv\Scripts\pip.exe exists: AGENTS.md tells people to run
rem `pip install -e ".[train]"` inside the venv, and a uv venv has no pip without it.
if not exist "%venv_python%" "%uv%" venv --seed --python %want% .venv
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
"%uv%" pip install -e ".[dev]" >nul 2>nul || "%uv%" pip install -e .
python -m minimax_studio !app_args!
exit /b

rem A .venv built on another Python looks ready and silently cannot install the
rem pinned [train] extra, so it is moved aside, never deleted.
:retire
if not exist "%venv_python%" exit /b 0
"%venv_python%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(%maj%,%min%) else 1)" >nul 2>nul
if not errorlevel 1 exit /b 0
echo .venv was built with a different Python - moving it to .venv.stale and
echo rebuilding on Python %want%. Delete .venv.stale when you are done with it.
if exist .venv.stale rmdir /s /q .venv.stale
ren .venv .venv.stale
exit /b 0
