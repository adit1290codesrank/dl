@echo off
setlocal enabledelayedexpansion

:: Target source file (defaults to examples/test.cpp if no argument is provided)
set MAIN_SRC=examples\test.cpp
if "%~1" NEQ "" set MAIN_SRC=%~1

:: GPU architecture (defaults to sm_86 for RTX 3050)
set ARCH=sm_86
if "%~2" NEQ "" set ARCH=%~2

:: If the file doesn't exist, error out early
if not exist "%MAIN_SRC%" (
    echo Error: Target source file "%MAIN_SRC%" not found.
    exit /b 1
)

:: Extract base name for the executable
for %%I in ("%MAIN_SRC%") do set TARGET=%%~nI

echo ==========================================
echo Building %MAIN_SRC% to %TARGET%.exe
echo GPU arch: %ARCH%
echo ==========================================

set BUILD_DIR=build
set OBJ_DIR=build\obj
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"
if not exist "weights" mkdir "weights"
if not exist "outputs" mkdir "outputs"

:: Check for cl.exe
where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] cl.exe not found in PATH.
    echo Please run this script from the "x64 Native Tools Command Prompt for VS"
    echo or run "vcvars64.bat" first to set up the MSVC environment.
    exit /b 1
)

:: Compiler Flags
set NVCCFLAGS=-O3 -m64 -std=c++17 -I. -Iinclude -arch=%ARCH% -cudart static
set LDFLAGS=-lcublas -lcudart

set OBJS=

:: Compile all C++ files recursively in src folder
for /R src %%f in (*.cpp) do (
    echo [CXX] %%f
    nvcc %NVCCFLAGS% -c "%%f" -o "%OBJ_DIR%\%%~nf__cpp.obj"
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Compilation failed for %%f
        exit /b 1
    )
    set OBJS=!OBJS! "%OBJ_DIR%\%%~nf__cpp.obj"
)

:: Compile all CUDA files recursively in src folder
for /R src %%f in (*.cu) do (
    echo [NVCC] %%f
    nvcc %NVCCFLAGS% -c "%%f" -o "%OBJ_DIR%\%%~nf__cu.obj"
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Compilation failed for %%f
        exit /b 1
    )
    set OBJS=!OBJS! "%OBJ_DIR%\%%~nf__cu.obj"
)

:: Compile the entry point
echo [CXX] %MAIN_SRC%
nvcc %NVCCFLAGS% -c "%MAIN_SRC%" -o "%OBJ_DIR%\%TARGET%__main.obj"
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Compilation failed for %MAIN_SRC%
    exit /b 1
)
set OBJS=!OBJS! "%OBJ_DIR%\%TARGET%__main.obj"

:: Link everything together
echo.
echo Linking %TARGET%.exe...
nvcc %NVCCFLAGS% !OBJS! -o "%TARGET%.exe" %LDFLAGS%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo Build Successful!
    echo Run using: %TARGET%.exe
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo Build Failed!
    echo ==========================================
)
