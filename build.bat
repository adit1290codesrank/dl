@echo off
setlocal enabledelayedexpansion

:: Target source file (defaults to test.cpp if no argument is provided)
set MAIN_SRC=test.cpp
if "%~1" NEQ "" set MAIN_SRC=%~1

:: If the file doesn't exist, error out early
if not exist "%MAIN_SRC%" (
    echo Error: Target source file "%MAIN_SRC%" not found.
    exit /b 1
)

:: Extract base name for the executable
for %%I in ("%MAIN_SRC%") do set TARGET=%%~nI

echo ==========================================
echo Building for Windows ^(RTX 3050 - sm_86^)
echo Building %MAIN_SRC% to %TARGET%.exe...
echo ==========================================

set BUILD_DIR=build
set OBJ_DIR=build\obj
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"

:: Check for cl.exe
where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] cl.exe not found in PATH.
    echo Please run this script from the "x64 Native Tools Command Prompt for VS"
    echo or run "vcvars64.bat" first to set up the MSVC environment.
    exit /b 1
)

:: Compiler Flags (RTX 3050 uses Ampere architecture sm_86)
:: We use nvcc to compile both C++ and CUDA files on Windows to ensure MSVC host compiler alignment.
set NVCCFLAGS=-O3 -std=c++17 -Iinclude -arch=sm_86
set LDFLAGS=-lcublas -lcudart

set OBJS=

:: Compile all C++ files recursively in src folder
for /R src %%f in (*.cpp) do (
    echo [CXX] %%f
    nvcc %NVCCFLAGS% -c "%%f" -o "%OBJ_DIR%\%%~nf__cpp.obj"
    set OBJS=!OBJS! "%OBJ_DIR%\%%~nf__cpp.obj"
)

:: Compile all CUDA files recursively in src folder
for /R src %%f in (*.cu) do (
    echo [NVCC] %%f
    nvcc %NVCCFLAGS% -c "%%f" -o "%OBJ_DIR%\%%~nf__cu.obj"
    set OBJS=!OBJS! "%OBJ_DIR%\%%~nf__cu.obj"
)

:: Compile the entry point
echo [CXX] %MAIN_SRC%
nvcc %NVCCFLAGS% -c "%MAIN_SRC%" -o "%OBJ_DIR%\%TARGET%__main.obj"
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
