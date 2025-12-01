@echo off
REM Build script for Cython extensions on Windows
echo Building Cython extensions...
python setup.py build_ext --inplace
if %ERRORLEVEL% EQU 0 (
    echo Cython build successful!
    echo The optimized emotion_utils module is now available.
) else (
    echo Cython build failed. Falling back to Python implementation.
)
pause

