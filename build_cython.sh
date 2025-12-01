#!/bin/bash
# Build script for Cython extensions on Linux/Mac
echo "Building Cython extensions..."
python setup.py build_ext --inplace
if [ $? -eq 0 ]; then
    echo "Cython build successful!"
    echo "The optimized emotion_utils module is now available."
else
    echo "Cython build failed. Falling back to Python implementation."
fi

