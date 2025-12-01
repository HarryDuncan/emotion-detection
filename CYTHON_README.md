# Cython Optimization for Emotion Detection

This project includes optional Cython optimizations to speed up emotion averaging calculations.

## What is Optimized?

The `calculate_emotion_averages()` function is optimized using Cython. This function:
- Processes emotion history (up to 150 entries)
- Calculates time-windowed averages
- Runs frequently (every 2-20 seconds for smoothing)

## Expected Performance Gain

- **10-30% faster** emotion averaging calculations
- Most noticeable when processing large emotion_history buffers
- Minimal impact on overall system (DeepFace inference is still the main bottleneck)

## Installation

### 1. Install Cython

```bash
pip install cython
```

Or it's already in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Build the Cython Extension

**Windows:**
```bash
build_cython.bat
```

**Linux/Mac:**
```bash
chmod +x build_cython.sh
./build_cython.sh
```

**Manual:**
```bash
python setup.py build_ext --inplace
```

### 3. Verify Installation

The code will automatically detect if Cython is available. If the build fails, it will gracefully fall back to the Python implementation.

## How It Works

- The code tries to import `emotion_utils` (the compiled Cython module)
- If available, uses the optimized version
- If not available, falls back to pure Python (no errors)

## Troubleshooting

**Build fails:**
- Make sure you have a C compiler installed (Visual Studio Build Tools on Windows, GCC on Linux/Mac)
- Check that Cython is installed: `pip install cython`
- The code will work fine without Cython - it's just slower

**Import errors:**
- The code handles ImportError gracefully
- Check that `emotion_utils.pyd` (Windows) or `emotion_utils.so` (Linux/Mac) exists in the project root

## When to Use

Cython is most beneficial when:
- Processing large emotion_history buffers (>100 entries)
- Running on slower CPUs
- Every millisecond counts for real-time applications

For most use cases, the Python version is fast enough since:
- The main bottleneck is DeepFace model inference (already optimized C++)
- Emotion averaging is a small part of the total processing time

