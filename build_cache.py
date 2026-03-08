"""
Warm up TensorFlow and DeepFace so emotion detection is ready.
If GPU gives CUDA_ERROR_INVALID_HANDLE, use CPU mode:
    EMOTION_CPU_ONLY=1 python build_cache.py
Then start the app with:
    EMOTION_CPU_ONLY=1 python appv2.py
"""
import os
import numpy as np

# CPU-only mode: use this when GPU fails with CUDA_ERROR_INVALID_HANDLE
USE_CPU = os.environ.get("EMOTION_CPU_ONLY", "").lower() in ("1", "true", "yes")
if USE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Running in CPU mode (EMOTION_CPU_ONLY=1). No GPU cache will be built.\n")

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
os.environ["TF_DISABLE_XLA"] = "1"
print("Configuring environment variables...")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_CACHE_DISABLE"] = "0"
os.environ["CUDA_CACHE_MAXSIZE"] = "2147483648"
os.environ["CUDA_CACHE_PATH"] = os.path.expanduser("~/.nv/ComputeCache")

import tensorflow as tf
from deepface import DeepFace

gpus = tf.config.list_physical_devices('GPU')
if gpus and not USE_CPU:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU memory growth enabled.")
else:
    print("Using CPU (no GPU or CPU-only mode).")

device = "/CPU:0" if USE_CPU else "/GPU:0"
print(f"\nWarming up TensorFlow and DeepFace on {device}...\n")

# Phase 1: Warm up TensorFlow
print("Phase 1: TensorFlow warmup...")
try:
    with tf.device(device):
        for _ in range(3):
            a = tf.random.normal((256, 256))
            b = tf.random.normal((256, 256))
            _ = tf.linalg.matmul(a, b).numpy()
    print("Phase 1 done.")
except Exception as e:
    print(f"Phase 1 warning: {e}")

# Phase 2: DeepFace emotion model
print("Phase 2: DeepFace emotion model...")
try:
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    DeepFace.analyze(img_path=dummy, actions=['emotion'], enforce_detection=False)
    print("Phase 2 done.")
except Exception as e:
    print(f"Phase 2 warning: {e}")

print("\n✅ SUCCESS! You can start the app.")
if USE_CPU:
    print("Start with: EMOTION_CPU_ONLY=1 python appv2.py")
else:
    print("Cache (if used) is in ~/.nv/ComputeCache")
    print("Start with: python appv2.py")