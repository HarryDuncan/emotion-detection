import tensorflow as tf
import sys

def verify_gpu():
    print("--- ID System Health Check ---")
    
    # 1. Check if TensorFlow sees the Blackwell architecture
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("ERROR: No GPU detected by TensorFlow.")
        return False
    
    # 2. Check for CUDA support
    if not tf.test.is_built_with_cuda():
        print("ERROR: TensorFlow was not built with CUDA support.")
        return False

    # 3. Simple Compute Test (Triggers JIT loading)
    try:
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.1, 2.1], [3.1, 4.1]])
            tf.matmul(a, b)
        print("SUCCESS: CUDA Compute test passed on RTX 5080.")
        return True
    except Exception as e:
        print(f"ERROR: Compute test failed: {e}")
        return False

if __name__ == "__main__":
    if not verify_gpu():
        sys.exit(1) # Exit with error code for Docker to catch