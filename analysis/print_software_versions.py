"""Print the software/hardware versions to cite in the paper. Run on Kaggle (GPU session if possible:
the GPU lines are blank on a CPU session). Usage: !python print_software_versions.py"""
import platform, subprocess, sys
print("python        :", sys.version.split()[0])
print("platform      :", platform.platform())
for name in ("tensorflow", "keras", "numpy", "scipy", "matplotlib"):
    try:
        m = __import__(name); print(f"{name:<14}:", m.__version__)
    except Exception as e:
        print(f"{name:<14}: not importable ({type(e).__name__})")
try:
    import tensorflow as tf
    bi = tf.sysconfig.get_build_info()
    print("TF built with CUDA :", bi.get("cuda_version"), "| cuDNN:", bi.get("cudnn_version"))
    print("GPUs visible       :", [d.name for d in tf.config.list_physical_devices("GPU")])
    print("default float dtype:", tf.keras.backend.floatx())
except Exception as e:
    print("tensorflow details unavailable:", e)
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20).stdout.strip() or "nvidia-smi: no GPU output")
except Exception as e:
    print("nvidia-smi unavailable:", e)
