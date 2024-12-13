# Description: This script is used to test the availability of CUDA and GPU on the system. It also checks the availability of other libraries that are used in the project.

import numpy as np
import torch
import gymnasium as gym
import stable_baselines3
import cv2
import mss
import pygetwindow as gw
import websockets
import tensorboard
import os

def check_cuda():
    if torch.cuda.is_available():
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda
        }
    return {"available": False}

def main():
    # Core ML Libraries
    print("\n=== Core ML Libraries ===")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    cuda_info = check_cuda()
    print("\n=== CUDA Information ===")
    if cuda_info["available"]:
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {cuda_info['cuda_version']}")
        print(f"GPU Count: {cuda_info['device_count']}")
        print(f"GPU Name: {cuda_info['device_name']}")
    else:
        print("CUDA Available: No")

    # RL Libraries
    print("\n=== RL Libraries ===")
    print(f"Gymnasium: {gym.__version__}")
    print(f"Stable-Baselines3: {stable_baselines3.__version__}")

    # Image Processing
    print("\n=== Image Processing ===")
    print(f"OpenCV: {cv2.__version__}")
    print(f"MSS: {mss.__version__}")

    # System Integration
    print("\n=== System Libraries ===")
    print(f"PyGetWindow: {gw.__version__}")
    print(f"WebSockets: {websockets.__version__}")
    
    # Check Java (needed for Minecraft)
    print("\n=== System Dependencies ===")
    try:
        java_version = os.popen('java -version 2>&1').read()
        print("Java: " + java_version.split('\n')[0])
    except:
        print("Java: Not found")

if __name__ == "__main__":
    main()