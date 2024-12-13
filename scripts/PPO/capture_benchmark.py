# capture_benchmark.py
import numpy as np
import cv2
import dxcam
import mss
import time
import pygetwindow as gw
import fnmatch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from threading import Thread, Lock
from abc import ABC, abstractmethod

class CaptureMethod(ABC):
    def __init__(self, window_bounds, name):
        self.bounds = window_bounds
        self.name = name
        self.frame_count = 0
        self.dropped_frames = 0
        self.total_latency = 0
        self.results = []
        
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def capture_frame(self):
        pass
    
    @abstractmethod
    def cleanup(self):
        pass

class DXCamDevicePool:
    _instances = {}
    _lock = Lock()
    _next_output = 0
    MAX_OUTPUTS = 4  # Adjust based on your DXCam capabilities

    @classmethod
    def get_device(cls):
        with cls._lock:
            if cls._next_output >= cls.MAX_OUTPUTS:
                return None
                
            output_idx = cls._next_output
            cls._next_output += 1
            
            key = (0, output_idx)
            if key not in cls._instances:
                try:
                    cls._instances[key] = dxcam.create(
                        device_idx=0,
                        output_idx=output_idx,
                        output_color="BGR"
                    )
                except Exception as e:
                    print(f"Failed to create DXCam instance {output_idx}: {e}")
                    return None
            return cls._instances[key]

    @classmethod
    def cleanup(cls):
        with cls._lock:
            for camera in cls._instances.values():
                try:
                    del camera
                except:
                    pass
            cls._instances.clear()
            cls._next_output = 0

class DXCamCapture(CaptureMethod):
    def initialize(self):
        self.camera = DXCamDevicePool.get_device()
        if self.camera is None:
            raise RuntimeError("Failed to initialize DXCam")
            
        left = max(self.bounds['left'], 0)
        top = max(self.bounds['top'], 0)
        right = min(left + self.bounds['width'], self.camera.width)
        bottom = min(top + self.bounds['height'], self.camera.height)
        self.region = (left, top, right, bottom)
        
    def capture_frame(self):
        return self.camera.grab(region=self.region)
        
    def cleanup(self):
        # Don't cleanup shared camera instance
        pass

class MSSCapture(CaptureMethod):
    def initialize(self):
        self.sct = mss.mss()
        
    def capture_frame(self):
        return np.array(self.sct.grab(self.bounds))[:,:,:3]
        
    def cleanup(self):
        self.sct.close()

class CaptureThread(Thread):
    def __init__(self, capture_method, target_fps, duration):
        super().__init__()
        self.capture = capture_method
        self.target_fps = target_fps
        self.duration = duration
        self.frame_interval = 1.0 / target_fps
        self.running = True
        
    def run(self):
        try:
            self.capture.initialize()
        except RuntimeError as e:
            print(f"{self.capture.name} initialization failed: {e}")
            return
        
        start_time = time.time()
        next_frame_time = start_time
        
        while time.time() - start_time < self.duration:
            current_time = time.time()
            
            if current_time >= next_frame_time:
                frame_start = time.time()
                frame = self.capture.capture_frame()
                capture_latency = time.time() - frame_start
                
                if frame is not None:
                    self.capture.frame_count += 1
                    self.capture.total_latency += capture_latency
                else:
                    self.capture.dropped_frames += 1
                
                self.capture.results.append({
                    'timestamp': current_time - start_time,
                    'latency': capture_latency * 1000,  # Convert to ms
                    'dropped': frame is None
                })
                
                next_frame_time += self.frame_interval
            
            # Smart sleep to reduce CPU usage
            sleep_time = next_frame_time - time.time()
            if (sleep_time > 0):
                time.sleep(sleep_time * 0.8)  # Sleep slightly less to account for overhead
                
        self.capture.cleanup()

def find_minecraft_windows():
    print("\n=== Starting Minecraft Window Detection ===")
    
    patterns = ["Minecraft*", "*1.21.3*", "*Singleplayer*"]
    windows = []
    seen_handles = set()

    # Get screen resolution using first DXCam instance
    try:
        camera = dxcam.create(device_idx=0, output_idx=0)
        screen_width = camera.width
        screen_height = camera.height
        print(f"Detected screen resolution: {screen_width}x{screen_height}")
        del camera  # Clean up test instance
    except Exception as e:
        print(f"Failed to get screen resolution from DXCam: {e}")
        print("Falling back to default 1920x1080")
        screen_width = 3840
        screen_height = 2160

    # Get all windows
    for title in gw.getAllTitles():
        for pattern in patterns:
            if (fnmatch.fnmatch(title, pattern)):
                for window in gw.getWindowsWithTitle(title):
                    if window._hWnd not in seen_handles:
                        # Calculate center of window
                        center_x = window.left + window.width // 2
                        center_y = window.top + window.height // 2
                        crop_size = 240
                        half = crop_size // 2
                        
                        # Ensure coordinates are within screen bounds
                        left = max(min(center_x - half, screen_width - crop_size), 0)
                        top = max(min(center_y - half, screen_height - crop_size), 0)
                        
                        window_info = {
                            "left": left,
                            "top": top,
                            "width": crop_size,
                            "height": crop_size
                        }
                        
                        print(f"Found window at ({left}, {top})")
                        windows.append(window_info)
                        seen_handles.add(window._hWnd)

    print(f"\nFound {len(windows)} Minecraft windows")
    return windows

def run_benchmark():
    try:
        windows = find_minecraft_windows()
        if not windows:
            print("No Minecraft windows found!")
            return []
            
        fps_tests = [30, 60]
        duration = 5
        results = []
        
        for fps in fps_tests:
            print(f"\nTesting at {fps} FPS...")
            threads = []
            
            # Mix DXCam and MSS based on available outputs
            for i, window in enumerate(windows):
                if i < DXCamDevicePool.MAX_OUTPUTS:
                    capture = DXCamCapture(window, f"DXCam-{i}")
                else:
                    capture = MSSCapture(window, f"MSS-{i}")
                    
                thread = CaptureThread(capture, fps, duration)
                threads.append((capture, thread))
                thread.start()
                time.sleep(0.2)  # Increased delay between starts
                
            # Wait for all threads to complete
            for capture, thread in threads:
                thread.join()
                
                if capture.frame_count > 0:
                    avg_latency = (capture.total_latency / capture.frame_count) * 1000
                    actual_fps = capture.frame_count / duration
                    drop_rate = (capture.dropped_frames / (capture.frame_count + capture.dropped_frames)) * 100
                    
                    results.append({
                        'method': capture.name,
                        'target_fps': fps,
                        'actual_fps': actual_fps,
                        'avg_latency_ms': avg_latency,
                        'drop_rate': drop_rate
                    })
        
        return results
    finally:
        DXCamDevicePool.cleanup()

def plot_results(results):
    df = pd.DataFrame(results)
    
    # Plot average latency
    plt.figure(figsize=(12, 6))
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.plot(method_data['target_fps'], method_data['avg_latency_ms'], 
                marker='o', label=method)
    
    plt.xlabel('Target FPS')
    plt.ylabel('Average Latency (ms)')
    plt.title('Capture Latency vs FPS by Method')
    plt.legend()
    plt.grid(True)
    plt.savefig('capture_latency.png')
    
    # Save detailed results
    df.to_csv('capture_benchmark_results.csv', index=False)

if __name__ == "__main__":
    results = run_benchmark()
    plot_results(results)
    print("Benchmark completed. Results saved to 'capture_latency.png' and 'capture_benchmark_results.csv'.")