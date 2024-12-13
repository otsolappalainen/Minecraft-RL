import time
import cv2
import numpy as np
import dxcam
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Test if cuda works
    _ = cv2.cuda_GpuMat()
    USE_CUDA = True
    logging.info("CUDA is available. Using GPU acceleration.")
except Exception as e:
    logging.warning(f"CUDA not available, falling back to CPU. Error: {e}")
    USE_CUDA = False

def main():
    # Suppose we have a grid of Minecraft clients:
    # Let's say 3x4 grid (12 total windows), each window is 120x120 pixels
    # so total grid = width=3*120=360, height=4*120=480
    # Adjust these values based on your actual arrangement.
    
    GRID_COLS = 3
    GRID_ROWS = 4
    WINDOW_WIDTH = 360
    WINDOW_HEIGHT = 360
    
    TOTAL_WIDTH = GRID_COLS * WINDOW_WIDTH
    TOTAL_HEIGHT = GRID_ROWS * WINDOW_HEIGHT

    # Hard-coded top-left corner for the entire captured region
    # In a real scenario, you would find where the Minecraft clients are arranged on the screen.
    CAPTURE_LEFT = 100
    CAPTURE_TOP = 100

    capture_region = (CAPTURE_LEFT, CAPTURE_TOP, CAPTURE_LEFT + TOTAL_WIDTH, CAPTURE_TOP + TOTAL_HEIGHT)
    
    # Initialize dxcam
    camera = dxcam.create(output_color="BGR")

    # Prepare GPU mats if using CUDA
    if USE_CUDA:
        gpu_frame = cv2.cuda_GpuMat()
    
    # Create a list of sub-windows (env windows)
    subwindows = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            left = c * WINDOW_WIDTH
            top = r * WINDOW_HEIGHT
            subwindows.append((left, top, WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Let's aim for a certain capture framerate
    target_fps = 30
    frame_interval = 1.0 / target_fps

    last_time = time.time()
    frame_count = 0
    capture_times = []
    processing_times = []
    run_duration = 10.0  # run for 10 seconds

    logging.info("Starting capture test...")
    start_time = time.time()

    while True:
        now = time.time()
        if now - start_time > run_duration:
            break

        # Capture start
        t0 = time.time()
        frame = camera.grab(region=capture_region)
        if frame is None:
            logging.warning("Failed to capture frame.")
            continue
        t1 = time.time()

        # Convert captured frame to GPU if available
        if USE_CUDA:
            gpu_frame.upload(frame)

        # Now let's simulate processing each subwindow
        # For example, maybe you want to downscale each to 120x120 (already 120x120, but let's do a GPU resize anyway)
        # In a real scenario, you might be resizing to a smaller size or just normalizing.
        processed_subwindows = {}
        for i, (sw_left, sw_top, sw_w, sw_h) in enumerate(subwindows):
            # Extract subwindow
            # frame is a numpy array, shape (H, W, 3)
            # subwindow region: [top:top+sw_h, left:left+sw_w]
            # If using GPU: we can slice on CPU first, then upload, or consider uploading full frame and using cudaWarpAffine (complex)
            # For simplicity: slice on CPU, then upload subwindow and resize.
            
            # slice subwindow from CPU frame:
            sub_img = frame[sw_top:sw_top+sw_h, sw_left:sw_left+sw_w, :]
            
            # Let's say we want to resize to 64x64 to reduce load
            new_w, new_h = 64, 64
            if USE_CUDA:
                g = cv2.cuda_GpuMat()
                g.upload(sub_img)
                g_resized = cv2.cuda.resize(g, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                resized = g_resized.download()
            else:
                resized = cv2.resize(sub_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Normalize to [0,1], float32 (simulate what env might do)
            resized_float = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW format
            processed_subwindows[i] = resized_float
        
        t2 = time.time()

        # Simulate passing processed_subwindows to each env
        # In a real scenario, we might store these in a shared memory or queue.
        # Here, we just discard them to simulate latency.
        # processed_subwindows dict done.

        capture_time = t1 - t0
        processing_time = t2 - t1
        capture_times.append(capture_time)
        processing_times.append(processing_time)

        frame_count += 1
        # Sleep to maintain target FPS (optional)
        elapsed = time.time() - now
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
    
    # After run_duration, stop and print stats
    avg_capture = np.mean(capture_times) if capture_times else 0
    avg_processing = np.mean(processing_times) if processing_times else 0
    total_time = time.time() - start_time
    actual_fps = frame_count / total_time if total_time > 0 else 0

    logging.info(f"Captured {frame_count} frames in {total_time:.2f}s")
    logging.info(f"Average capture time per frame: {avg_capture*1000:.2f} ms")
    logging.info(f"Average processing time per frame: {avg_processing*1000:.2f} ms")
    logging.info(f"Achieved FPS: {actual_fps:.2f}")

    logging.info("Test finished.")

if __name__ == "__main__":
    main()
