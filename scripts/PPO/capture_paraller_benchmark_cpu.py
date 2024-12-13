import time
import cv2
import numpy as np
import dxcam
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Grid setup for Minecraft clients
    GRID_COLS = 3
    GRID_ROWS = 4
    WINDOW_WIDTH = 360
    WINDOW_HEIGHT = 360
    
    TOTAL_WIDTH = GRID_COLS * WINDOW_WIDTH
    TOTAL_HEIGHT = GRID_ROWS * WINDOW_HEIGHT

    # Capture region setup
    CAPTURE_LEFT = 100
    CAPTURE_TOP = 100
    capture_region = (CAPTURE_LEFT, CAPTURE_TOP, CAPTURE_LEFT + TOTAL_WIDTH, CAPTURE_TOP + TOTAL_HEIGHT)
    
    # Initialize dxcam
    camera = dxcam.create(output_color="BGR")
    
    # Create sub-windows list
    subwindows = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            left = c * WINDOW_WIDTH
            top = r * WINDOW_HEIGHT
            subwindows.append((left, top, WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Performance tracking
    target_fps = 30
    frame_interval = 1.0 / target_fps
    last_time = time.time()
    frame_count = 0
    capture_times = []
    processing_times = []
    run_duration = 10.0

    logging.info("Starting capture test...")
    start_time = time.time()

    while True:
        now = time.time()
        if now - start_time > run_duration:
            break

        # Capture frame
        t0 = time.time()
        frame = camera.grab(region=capture_region)
        if frame is None:
            logging.warning("Failed to capture frame.")
            continue
        t1 = time.time()

        # Process each subwindow
        processed_subwindows = {}
        for i, (sw_left, sw_top, sw_w, sw_h) in enumerate(subwindows):
            # Extract and resize subwindow
            sub_img = frame[sw_top:sw_top+sw_h, sw_left:sw_left+sw_w, :]
            resized = cv2.resize(sub_img, (64, 64), interpolation=cv2.INTER_NEAREST)
            
            # Normalize and convert to CHW format
            resized_float = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
            processed_subwindows[i] = resized_float
        
        t2 = time.time()

        # Track timing
        capture_times.append(t1 - t0)
        processing_times.append(t2 - t1)
        frame_count += 1

        # Maintain target FPS
        elapsed = time.time() - now
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
    
    # Report results
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