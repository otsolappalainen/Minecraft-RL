import websocket
import json
import time
from pynput import mouse, keyboard
import threading
from collections import defaultdict
import mss
import cv2
import os
import numpy as np
import asyncio
import websockets

# Update constants
ACTIONS = [
    "move_forward", "move_left", "move_right", "jump_walk_forward",
    "jump", "look_left", "look_right", "look_up", "look_down",
    "attack", "reset 2", "tp 1", "tp 2", "tp 3", "tp 4", "attack 2", "monitor", "no_op"
]
SERVER_URI = "ws://localhost:8080"

# Add screenshot setup
SCREENSHOT_DIR = "test_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Global control flags
should_send = False
running = True
screenshot_counter = 0  # Add counter


def format_response(data):
    try:
        player_info = f"Player: x:{data.get('x', 0):.2f} y:{data.get('y', 0):.2f} z:{data.get('z', 0):.2f} " \
                     f"health:{data.get('health', 0):.1f} alive:{data.get('alive', False)}"
        mobs = data.get('mobs', [0, 0, 0])
        mob_info = f"Mobs: type:{mobs[0]:.0f} dist:{mobs[1]:.2f} status:{mobs[2]:.2f}"
        return f"\r{player_info}\n{mob_info}"
    except Exception as e:
        return f"Error formatting response: {e}"

async def get_window_info(port):
    uri = f"ws://localhost:{port}"
    try:
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"action": "monitor"}))
            response = await ws.recv()
            window_data = json.loads(response)
            
            center_x = window_data["x"] + window_data["width"] // 2
            center_y = window_data["y"] + window_data["height"] // 2
            half_width = 856 // 2  # Using same dimensions as original
            half_height = 482 // 2

            return {
                "left": center_x - half_width,
                "top": center_y - half_height,
                "width": window_data["width"],
                "height": window_data["height"]
            }
    except Exception as e:
        print(f"Failed to connect to {uri}: {e}")
        return None

def attempt_window_connection(retries=2):
    for i in range(retries):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(get_window_info(8080))
            loop.close()
            if result:
                return result
            print(f"Retry {i+1}/{retries}")
            time.sleep(2)
        except Exception as e:
            print(f"Error on try {i+1}: {e}")
    return None

def capture_screenshot(bounds):
    try:
        with mss.mss() as sct:
            # Capture screenshot
            screenshot = sct.grab(bounds)
            # Convert to numpy array and correct format for OpenCV
            img = np.array(screenshot)
            img = img[:, :, :3]  # Remove alpha channel
            
            # Convert to BGR format for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Calculate center coordinates
            center_x = img.shape[1] // 2
            center_y = img.shape[0] // 2
            
            # Draw white dot (4x4 circle) - using properly formatted image
            cv2.circle(img, (center_x, center_y), 2, (255, 255, 255), thickness=-1)
            
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB
    except Exception as e:
        print(f"Screenshot error: {e}")
        return None

def on_click(x, y, button, pressed):
    global should_send
    if button == mouse.Button.x1:  # Mouse button 4
        should_send = pressed
        if pressed:
            print("\nAction sending started")
        else:
            print("\nAction sending stopped")
    return running

def on_press(key):
    global running
    if key == keyboard.Key.esc:
        running = False
        return False
    return True

def send_action(ws, action, bounds=None):
    global screenshot_counter
    try:
        # Send action
        ws.send(json.dumps({"action": action}))
        response = ws.recv()
        data = json.loads(response)
        
        # Capture and save screenshot
        bounds = bounds or attempt_window_connection()
        screenshot = capture_screenshot(bounds)
        if screenshot is not None:
            # Get mob presence (1.0 = yes, 0.0 = no)
            mobs = data.get('mobs', [0, 0, 0])
            mob_present = "yes" if mobs[0] == 1.0 else "no"
            
            # Create filename
            filename = f"img_{screenshot_counter}_{mob_present}.png"
            filepath = os.path.join(SCREENSHOT_DIR, filename)
            cv2.imwrite(filepath, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
            screenshot_counter += 1
        
        print(format_response(data))
    except Exception as e:
        print(f"\rError: {e}")

def establish_connection():
    try:
        ws = websocket.create_connection(SERVER_URI)
        print(f"Connected to {SERVER_URI}")
        return ws
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def action_loop(ws, action, window_bounds):
    global running, should_send
    last_send_time = 0
    
    while running:
        current_time = time.time()
        if should_send and (current_time - last_send_time) >= 0.1:  # 100ms check
            send_action(ws, action, window_bounds)
            last_send_time = current_time
        time.sleep(0.01)  # Small sleep to prevent CPU spinning

def main():
    global running, should_send
    
    # Get window bounds at startup
    print("Detecting Minecraft window...")
    window_bounds = attempt_window_connection()
    if not window_bounds:
        print("Could not find Minecraft window. Exiting...")
        return

    # Update find_minecraft_window to use detected bounds

    # Start listeners
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener = mouse.Listener(on_click=on_click)
    
    keyboard_listener.start()
    mouse_listener.start()

    while running:
        print("\nAvailable actions:")
        for idx, action in enumerate(ACTIONS, 1):
            print(f"{idx}. {action}")
        
        action_input = input("\nEnter action (number/name) or 'exit': ").strip()
        
        if action_input.lower() == 'exit':
            running = False
            break

        if action_input.isdigit():
            idx = int(action_input) - 1
            if 0 <= idx < len(ACTIONS):
                action = ACTIONS[idx]
            else:
                print("Invalid number")
                continue
        elif action_input in ACTIONS:
            action = action_input
        else:
            print("Invalid action")
            continue

        print(f"\nSelected: {action}")
        print("Hold Mouse Button 4 to send actions")
        print("ESC to exit")

        ws = establish_connection()
        if not ws:
            continue

        try:
            action_loop(ws, action, window_bounds)
        except KeyboardInterrupt:
            running = False
        finally:
            ws.close()

    keyboard_listener.stop()
    print(f"stop")
    mouse_listener.stop()

if __name__ == "__main__":
    main()