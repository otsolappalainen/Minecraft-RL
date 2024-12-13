import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
import websockets
import threading
import json
import time
import cv2
import pygetwindow as gw
import fnmatch
import os
import logging
import uuid
from collections import deque
import math
from threading import Lock
import concurrent.futures
import dxcam  # New: GPU-assisted screen capture

# Training settings
MAX_EPISODE_STEPS = 512

# Reward settings
STEP_PENALTY = -0.1
MAX_BLOCK_REWARD = 10
ATTACK_BLOCK_REWARD_SCALE = 2
ATTACK_MISS_PENALTY = -0.3
JUMP_PENALTY = -1
FORWARD_MOVE_BONUS = 0.8
LOOK_ADJUST_BONUS = 0.05
SIDE_PENALTY = -0.2
BACKWARD_PENALTY = -0.5

# Timeout settings in seconds.
TIMEOUT_STEP = 5
TIMEOUT_STATE = 2
TIMEOUT_RESET = 10
TIMEOUT_STEP_LONG = 5
TIMEOUT_RESET_LONG = 5

# Screenshot settings
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 120
IMAGE_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)

# Surrounding blocks
SURROUNDING_BLOCKS_SHAPE = (13, 13, 4)


class MinecraftEnv(gym.Env):
    """
    Custom Environment that interfaces with Minecraft via WebSocket.
    Handles a single client per environment instance, with GPU-accelerated
    image capture and processing using dxcam and cv2.cuda.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, uri="ws://localhost:8080", task=None, window_bounds=None, save_example_step_data=False):
        super(MinecraftEnv, self).__init__()
        
        # Initialize logging
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Require window bounds
        if window_bounds is None:
            raise ValueError("window_bounds parameter is required")

        self.minecraft_bounds = window_bounds
        self.capture_lock = Lock()

        # Attempt to use GPU-accelerated resizing
        self.use_cuda = True
        try:
            _ = cv2.cuda_GpuMat()
        except Exception as e:
            logging.warning(f"Could not use cv2.cuda. Falling back to CPU processing. Error: {e}")
            self.use_cuda = False

        self.capture_failure_count = 0
        self.max_capture_failures = 5

        # Initialize dxcam camera
        self.camera = None
        self.init_camera()

        self.save_screenshots = True
        if self.save_screenshots:
            self.screenshot_dir = "env_screenshots"
            os.makedirs(self.screenshot_dir, exist_ok=True)

        # Single URI
        self.uri = uri

        # Define action and observation space
        self.ACTION_MAPPING = {
            0: "move_forward",
            1: "move_backward",
            2: "move_left",
            3: "move_right",
            4: "jump_walk_forward",
            5: "jump",
            6: "sneak",
            7: "look_left",
            8: "look_right",
            9: "look_up",
            10: "look_down",
            11: "attack",
            12: "use",
            13: "next_item",
            14: "previous_item",
            15: "no_op"
        }

        self.action_space = spaces.Discrete(len(self.ACTION_MAPPING))

        # Observation space
        blocks_dim = 4  # [blocktype, x, y, z]
        hand_dim = 5
        target_block_dim = 3  # type, distance, break_progress
        surrounding_blocks_shape = SURROUNDING_BLOCKS_SHAPE
        player_state_dim = 8  # x, y, z, yaw, pitch, health, alive, light_level

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=IMAGE_SHAPE, dtype=np.float32),
            'blocks': spaces.Box(low=0, high=1, shape=(blocks_dim,), dtype=np.float32),
            'hand': spaces.Box(low=0, high=1, shape=(hand_dim,), dtype=np.float32),
            'target_block': spaces.Box(low=0, high=1, shape=(target_block_dim,), dtype=np.float32),
            'surrounding_blocks': spaces.Box(low=0, high=1, shape=surrounding_blocks_shape[::-1], dtype=np.float32),
            'player_state': spaces.Box(low=0, high=1, shape=(player_state_dim,), dtype=np.float32)
        })

        # Initialize WebSocket connection parameters
        self.websocket = None
        self.loop = None
        self.connection_thread = None
        self.connected = False

        # Start WebSocket connection in a separate thread
        self.state_queue = asyncio.Queue()
        self.start_connection()

        # Initialize step counters and parameters
        self.steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.step_penalty = STEP_PENALTY

        self.cumulative_rewards = 0.0
        self.episode_counts = 0
        self.cumulative_directional_rewards = 0.0
        self.cumulative_movement_bonus = 0.0
        self.cumulative_block_reward = 0.0

        self.repetitive_non_productive_counter = 0
        self.prev_target_block = 0
        self.had_target_last_block = False
        self.block_break_history = deque(maxlen=30)
        self.recent_block_breaks = deque(maxlen=20)
        self.prev_break_progress = 0.0
        self.last_screenshot = None

    def init_camera(self):
        with self.capture_lock:
            if self.camera is not None:
                # dxcam doesn't need explicit release, but we re-init anyway
                pass

            left = self.minecraft_bounds['left']
            top = self.minecraft_bounds['top']
            right = left + self.minecraft_bounds['width']
            bottom = top + self.minecraft_bounds['height']

            self.camera = dxcam.create(output_color="BGR")
            self.capture_region = (left, top, right, bottom)
            self.capture_failure_count = 0

    def start_connection(self):
        self.loop = asyncio.new_event_loop()
        self.connection_thread = threading.Thread(target=self.run_loop, args=(self.loop,), daemon=True)
        self.connection_thread.start()

    def run_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect())

    async def connect(self):
        try:
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                self.connected = True
                logging.info(f"Connected to {self.uri}")

                while self.connected:
                    tasks = [asyncio.ensure_future(self.receive_state())]
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
        finally:
            self.connected = False
            self.websocket = None

    async def receive_state(self):
        try:
            message = await self.websocket.recv()
            state = json.loads(message)
            await self.state_queue.put(state)
        except websockets.ConnectionClosed:
            logging.warning("WebSocket connection closed.")
            self.connected = False
        except Exception as e:
            logging.error(f"Error receiving state: {e}")
            await asyncio.sleep(0.1)

    async def send_action(self, action_name):
        if self.connected and self.websocket is not None:
            message = {'action': action_name}
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logging.error(f"Error sending action: {e}")

    def normalize_blocks(self, broken_blocks):
        block_features = 4
        if isinstance(broken_blocks, list) and len(broken_blocks) > 0 and isinstance(broken_blocks[0], list):
            broken_blocks = broken_blocks[0]

        if isinstance(broken_blocks, list) and len(broken_blocks) == 4:
            broken_blocks = np.array(broken_blocks, dtype=np.float32)
            broken_blocks = np.clip(broken_blocks, 0.0, 1.0)
            return broken_blocks
        else:
            return np.zeros(block_features, dtype=np.float32)

    def normalize_target_block(self, state_dict):
        target_block = state_dict.get('target_block', [0.0, 0.0, 0.0])
        return np.array(target_block, dtype=np.float32)
    
    def normalize_hand(self, state_dict):
        held_item = state_dict.get('held_item', [0, 0, 0, 0, 0])
        held_item = np.array(held_item, dtype=np.float32)
        held_item = np.clip(held_item, 0.0, 1.0)
        return held_item

    def flatten_surrounding_blocks(self, state_dict):
        surrounding = state_dict.get('surrounding_blocks', [])
        flattened = np.array(surrounding, dtype=np.float32).reshape(13, 13, 4)
        flattened = flattened.transpose(2, 0, 1)
        flattened = np.clip(flattened, 0.0, 1.0)
        return flattened

    def step(self, action):
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer, got {type(action)}")

        action = int(action)

        future = asyncio.run_coroutine_threadsafe(
            self._async_step(action_name=self.ACTION_MAPPING[action]),
            self.loop
        )

        try:
            result = future.result(timeout=TIMEOUT_STEP_LONG)
            return result
        except Exception as e:
            logging.error(f"Error during step: {e}")
            raise e

    async def _async_step(self, action_name=None):
        # Start screenshot capture and action sending in parallel
        screenshot_future = asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)
        action_task = asyncio.create_task(self.send_action(action_name))
        await action_task

        # Get state with timeout
        try:
            state = await asyncio.wait_for(self.state_queue.get(), timeout=TIMEOUT_STATE)
        except asyncio.TimeoutError:
            state = None
            logging.warning("Did not receive state in time.")

        # Wait for screenshot (no timeout here since we trust capture to be quick)
        screenshot = await screenshot_future

        reward = STEP_PENALTY

        if state is not None:
            broken_blocks = state.get('broken_blocks', [0, 0, 0, 0])
            if isinstance(broken_blocks, list) and len(broken_blocks) > 0 and isinstance(broken_blocks[0], list):
                broken_blocks = broken_blocks[0]

            blocks_norm = self.normalize_blocks(broken_blocks)
            hand_norm = self.normalize_hand(state)
            target_block_norm = self.normalize_target_block(state)
            surrounding_blocks_norm = self.flatten_surrounding_blocks(state)
            
            x = state.get('x', 0.0)
            y = state.get('y', 0.0)
            z = state.get('z', 0.0)
            yaw = state.get('yaw', 0.0)
            pitch = state.get('pitch', 0.0)
            health = state.get('health', 1.0)
            alive = state.get('alive', True)
            light_level = state.get('light_level', 0)

            player_state = np.array([
                x, y, z, yaw, pitch, health, 1.0 if alive else 0.0, light_level
            ], dtype=np.float32)

            state_data = {
                'image': screenshot,
                'blocks': blocks_norm,
                'hand': hand_norm,
                'target_block': target_block_norm,
                'surrounding_blocks': surrounding_blocks_norm,
                'player_state': player_state
            }

            if not np.any(screenshot):
                logging.warning(f"Step {self.steps}: Screenshot is all zeros.")
            else:
                logging.debug(f"Step {self.steps}: Screenshot captured successfully.")

            # Rewards for blocks
            if blocks_norm[0] > 0.0:
                block_value = blocks_norm[0]  
                block_reward = (block_value ** 3) * MAX_BLOCK_REWARD
                reward += block_reward
                self.cumulative_block_reward += block_reward
                self.block_break_history.append(True)
                self.prev_break_progress = 0.0
            else:
                self.block_break_history.append(False)

            if target_block_norm[0] == 0:
                target_block_norm[2] = 0.0

            if action_name == "attack":
                blocks_broken_recently = any(self.block_break_history)
                current_progress = target_block_norm[2]

                if target_block_norm[0] > 0.0:
                    if current_progress > self.prev_break_progress:
                        progress_delta = current_progress - self.prev_break_progress
                        progress_reward = progress_delta * ATTACK_BLOCK_REWARD_SCALE
                        reward += progress_reward
                        self.cumulative_block_reward += progress_reward
                    elif current_progress < self.prev_break_progress and not blocks_broken_recently:
                        penalty = ATTACK_MISS_PENALTY * 5.0
                        reward += penalty
                        self.cumulative_block_reward += penalty
                else:
                    reward += ATTACK_MISS_PENALTY
                    self.cumulative_block_reward += ATTACK_MISS_PENALTY

                self.prev_break_progress = current_progress

            # Jump penalty
            if action_name in ["jump", "jump_walk_forward"]:
                reward += JUMP_PENALTY
                self.cumulative_directional_rewards += JUMP_PENALTY

            self.save_screenshot_if_needed(screenshot)

        else:
            logging.warning("No state received after action.")
            state_data = self._get_default_state()

        self.steps += 1

        combined_observation = state_data

        self.cumulative_rewards += reward
        if self.steps % 50 == 0 and self.uri == "ws://localhost:8081":
            print(f"Reward: {reward:.2f}, Cumulative Direction Reward: {self.cumulative_directional_rewards:.2f} Cumulative Rewards: {self.cumulative_rewards:.2f}, Cumulative Block Reward: {self.cumulative_block_reward:.2f}")

        terminated = self.steps >= self.max_episode_steps
        truncated = False
        info = {}

        return combined_observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        future = asyncio.run_coroutine_threadsafe(
            self._async_reset(),
            self.loop
        )

        try:
            result = future.result(timeout=TIMEOUT_RESET_LONG)
            return result
        except Exception as e:
            logging.error(f"Error during reset: {e}")
            raise e

    async def _async_reset(self):
        self.steps = 0
        self.cumulative_rewards = 0.0
        self.episode_counts = 0
        self.cumulative_directional_rewards = 0.0
        self.cumulative_movement_bonus = 0.0
        self.cumulative_block_reward = 0.0
        self.block_break_history.clear()

        self.repetitive_non_productive_counter = 0
        self.prev_target_block = 0
        self.prev_break_progress = 0.0

        try:
            # Clear state queue
            while not self.state_queue.empty():
                self.state_queue.get_nowait()

            if not self.connected or self.websocket is None:
                logging.error("WebSocket not connected.")
                state_data = self._get_default_state()
                return state_data, {}

            # Start action and screenshot in parallel
            screenshot_future = asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)
            action_task = asyncio.create_task(self.send_action("reset 2"))
            await action_task

            # Wait for state with timeout
            try:
                state = await asyncio.wait_for(self.state_queue.get(), timeout=TIMEOUT_RESET)
            except asyncio.TimeoutError:
                logging.warning("Reset: No state received.")
                state = None

            # Wait for screenshot (no direct timeout here, but usually quick)
            screenshot = await screenshot_future

            if not np.any(screenshot):
                logging.warning("Reset: Screenshot is all zeros.")
            else:
                logging.debug("Reset: Screenshot captured successfully.")

            if state is not None:
                broken_blocks = state.get('broken_blocks', [0, 0, 0, 0])
                blocks_norm = self.normalize_blocks(broken_blocks)
                hand_norm = self.normalize_hand(state)
                target_block_norm = self.normalize_target_block(state)
                surrounding_blocks_norm = self.flatten_surrounding_blocks(state)

                x = state.get('x', 0.0)
                y = state.get('y', 0.0)
                z = state.get('z', 0.0)
                yaw = state.get('yaw', 0.0)
                pitch = state.get('pitch', 0.0)
                health = state.get('health', 20.0)
                alive = state.get('alive', True)
                light_level = state.get('light_level', 0)

                player_state = np.array([
                    x, y, z, yaw, pitch, health, 1.0 if alive else 0.0, light_level
                ], dtype=np.float32)

                state_data = {
                    'image': screenshot,
                    'blocks': blocks_norm,
                    'hand': hand_norm,
                    'target_block': target_block_norm,
                    'surrounding_blocks': surrounding_blocks_norm,
                    'player_state': player_state
                }
                self.prev_sum_surrounding = surrounding_blocks_norm.sum()
            else:
                state_data = self._get_default_state()
                self.prev_sum_surrounding = 0.0

            # Optionally save screenshot if enabled
            self.save_screenshot_if_needed(screenshot)

            return state_data, {}

        except Exception as e:
            logging.error(f"Reset error: {e}")
            state_data = self._get_default_state()
            return state_data, {}

    def capture_screenshot(self):
        """Thread-safe screenshot capture with GPU acceleration and fallback."""
        with self.capture_lock:
            try:
                frame = self.camera.grab(region=self.capture_region)
                if frame is None:
                    # Frame not captured
                    self.capture_failure_count += 1
                    if self.capture_failure_count >= self.max_capture_failures:
                        logging.error("Max capture failures reached. Reinitializing camera.")
                        self.init_camera()
                    if self.last_screenshot is not None:
                        logging.warning("Using last valid screenshot due to capture failure.")
                        return self.last_screenshot
                    else:
                        return np.zeros(IMAGE_SHAPE, dtype=np.float32)
                
                self.capture_failure_count = 0  # Reset on success

                # GPU resizing
                if self.use_cuda:
                    try:
                        gpu_frame = cv2.cuda_GpuMat()
                        gpu_frame.upload(frame)
                        gpu_resized = cv2.cuda.resize(gpu_frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
                        resized = gpu_resized.download()
                    except Exception as e:
                        logging.warning(f"CUDA resize failed, fallback to CPU: {e}")
                        resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
                else:
                    resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

                img = resized.transpose(2, 0, 1) / 255.0
                img = img.astype(np.float32)
                self.last_screenshot = img
                return img
            except Exception as e:
                logging.error(f"Critical screenshot error: {e}")
                if self.last_screenshot is not None:
                    logging.warning("Using last valid screenshot due to critical error.")
                    return self.last_screenshot
                return np.zeros(IMAGE_SHAPE, dtype=np.float32)

    def save_screenshot_if_needed(self, screenshot):
        if self.save_screenshots and self.steps % 250 == 0:
            try:
                uri_suffix = self.uri[-1]
                timestamp = int(time.time() * 1000)
                img_save = (screenshot.transpose(1, 2, 0) * 255).astype(np.uint8)
                filename = f"screenshot_{timestamp}_env{uri_suffix}.png"
                filepath = os.path.join(self.screenshot_dir, filename)
                cv2.imwrite(filepath, cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
                logging.info(f"Saved screenshot: {filename}")
            except Exception as e:
                logging.error(f"Error saving screenshot: {e}")

    def _get_default_state(self):
        default_player_state = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

        default = {
            'image': np.zeros(IMAGE_SHAPE, dtype=np.float32),
            'blocks': np.zeros(4, dtype=np.float32),
            'hand': np.zeros(5, dtype=np.float32),
            'target_block': np.zeros(3, dtype=np.float32),
            'surrounding_blocks': np.zeros(SURROUNDING_BLOCKS_SHAPE, dtype=np.float32),
            'player_state': default_player_state
        }
        return default

    def render(self, mode='human'):
        pass

    def close(self):
        if self.connected and self.websocket:
            self.connected = False
            asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
            self.websocket = None

        with self.capture_lock:
            # dxcam cleanup if needed
            pass

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=1)

    def __del__(self):
        self.close()
