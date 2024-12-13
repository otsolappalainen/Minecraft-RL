import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
import websockets
import threading
import json
import time
import mss
import cv2
import pygetwindow as gw
import fnmatch
import os
import logging
import uuid
from collections import deque
import math
from threading import Lock


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

# Updated surrounding_blocks_dim for 13x13x4
SURROUNDING_BLOCKS_SHAPE = (13, 13, 4)




# Training environment. Client specific.

class MinecraftEnv(gym.Env):
    """
    Custom Environment that interfaces with Minecraft via WebSocket.
    Handles a single client per environment instance.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, uri="ws://localhost:8080", task=None, window_bounds=None, save_example_step_data=False):
        super(MinecraftEnv, self).__init__()
        
        # Initialize mss with thread safety
        self.screenshot_lock = Lock()
        self.sct = None  # Will be initialized in capture_screenshot
        self.minecraft_bounds = window_bounds if window_bounds else self.find_minecraft_window()
        
        # Set up logging
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


        # Single URI
        self.uri = uri

        # For testing the screenshot capture logic. Set to False for normal operation.
        self.save_screenshots = False  # Set to True to save screenshots
        if self.save_screenshots:
            self.screenshot_dir = "env_screenshots"
            os.makedirs(self.screenshot_dir, exist_ok=True)

         # Require window bounds
        if window_bounds is None:
            raise ValueError("window_bounds parameter is required")

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

        # Observation space without 'tasks'
        blocks_dim = 4  # 4 features per block
        hand_dim = 5
        target_block_dim = 3  # type, distance, break_progress
        surrounding_blocks_shape = SURROUNDING_BLOCKS_SHAPE  # 13x13x4
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

        # Initialize screenshot parameters
        self.minecraft_bounds = window_bounds if window_bounds else self.find_minecraft_window()

        # Initialize action and state queues
        self.state_queue = asyncio.Queue()

        # Start WebSocket connection in a separate thread
        self.start_connection()

        # Initialize task with height targets
        self.default_task = self.normalize_task({})
        self.task = self.default_task if task is None else np.array(task, dtype=np.float32)

        # Initialize step counters and parameters
        self.steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS
        
        self.step_penalty = STEP_PENALTY

        # Cumulative reward and episode count. Additional reward tracking.
        self.cumulative_rewards = 0.0
        self.episode_counts = 0
        self.cumulative_directional_rewards = 0.0
        self.cumulative_movement_bonus = 0.0
        self.cumulative_block_reward = 0.0

        # Initialize additional variables
        self.repetitive_non_productive_counter = 0  # Counter from 0 to REPETITIVE_NON_PRODUCTIVE_MAX
        self.prev_target_block = 0  # To track state changes
        self.had_target_last_block = False  # Track if the previous state had target_block = 1

        
        # Initialize block break history and progress. Used for rewards.
        self.block_break_history = deque(maxlen=30)
        self.recent_block_breaks = deque(maxlen=20)  # Track block breaks in last 20 steps
        self.prev_break_progress = 0.0


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

    def normalize_to_unit(self, value, range_min, range_max):
        """Normalize a value to [0, 1] range based on provided min and max."""
        return (value - range_min) / (range_max - range_min)

    def normalize_value(self, value, range_min, range_max):
        """Normalize a value to [-1, 1] range"""
        return 2 * (value - range_min) / (range_max - range_min) - 1

    def normalize_blocks(self, broken_blocks):
        """
        Process broken blocks data where input is [blocktype, x, y, z]
        Values should already be normalized between 0 and 1
        Returns array of 4 values, zeros if input invalid
        """
        block_features = 4  # [blocktype, x, y, z]

        # Handle multiple broken blocks by taking the first one
        if isinstance(broken_blocks, list) and len(broken_blocks) > 0 and isinstance(broken_blocks[0], list):
            broken_blocks = broken_blocks[0]

        if isinstance(broken_blocks, list) and len(broken_blocks) == 4:
            broken_blocks = np.array(broken_blocks, dtype=np.float32)
            # Just clip to ensure values are in valid range
            broken_blocks = np.clip(broken_blocks, 0.0, 1.0)
            return broken_blocks
        else:
            return np.zeros(block_features, dtype=np.float32)

    def normalize_target_block(self, state_dict):
        """Normalize target block data - use direct values"""
        target_block = state_dict.get('target_block', [0.0, 0.0, 0.0])
        return np.array(target_block, dtype=np.float32)
    

    def normalize_hand(self, state_dict):
        """Normalize hand/item data - use full array"""
        held_item = state_dict.get('held_item', [0, 0, 0, 0, 0])
        held_item = np.array(held_item, dtype=np.float32)
        held_item = np.clip(held_item, 0.0, 1.0)  # Ensure values are between 0 and 1
        return held_item


    def flatten_surrounding_blocks(self, state_dict):
        """Flatten surrounding blocks data and transpose to (4, 13, 13)"""
        surrounding = state_dict.get('surrounding_blocks', [])
        flattened = np.array(surrounding, dtype=np.float32).reshape(13, 13, 4)
        flattened = flattened.transpose(2, 0, 1)  # Transpose to (4, 13, 13)
        flattened = np.clip(flattened, 0.0, 1.0)  # Ensure values are between 0 and 1
        return flattened


    def step(self, action):
        """Handle single action"""
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer, got {type(action)}")

        action = int(action)

        # Schedule the coroutine and get a Future
        future = asyncio.run_coroutine_threadsafe(
            self._async_step(action_name=self.ACTION_MAPPING[action]),
            self.loop
        )

        try:
            # Wait for the result with a timeout if necessary
            result = future.result(timeout=TIMEOUT_STEP_LONG)  # Adjust timeout as needed
            return result
        except Exception as e:
            logging.error(f"Error during step: {e}")
            raise e

    async def _async_step(self, action_name=None):
        if action_name:
            await self.send_action(action_name)

        # Capture screenshot asynchronously
        screenshot_task = asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)

        # Receive state
        try:
            state = await asyncio.wait_for(self.state_queue.get(), timeout=TIMEOUT_STATE)
        except asyncio.TimeoutError:
            state = None
            logging.warning("Did not receive state in time.")

        # Wait for screenshot to complete
        screenshot = await screenshot_task

        # Initialize reward with small constant step penalty
        reward = STEP_PENALTY  # Base step penalty

        if state is not None:
            # Extract relevant data from state
            broken_blocks = state.get('broken_blocks', [0, 0, 0, 0])
            if isinstance(broken_blocks, list) and len(broken_blocks) > 0 and isinstance(broken_blocks[0], list):
                broken_blocks = broken_blocks[0]

            blocks_norm = self.normalize_blocks(broken_blocks)
            hand_norm = self.normalize_hand(state)
            target_block_norm = self.normalize_target_block(state)
            surrounding_blocks_norm = self.flatten_surrounding_blocks(state)
            
            x = state.get('x', 0.0)  # Normalized
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

            # Check if the screenshot is valid
            if not np.any(screenshot):
                logging.warning(f"Step {self.steps}: Screenshot is all zeros.")
            else:
                logging.debug(f"Step {self.steps}: Screenshot captured successfully.")

            # Optionally, log the shape and some statistics of the image
            logging.debug(f"Step {self.steps}: Image shape: {screenshot.shape}, "
                          f"Min: {screenshot.min()}, Max: {screenshot.max()}, Mean: {screenshot.mean()}")
            

            ##print(f"Min: {screenshot.min()}, Max: {screenshot.max()}, Mean: {screenshot.mean()}")

            if blocks_norm[0] > 0.0:
                # Block successfully broken
                block_value = blocks_norm[0]  
                block_reward = (block_value ** 3) * MAX_BLOCK_REWARD
                reward += block_reward
                self.cumulative_block_reward += block_reward
                self.block_break_history.append(True)
                self.prev_break_progress = 0.0  # Reset progress
            else:
                self.block_break_history.append(False)

            if target_block_norm[0] == 0:
                target_block_norm[2] = 0.0  # Reset progress if no target
            

            if action_name == "attack":
                blocks_broken_recently = any(self.block_break_history)
                current_progress = target_block_norm[2]  # Get breaking progress
                
                if target_block_norm[0] > 0.0:
                    # Target exists
                    if current_progress > self.prev_break_progress:
                        # Progress increased
                        progress_delta = current_progress - self.prev_break_progress
                        progress_reward = progress_delta * ATTACK_BLOCK_REWARD_SCALE
                        reward += progress_reward
                        self.cumulative_block_reward += progress_reward
                    elif current_progress < self.prev_break_progress and not blocks_broken_recently:
                        # Breaking interrupted
                        penalty = ATTACK_MISS_PENALTY * 5.0  # Double penalty for interruption
                        reward += penalty
                        self.cumulative_block_reward += penalty
                else:
                    # No valid target
                    reward += ATTACK_MISS_PENALTY
                    self.cumulative_block_reward += ATTACK_MISS_PENALTY
                
                self.prev_break_progress = current_progress

            # Jump penalty
            if action_name in ["jump", "jump_walk_forward"]:
                reward += JUMP_PENALTY
                self.cumulative_directional_rewards += JUMP_PENALTY

            # Movement and looking rewards/penalties
            blocks_broken_recently = any(self.recent_block_breaks)
            if blocks_broken_recently:
                if action_name == "move_forward" and (abs(x - 0.5) > 0.02 or abs(z - 0.5) > 0.02):
                    reward += FORWARD_MOVE_BONUS
                    self.cumulative_directional_rewards += FORWARD_MOVE_BONUS

                if action_name in ["look_up", "look_down"]:
                    reward += LOOK_ADJUST_BONUS
                    self.cumulative_directional_rewards += LOOK_ADJUST_BONUS

                if action_name in ["look_left", "look_right", "move_left", "move_right"] and target_block_norm[0] < 0.5:
                    reward += SIDE_PENALTY
                    self.cumulative_directional_rewards += SIDE_PENALTY

                if action_name in ["move_backward", "jump"]:
                    reward += BACKWARD_PENALTY
                    self.cumulative_directional_rewards += BACKWARD_PENALTY

            # Update previous action
            self.previous_action = action_name

            self.save_screenshot_if_needed(screenshot)

        else:
            logging.warning("No state received after action.")
            state_data = self._get_default_state()

        self.steps += 1

        # Prepare combined_observation
        combined_observation = state_data

        # Prints how the different rewards are contributing to the cumulative reward. Only prints for the 1st training environment.
        self.cumulative_rewards += reward
        if self.steps % 50 == 0 and self.uri == "ws://localhost:8081":
                print(f"Reward: {reward:.2f}, Cumulative Direction Reward: {self.cumulative_directional_rewards:.2f} Cumulative Rewards: {self.cumulative_rewards:.2f}, Cumulative Block Reward: {self.cumulative_block_reward:.2f}")


        # Check if episode is terminated
        terminated = self.steps >= self.max_episode_steps
        truncated = False
        info = {}

        return combined_observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Gym synchronous reset method"""
        if seed is not None:
            np.random.seed(seed)

        # Schedule the coroutine and get a Future
        future = asyncio.run_coroutine_threadsafe(
            self._async_reset(),
            self.loop
        )

        try:
            # Wait for the result with a timeout if necessary
            result = future.result(timeout=TIMEOUT_RESET_LONG)  # Adjust timeout as needed
            return result
        except Exception as e:
            logging.error(f"Error during reset: {e}")
            raise e

    async def _async_reset(self):
        """Asynchronous reset implementation"""
        self.steps = 0
        self.cumulative_rewards = 0.0
        self.episode_counts = 0
        self.cumulative_directional_rewards = 0.0
        self.cumulative_movement_bonus = 0.0
        self.cumulative_block_reward = 0.0
        self.block_break_history.clear()

        # Reset additional variables
        self.repetitive_non_productive_counter = 0
        self.prev_target_block = 0
        self.prev_break_progress = 0.0

        try:
            # Clear the state queue
            while not self.state_queue.empty():
                self.state_queue.get_nowait()

            if not self.connected or self.websocket is None:
                logging.error("WebSocket not connected.")
                state_data = self._get_default_state()
                return state_data, {}

            # Send reset action
            await self.send_action("reset 2")

            # Receive state with timeout
            try:
                state = await asyncio.wait_for(self.state_queue.get(), timeout=TIMEOUT_RESET)
            except asyncio.TimeoutError:
                logging.warning("Reset: No state received.")
                state_data = self._get_default_state()
                return state_data, {}

            # Capture screenshot asynchronously
            screenshot = await asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)
            
            # Check if the screenshot is valid
            if not np.any(screenshot):
                logging.warning("Reset: Screenshot is all zeros.")
            else:
                logging.debug("Reset: Screenshot captured successfully.")

            # Optionally, log the shape and some statistics of the image
            logging.debug(f"Reset: Image shape: {screenshot.shape}, "
                          f"Min: {screenshot.min()}, Max: {screenshot.max()}, Mean: {screenshot.mean()}")

            # Process state
            if state is not None:
                broken_blocks = state.get('broken_blocks', [0, 0, 0, 0])
                blocks_norm = self.normalize_blocks(broken_blocks)
                hand_norm = self.normalize_hand(state)
                target_block_norm = self.normalize_target_block(state)
                surrounding_blocks_norm = self.flatten_surrounding_blocks(state)
                

                # Extract and normalize player state
                x = state.get('x', 0.0)  # Now comes pre-normalized
                y = state.get('y', 0.0)  # Now comes pre-normalized 
                z = state.get('z', 0.0)  # Now comes pre-normalized
                yaw = state.get('yaw', 0.0)
                pitch = state.get('pitch', 0.0)
                health = state.get('health', 20.0)
                alive = state.get('alive', True)
                light_level = state.get('light_level', 0)

                # Remove coordinate normalization, keep other normalizations
                player_state = np.array([
                    x,  # Already normalized
                    y,  # Already normalized
                    z,  # Already normalized
                    yaw,
                    pitch,
                    health,
                    1.0 if alive else 0.0,
                    light_level
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

            return state_data, {}

        except Exception as e:
            logging.error(f"Reset error: {e}")
            state_data = self._get_default_state()
            return state_data, {}

    def capture_screenshot(self):
        """Thread-safe screenshot capture"""
        with self.screenshot_lock:
            try:
                # Lazy initialization of mss
                if self.sct is None:
                    self.sct = mss.mss()
                
                screenshot = self.sct.grab(self.minecraft_bounds)
                img = np.array(screenshot)[:, :, :3]
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), 
                               interpolation=cv2.INTER_NEAREST)
                img = img.transpose(2, 0, 1) / 255.0
                return img.astype(np.float32)
            except Exception as e:
                # Only log critical screenshot errors
                if "srcdc" not in str(e):
                    logging.error(f"Critical screenshot error: {e}")
                return np.zeros(IMAGE_SHAPE, dtype=np.float32)

    def save_screenshot_if_needed(self, screenshot):
        """Save screenshot if enabled and meets timing conditions"""
        if self.save_screenshots and self.steps % 250 == 0:
            try:
                # Get the last character of the URI (e.g., '0' or '1')
                uri_suffix = self.uri[-1]
                
                # Create timestamp
                timestamp = int(time.time() * 1000)
                
                # Convert screenshot from CHW to HWC format and scale to 0-255
                img_save = (screenshot.transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Create filename with timestamp and URI suffix
                filename = f"screenshot_{timestamp}_env{uri_suffix}.png"
                filepath = os.path.join(self.screenshot_dir, filename)
                
                # Save the image
                cv2.imwrite(filepath, cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
                logging.info(f"Saved screenshot: {filename}")
                
            except Exception as e:
                logging.error(f"Error saving screenshot: {e}")


    # If some of the observations are not available, return default values, to avoid crashing the training loop.
    def _get_default_state(self):
        """Return default state when real state cannot be obtained"""
        default_player_state = np.array([
            0.0,  # x
            0.0,  # y
            0.0,  # z
            0.0,  # yaw
            0.0,  # pitch
            0.0,  # health
            0.0,  # alive
            0.0   # light_level
        ], dtype=np.float32)

        default = {
            'image': np.zeros(IMAGE_SHAPE, dtype=np.float32),
            'blocks': np.zeros(4, dtype=np.float32),    # block_features = 4
            'hand': np.zeros(5, dtype=np.float32),      # hand_dim = 5
            'target_block': np.full(3, dtype=np.float32),  # New spatial shape with 0.5 default
            'surrounding_blocks': np.zeros(SURROUNDING_BLOCKS_SHAPE, dtype=np.float32),  # 13x13x4
            'player_state': default_player_state
        }
        return default

    # Not used currently
    def render(self, mode='human'):
        """Render the environment if needed."""
        pass

    # Clean up resources
    def close(self):
        """Clean up resources"""
        if self.connected and self.websocket:
            self.connected = False
            asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
            self.websocket = None

        # Clean up mss
        with self.screenshot_lock:
            if self.sct:
                self.sct.close()
                self.sct = None

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=1)

    def __del__(self):
        """Cleanup before deleting the environment object."""
        self.close()
