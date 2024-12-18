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
import concurrent.futures

# At the top of the file, add these color codes
COLORS = {
    'blue': '\033[94m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'purple': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'end': '\033[0m'
}

# Training settings
MAX_EPISODE_STEPS = 1024

# Reward settings
STEP_PENALTY = -0.1  # Smaller constant penalty
MOVEMENT_REWARD_SCALE = 20.0  # Scale for x-position based reward
HEALTH_LOSS_PENALTY = -5.0  # Penalty for losing health
DEATH_PENALTY = -10.0  # Penalty for being dead

# Timeout settings in seconds.
TIMEOUT_STEP = 30
TIMEOUT_STATE = 30
TIMEOUT_RESET = 30
TIMEOUT_STEP_LONG = 30
TIMEOUT_RESET_LONG = 30
MIN_POS = 0.49
MAX_POS = 0.51

# Screenshot settings
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 240
IMAGE_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)

# Replace SURROUNDING_BLOCKS_SHAPE with:
SURROUNDING_BLOCKS_DIM = 12  # 12 directional values




# Training environment. Client specific.

class MinecraftEnv(gym.Env):
    """
    Custom Environment that interfaces with Minecraft via WebSocket.
    Handles a single client per environment instance.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, uri="ws://localhost:8080", task=None, window_bounds=None, save_example_step_data=False):
        super(MinecraftEnv, self).__init__()
        
        # Add connection retry parameters at the start of __init__
        self.max_reconnect_attempts = 3
        self.reconnect_delay = 2  # seconds
        
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
            11: "attack 2",
            12: "use",
            13: "next_item",
            14: "previous_item",
            15: "no_op"
        }

        self.action_space = spaces.Discrete(len(self.ACTION_MAPPING))

        # Observation space without 'tasks'
        blocks_dim = 4  # 4 features per block
        hand_dim = 5
        mobs_dim = 3  # type, distance, break_progress
        surrounding_blocks_shape = SURROUNDING_BLOCKS_DIM  # 13x13x4
        player_state_dim = 8  # x, y, z, yaw, pitch, health, alive, light_level

        # Update observation space in __init__:
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=IMAGE_SHAPE, dtype=np.float32),
            'blocks': spaces.Box(low=0, high=1, shape=(blocks_dim,), dtype=np.float32),
            'hand': spaces.Box(low=0, high=1, shape=(hand_dim,), dtype=np.float32),
            'mobs': spaces.Box(low=0, high=1, shape=(mobs_dim,), dtype=np.float32),
            'surrounding_blocks': spaces.Box(low=0, high=1, shape=(SURROUNDING_BLOCKS_DIM,), dtype=np.float32),
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
        self.prev_small_blocks_count = None
        self.cumulative_obstacle_reward = 0.0
        self.total_walk_forward_reward = 0.0
        self.total_walk_reward = 0.0
        self.total_walk_forward_look_forward_reward = 0.0
        self.total_health_loss = 0.0
        self.total_attack_reward = 0.0
        self.total_hit_reward = 0.0
        self.total_death_penalty = 0.0
        self.prev_surrounding_blocks = None
        self.DIRECTIONAL_REWARD = 0.5

        # Initialize additional variables
        self.repetitive_non_productive_counter = 0  # Counter from 0 to REPETITIVE_NON_PRODUCTIVE_MAX
        self.prev_mobs = 0  # To track state changes
        self.had_mob = False  # Track if the previous state had target_block = 1

        
        # Initialize block break history and progress. Used for rewards.
        self.block_break_history = deque(maxlen=30)
        self.recent_block_breaks = deque(maxlen=20)  # Track block breaks in last 20 steps
        self.prev_break_progress = 0.0

        self.last_screenshot = None


    def start_connection(self):
        self.loop = asyncio.new_event_loop()
        self.connection_thread = threading.Thread(target=self.run_loop, args=(self.loop,), daemon=True)
        self.connection_thread.start()

    def run_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect())

    async def connect(self):
        attempts = 0
        while attempts < self.max_reconnect_attempts:
            try:
                async with websockets.connect(self.uri) as websocket:
                    self.websocket = websocket
                    self.connected = True
                    logging.info(f"Connected to {self.uri}")

                    while self.connected:
                        try:
                            tasks = [asyncio.ensure_future(self.receive_state())]
                            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        except asyncio.TimeoutError:
                            logging.warning("State receive timeout, attempting reconnection")
                            break
                        
            except Exception as e:
                attempts += 1
                logging.error(f"Connection attempt {attempts} failed: {e}")
                if attempts < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    logging.error("Max reconnection attempts reached")
                    raise

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
        Returns array of exactly 4 values, padded or truncated as needed
        """
        block_features = 4  # [blocktype, x, y, z]
        
        # Handle multiple broken blocks by taking the first one
        if isinstance(broken_blocks, list) and len(broken_blocks) > 0 and isinstance(broken_blocks[0], list):
            broken_blocks = broken_blocks[0]
        
        # Convert to numpy array if it's a list
        if isinstance(broken_blocks, list):
            broken_blocks = np.array(broken_blocks, dtype=np.float32)
        
        # Create output array
        result = np.zeros(block_features, dtype=np.float32)
        
        # Copy data, handling both short and long inputs
        if len(broken_blocks) > 0:
            result[:min(len(broken_blocks), block_features)] = broken_blocks[:block_features]
        
        # Clip values to valid range
        return np.clip(result, 0.0, 1.0)

    def normalize_mobs(self, state_dict):
        """Normalize target block data - use direct values"""
        mobs = state_dict.get('mobs', [0.0, 0.0, 0.0])
        return np.array(mobs, dtype=np.float32)
    

    def get_highest_value_index(self, surrounding_blocks):
        """Get index of highest value in surrounding blocks array"""
        return np.argmax(surrounding_blocks)
    
    def calculate_directional_reward(self, prev_index, new_index, action_name, mobs_norm):
        """Calculate reward for turning towards highest value"""
        if mobs_norm[0] == 1:  # If mob present, no directional reward
            return 0.0
            
        # Only process turn actions
        if action_name in ["look_left", "look_right"]:
            return 0.0
            
        # No previous state
        if prev_index is None:
            return 0.0
            
        array_len = SURROUNDING_BLOCKS_DIM
        target_index = array_len - 1  # 12 o'clock position (last index)
        
        # Calculate distances to 12 o'clock before and after turn
        prev_distance = min(
            abs(prev_index - target_index),
            abs(prev_index + array_len - target_index)
        )
        new_distance = min(
            abs(new_index - target_index),
            abs(new_index + array_len - target_index)
        )
        
        # Reward if turned closer to 12 o'clock
        if new_distance < prev_distance:
            return self.DIRECTIONAL_REWARD
        elif new_distance > prev_distance:
            return -self.DIRECTIONAL_REWARD
        return 0.0

    def normalize_hand(self, state_dict):
        """
        Normalize hand/item data to exactly 5 values
        """
        hand_dim = 5
        held_item = state_dict.get('held_item', [0] * hand_dim)
        
        # Convert to numpy array if needed
        if isinstance(held_item, list):
            held_item = np.array(held_item, dtype=np.float32)
        
        # Create output array
        result = np.zeros(hand_dim, dtype=np.float32)
        
        # Copy data, handling both short and long inputs
        if len(held_item) > 0:
            result[:min(len(held_item), hand_dim)] = held_item[:hand_dim]
        
        return np.clip(result, 0.0, 1.0)

    # Replace flatten_surrounding_blocks with:
    def flatten_surrounding_blocks(self, state_dict):
        """Convert surrounding blocks data into 12 directional values"""
        try:
            surrounding = state_dict.get('surrounding_blocks', [])
            
            if not surrounding or len(surrounding) == 0:
                return np.zeros(SURROUNDING_BLOCKS_DIM, dtype=np.float32)
                
            # Convert incoming data to 12 directional values
            # Example: [front, back, left, right, up, down, front-left, front-right, back-left, back-right, up-front, up-back]
            directional = np.array(surrounding[:SURROUNDING_BLOCKS_DIM], dtype=np.float32)
            
            # Pad if too short, truncate if too long
            if len(directional) < SURROUNDING_BLOCKS_DIM:
                directional = np.pad(directional, (0, SURROUNDING_BLOCKS_DIM - len(directional)))
            else:
                directional = directional[:SURROUNDING_BLOCKS_DIM]
                
            return np.clip(directional, 0.0, 1.0)
                
        except Exception as e:
            logging.warning(f"Error processing surrounding blocks: {e}")
            return np.zeros(SURROUNDING_BLOCKS_DIM, dtype=np.float32)

    def count_small_surrounding_blocks(self, state_dict, threshold=0.4):
       
        surrounding_blocks = self.flatten_surrounding_blocks(state_dict)
        # Count values below threshold
        count = np.sum(surrounding_blocks < threshold)
        return count
    
    def normalize_block_change(self, change, reference_change=10.0):
        """Normalize block count change to [-1,1] range"""
        return np.clip(change / reference_change, -1.0, 1.0)


    def step(self, action):
        """Handle single action with improved error handling"""
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer, got {type(action)}")

        try:
            # Schedule the coroutine and get a Future
            future = asyncio.run_coroutine_threadsafe(
                self._async_step(action_name=self.ACTION_MAPPING[action]),
                self.loop
            )

            # Wait for result with timeout
            result = future.result(timeout=TIMEOUT_STEP_LONG)
            return result
            
        except TimeoutError:
            logging.error("Step timeout - returning default state")
            # Return safe default values instead of raising
            return self._get_default_state(), STEP_PENALTY, True, False, {}
            
        except Exception as e:
            logging.error(f"Error during step: {e}")
            # Attempt reconnection
            self.start_connection()
            return self._get_default_state(), STEP_PENALTY, True, False, {}

    async def _async_step(self, action_name=None):
        if action_name:
            await self.send_action(action_name)

        # Capture screenshot asynchronously with timeout
        try:
            screenshot_task = asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)
            screenshot = await asyncio.wait_for(screenshot_task, timeout=TIMEOUT_STEP)
        
                
        except asyncio.TimeoutError:
            logging.warning("Screenshot capture timed out")
            screenshot = np.zeros(IMAGE_SHAPE, dtype=np.float32)

        if self.save_screenshots:
            self.save_screenshot_if_needed(screenshot)

        # Receive state
        try:
            state = await asyncio.wait_for(self.state_queue.get(), timeout=TIMEOUT_STATE)
        except asyncio.TimeoutError:
            state = None
            logging.warning("Did not receive state in time.")

        # Optionally, log the shape and some statistics of the image
            logging.debug(f"Reset: Image shape: {screenshot.shape}, "
                          f"Min: {screenshot.min()}, Max: {screenshot.max()}, Mean: {screenshot.mean()}")

        # Initialize reward with small constant step penalty
        reward = STEP_PENALTY  # Base step penalty

        if state is not None:
            
            # Extract relevant data from state
            broken_blocks = state.get('broken_blocks', [0, 0, 0, 0])
            if isinstance(broken_blocks, list) and len(broken_blocks) > 0 and isinstance(broken_blocks[0], list):
                broken_blocks = broken_blocks[0]

            blocks_norm = self.normalize_blocks(broken_blocks)
            hand_norm = self.normalize_hand(state)
            mobs_norm = self.normalize_mobs(state)
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

            
            
            # Store previous health if not stored
            if not hasattr(self, 'previous_health'):
                self.previous_health = health

            if self.prev_surrounding_blocks is not None:
                prev_highest = self.get_highest_value_index(self.prev_surrounding_blocks)
                new_highest = self.get_highest_value_index(surrounding_blocks_norm)
                
                directional_reward = self.calculate_directional_reward(
                    prev_highest, new_highest, action_name, mobs_norm
                )
                reward += directional_reward

            self.prev_surrounding_blocks = surrounding_blocks_norm.copy()

            # Add penalty for health loss
            if health < self.previous_health:
                reward += HEALTH_LOSS_PENALTY
                self.total_health_loss += HEALTH_LOSS_PENALTY
            
            # Add death penalty if not alive
            if not alive:
                reward += DEATH_PENALTY
                self.total_death_penalty += DEATH_PENALTY

            if action_name == "attack 2":
                # Reward for attacking a mob
                reward += 0.05
                self.total_attack_reward += 0.05

            if x < MIN_POS or x > MAX_POS or z < MIN_POS or z > MAX_POS:
                reward += 0.05

            if mobs_norm[1] > 0.0:
                # Reward for looking at a mob
                reward += 4.0
                self.total_hit_reward += 4.0
        

            # Update previous health
            self.previous_health = health

            if self.steps == 0:
                # Clip reward between -1 and 1
                reward = max(-0.5, min(0.5, reward))

            # Prepare state data dictionary
            state_data = {
                'image': screenshot,
                'blocks': blocks_norm,
                'hand': hand_norm,
                'mobs': mobs_norm,
                'surrounding_blocks': surrounding_blocks_norm,
                'player_state': player_state
            }

        else:
            logging.warning("No state received after action.")
            state_data = self._get_default_state()

        self.steps += 1
        
        # Update cumulative rewards for tracking
        self.cumulative_rewards += reward

        # Print debug info periodically
        if self.steps % 50 == 0 and self.uri == "ws://localhost:8081":
            # Format each stat with fixed width and color
            stats = [
                f"{COLORS['white']}Step: {COLORS['cyan']}{self.steps:<6}",
                f"{COLORS['white']}HEALTH: {COLORS['yellow']}{health:>6.2f}",
                f"{COLORS['white']}Reward: {COLORS['green']}{reward:>8.2f}",
                f"{COLORS['white']}Total: {COLORS['red']}{self.cumulative_rewards:>8.2f}",
            ]
            
            walk_stats = [
                f"{COLORS['white']}HIT: {COLORS['yellow']}{self.total_hit_reward:>8.2f}",
                f"{COLORS['white']}DEATH: {COLORS['green']}{self.total_death_penalty:>8.2f}",
                f"{COLORS['white']}HEALTH: {COLORS['cyan']}{self.total_health_loss:>8.2f}",
                f"{COLORS['white']}ATTACK: {COLORS['cyan']}{self.total_attack_reward:>8.2f}"
            ]
            
            # Print both lines with end color code
            print('  '.join(stats) + COLORS['end'])
            print('  '.join(walk_stats) + COLORS['end'])


        terminated = self.steps >= self.max_episode_steps
        truncated = False
        info = {}

        return state_data, reward, terminated, truncated, info

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
        self.cumulative_obstacle_reward = 0.0
        self.total_walk_forward_reward = 0.0
        self.total_walk_reward = 0.0
        self.total_walk_forward_look_forward_reward = 0.0
        self.total_health_loss = 0.0
        self.total_attack_reward = 0.0
        self.total_hit_reward = 0.0
        self.total_death_penalty = 0.0
        self.prev_surrounding_blocks = None

        # Reset additional variables
        self.repetitive_non_productive_counter = 0
        self.prev_mobs = 0
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

            # Capture screenshot asynchronously with timeout
            try:
                screenshot_task = asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)
                screenshot = await asyncio.wait_for(screenshot_task, timeout=TIMEOUT_STEP)
            except asyncio.TimeoutError:
                logging.warning("Reset: Screenshot capture timed out")
                screenshot = np.zeros(IMAGE_SHAPE, dtype=np.float32)

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
                mobs_norm = self.normalize_mobs(state)
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
                    'mobs': mobs_norm,
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
                img = img.transpose(2, 0, 1) / 255.0
                return img.astype(np.float32)
            except Exception as e:
                # Only log critical screenshot errors
                if "srcdc" not in str(e):
                    logging.error(f"Critical screenshot error: {e}")
                return np.zeros(IMAGE_SHAPE, dtype=np.float32)

    def save_screenshot_if_needed(self, screenshot):
        """Save screenshot with step count and timestamp"""
        try:
            # Get environment identifier from URI
            uri_suffix = self.uri.split(":")[-1]
            
            # Create timestamp and step count
            timestamp = int(time.time() * 1000)
            
            # Convert screenshot from CHW to HWC format and scale to 0-255
            img_save = (screenshot.transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # Create filename with step count, timestamp and URI suffix
            filename = f"step_{self.steps:06d}_port{uri_suffix}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
            logging.debug(f"Saved screenshot: {filename}")
            
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")


    # If some of the observations are not available, return default values, to avoid crashing the training loop.
    # Update _get_default_state:
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
            'mobs': np.full(3, 0.0, dtype=np.float32),  # Added fill_value=0.0
            'surrounding_blocks': np.zeros(SURROUNDING_BLOCKS_DIM, dtype=np.float32),
            'player_state': default_player_state
        }
        return default

    # Not used currently
    def render(self, mode='human'):
        """Render the environment if needed."""
        pass

    # Clean up resources
    def close(self):
        """Improved cleanup with error handling"""
        try:
            if self.connected and self.websocket:
                self.connected = False
                asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
                self.websocket = None

            # Clean up mss with lock protection
            with self.screenshot_lock:
                if hasattr(self, 'sct') and self.sct:
                    try:
                        self.sct.close()
                    except Exception as e:
                        logging.warning(f"Error closing screenshot capture: {e}")
                    self.sct = None

            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)

            if self.connection_thread and self.connection_thread.is_alive():
                self.connection_thread.join(timeout=1)
                
        except Exception as e:
            logging.error(f"Error during environment cleanup: {e}")

    def __del__(self):
        """Cleanup before deleting the environment object."""
        self.close()