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
import logging
from collections import deque
from threading import Lock
import os

# Reduce logging level to ERROR to minimize overhead
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_EPISODE_STEPS = 1024
STEP_PENALTY = -0.1
HEALTH_LOSS_PENALTY = -2.5
DEATH_PENALTY = -10.0
TIMEOUT_STEP_LONG = 30
TIMEOUT_STATE = 30
TIMEOUT_RESET_LONG = 30
TIMEOUT_RESET = 30
TIMEOUT_STEP = 30
MIN_POS = 0.49
MAX_POS = 0.51
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 120
IMAGE_WIDTH = 240
IMAGE_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
SURROUNDING_BLOCKS_DIM = 12

# Preallocate default states to avoid repeated np.zeros calls
DEFAULT_IMAGE = np.zeros(IMAGE_SHAPE, dtype=np.float32)
DEFAULT_BLOCKS = np.zeros(4, dtype=np.float32)
DEFAULT_HAND = np.zeros(5, dtype=np.float32)
DEFAULT_MOBS = np.zeros(3, dtype=np.float32)
DEFAULT_SURROUNDING = np.zeros(SURROUNDING_BLOCKS_DIM, dtype=np.float32)
DEFAULT_PLAYER_STATE = np.zeros(8, dtype=np.float32)  # [x,y,z,yaw,pitch,health,alive,light]
DEFAULT_STATE = {
    'image': DEFAULT_IMAGE,
    'blocks': DEFAULT_BLOCKS,
    'hand': DEFAULT_HAND,
    'mobs': DEFAULT_MOBS,
    'surrounding_blocks': DEFAULT_SURROUNDING,
    'player_state': DEFAULT_PLAYER_STATE
}

class MinecraftEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, uri="ws://localhost:8080", window_bounds=None):
        super().__init__()
        
        if window_bounds is None:
            raise ValueError("window_bounds parameter is required")
        self.minecraft_bounds = window_bounds
        self.uri = uri

        self.max_reconnect_attempts = 3
        self.reconnect_delay = 2

        # Thread-safe screenshot capture
        self.screenshot_lock = Lock()
        self.sct = None

        self.save_screenshots = False
        self.screenshot_dir = "env_screenshots"  # Add this line
        if self.save_screenshots:
            os.makedirs(self.screenshot_dir, exist_ok=True)  # Add this line

        self.ACTION_MAPPING = {
            0: "move_forward", 1: "move_backward", 2: "move_left", 3: "move_right",
            4: "jump_walk_forward", 5: "jump", 6: "sneak", 7: "look_left",
            8: "look_right", 9: "look_up", 10: "look_down", 11: "attack 2",
            12: "use", 13: "next_item", 14: "previous_item", 15: "no_op"
        }
        self.action_space = spaces.Discrete(len(self.ACTION_MAPPING))

        blocks_dim = 4
        hand_dim = 5
        mobs_dim = 3
        player_state_dim = 8

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=IMAGE_SHAPE, dtype=np.float32),
            'blocks': spaces.Box(low=0, high=1, shape=(blocks_dim,), dtype=np.float32),
            'hand': spaces.Box(low=0, high=1, shape=(hand_dim,), dtype=np.float32),
            'mobs': spaces.Box(low=0, high=1, shape=(mobs_dim,), dtype=np.float32),
            'surrounding_blocks': spaces.Box(low=0, high=1, shape=(SURROUNDING_BLOCKS_DIM,), dtype=np.float32),
            'player_state': spaces.Box(low=0, high=1, shape=(player_state_dim,), dtype=np.float32)
        })

        self.websocket = None
        self.loop = None
        self.connection_thread = None
        self.connected = False

        self.state_queue = asyncio.Queue()
        self.start_connection()

        # Environment variables
        self.steps = 0
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.step_penalty = STEP_PENALTY

        # Tracking cumulative rewards and stats
        self.reset_internal_stats()

        self.block_break_history = deque(maxlen=30)
        self.recent_block_breaks = deque(maxlen=20)
        self.prev_break_progress = 0.0
        self.DIRECTIONAL_REWARD = 0.5
        self.last_screenshot = None

    def reset_internal_stats(self):
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
        self.repetitive_non_productive_counter = 0
        self.prev_mobs = 0
        self.had_mob = False

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
                    while self.connected:
                        try:
                            await asyncio.wait([asyncio.ensure_future(self.receive_state())], return_when=asyncio.FIRST_COMPLETED)
                        except asyncio.TimeoutError:
                            break
            except:
                attempts += 1
                if attempts < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    raise

    async def receive_state(self):
        try:
            message = await self.websocket.recv()
            state = json.loads(message)
            await self.state_queue.put(state)
        except:
            self.connected = False

    async def send_action(self, action_name):
        if self.connected and self.websocket is not None:
            message = {'action': action_name}
            try:
                await self.websocket.send(json.dumps(message))
            except:
                pass

    def _get_default_state(self):
        return DEFAULT_STATE

    def normalize_blocks(self, broken_blocks):
        block_features = 4
        if isinstance(broken_blocks, list) and len(broken_blocks) > 0 and isinstance(broken_blocks[0], list):
            broken_blocks = broken_blocks[0]
        if isinstance(broken_blocks, list):
            arr = np.array(broken_blocks[:block_features], dtype=np.float32)
        else:
            arr = broken_blocks
        result = np.zeros(block_features, dtype=np.float32)
        length = min(len(arr), block_features)
        result[:length] = arr[:length]
        return np.clip(result, 0.0, 1.0)

    def normalize_hand(self, state):
        hand_dim = 5
        held_item = state.get('held_item', [0]*hand_dim)
        if isinstance(held_item, list):
            held_item = np.array(held_item[:hand_dim], dtype=np.float32)
        result = np.zeros(hand_dim, dtype=np.float32)
        length = min(len(held_item), hand_dim)
        result[:length] = held_item[:length]
        return np.clip(result, 0.0, 1.0)

    def normalize_mobs(self, state):
        mobs = state.get('mobs', [0.0,0.0,0.0])
        return np.array(mobs, dtype=np.float32)

    def flatten_surrounding_blocks(self, state):
        surrounding = state.get('surrounding_blocks', [])
        if not surrounding:
            return DEFAULT_SURROUNDING
        directional = np.array(surrounding[:SURROUNDING_BLOCKS_DIM], dtype=np.float32)
        if len(directional) < SURROUNDING_BLOCKS_DIM:
            directional = np.pad(directional, (0, SURROUNDING_BLOCKS_DIM - len(directional)))
        else:
            directional = directional[:SURROUNDING_BLOCKS_DIM]
        return np.clip(directional, 0.0, 1.0)

    def get_highest_value_index(self, arr):
        return np.argmax(arr)

    def calculate_directional_reward(self, prev_index, new_index, action_name, mobs_norm):
        if mobs_norm[0] == 1:  # no directional reward if mob present
            return 0.0
        if action_name in ["look_left", "look_right"] or prev_index is None:
            return 0.0
        array_len = SURROUNDING_BLOCKS_DIM
        target_index = array_len - 1
        prev_distance = min(abs(prev_index - target_index), abs(prev_index + array_len - target_index))
        new_distance = min(abs(new_index - target_index), abs(new_index + array_len - target_index))
        if new_distance < prev_distance:
            return self.DIRECTIONAL_REWARD
        elif new_distance > prev_distance:
            return -self.DIRECTIONAL_REWARD
        return 0.0

    def step(self, action):
        if not isinstance(action, (int, np.integer)):
            raise ValueError("Action must be an integer")

        try:
            future = asyncio.run_coroutine_threadsafe(self._async_step(self.ACTION_MAPPING[action]), self.loop)
            return future.result(timeout=TIMEOUT_STEP_LONG)
        except:
            # On timeout or error, return default state
            self.start_connection()  # Attempt reconnection
            return self._get_default_state(), STEP_PENALTY, True, False, {}

    async def _async_step(self, action_name=None):
        if action_name:
            await self.send_action(action_name)

        try:
            screenshot_task = asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)
            screenshot = await asyncio.wait_for(screenshot_task, timeout=TIMEOUT_STEP)
        except asyncio.TimeoutError:
            screenshot = DEFAULT_IMAGE

        if self.save_screenshots and self.uri == "ws://localhost:8081":
            self.save_screenshot_if_needed(screenshot)

        try:
            state = await asyncio.wait_for(self.state_queue.get(), timeout=TIMEOUT_STATE)
        except asyncio.TimeoutError:
            state = None

        reward = STEP_PENALTY
        if state is not None:
            broken_blocks = state.get('broken_blocks', [0,0,0,0])
            blocks_norm = self.normalize_blocks(broken_blocks)
            hand_norm = self.normalize_hand(state)
            mobs_norm = self.normalize_mobs(state)
            surrounding_blocks_norm = self.flatten_surrounding_blocks(state)

            x = state.get('x',0.0)
            y = state.get('y',0.0)
            z = state.get('z',0.0)
            yaw = state.get('yaw',0.0)
            pitch = state.get('pitch',0.0)
            health = state.get('health',1.0)
            alive = state.get('alive',True)
            light_level = state.get('light_level',0)
            player_state = np.array([x,y,z,yaw,pitch,health,1.0 if alive else 0.0,light_level], dtype=np.float32)

            if not hasattr(self, 'previous_health'):
                self.previous_health = health

            if self.prev_surrounding_blocks is not None:
                prev_hi = self.get_highest_value_index(self.prev_surrounding_blocks)
                new_hi = self.get_highest_value_index(surrounding_blocks_norm)
                reward += self.calculate_directional_reward(prev_hi, new_hi, action_name, mobs_norm)

            self.prev_surrounding_blocks = surrounding_blocks_norm.copy()

            if health < self.previous_health:
                reward += HEALTH_LOSS_PENALTY
                self.total_health_loss += HEALTH_LOSS_PENALTY
            if not alive:
                reward += DEATH_PENALTY
                self.total_death_penalty += DEATH_PENALTY
            if action_name == "attack 2":
                reward += 0.05
                self.total_attack_reward += 0.05
            if x < MIN_POS or x > MAX_POS or z < MIN_POS or z > MAX_POS:
                reward += 0.05
            if mobs_norm[1] > 0.0:
                reward += 4.0
                self.total_hit_reward += 4.0

            self.previous_health = health
            if self.steps == 0:
                reward = np.clip(reward, -0.5, 0.5)

            state_data = {
                'image': screenshot,
                'blocks': blocks_norm,
                'hand': hand_norm,
                'mobs': mobs_norm,
                'surrounding_blocks': surrounding_blocks_norm,
                'player_state': player_state
            }
        else:
            state_data = self._get_default_state()

        if self.steps % 50 == 0 and self.uri == "ws://localhost:8081":
            print(f"Step: {self.steps}, Reward: {reward:.2f}, Cumulative Reward: {self.cumulative_rewards:.2f}")

        self.steps += 1
        self.cumulative_rewards += reward
        terminated = self.steps >= self.max_episode_steps
        truncated = False
        info = {}
        return state_data, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        future = asyncio.run_coroutine_threadsafe(self._async_reset(), self.loop)
        try:
            return future.result(timeout=TIMEOUT_RESET_LONG)
        except Exception as e:
            raise e

    async def _async_reset(self):
        self.steps = 0
        self.reset_internal_stats()
        self.prev_break_progress = 0.0

        try:
            while not self.state_queue.empty():
                self.state_queue.get_nowait()

            if not self.connected or self.websocket is None:
                return self._get_default_state(), {}

            await self.send_action("reset 2")
            try:
                state = await asyncio.wait_for(self.state_queue.get(), timeout=TIMEOUT_RESET)
            except asyncio.TimeoutError:
                return self._get_default_state(), {}

            try:
                screenshot_task = asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)
                screenshot = await asyncio.wait_for(screenshot_task, timeout=TIMEOUT_STEP)
            except asyncio.TimeoutError:
                screenshot = DEFAULT_IMAGE

            if state is not None:
                blocks_norm = self.normalize_blocks(state.get('broken_blocks',[0,0,0,0]))
                hand_norm = self.normalize_hand(state)
                mobs_norm = self.normalize_mobs(state)
                surr_norm = self.flatten_surrounding_blocks(state)

                x = state.get('x',0.0)
                y = state.get('y',0.0)
                z = state.get('z',0.0)
                yaw = state.get('yaw',0.0)
                pitch = state.get('pitch',0.0)
                health = state.get('health',20.0)
                alive = state.get('alive',True)
                light_level = state.get('light_level',0)

                player_state = np.array([x,y,z,yaw,pitch,health,1.0 if alive else 0.0,light_level], dtype=np.float32)
                state_data = {
                    'image': screenshot,
                    'blocks': blocks_norm,
                    'hand': hand_norm,
                    'mobs': mobs_norm,
                    'surrounding_blocks': surr_norm,
                    'player_state': player_state
                }
                self.prev_sum_surrounding = surr_norm.sum()
            else:
                state_data = self._get_default_state()
                self.prev_sum_surrounding = 0.0

            return state_data, {}
        except:
            return self._get_default_state(), {}

    def capture_screenshot(self):
        with self.screenshot_lock:
            try:
                if self.sct is None:
                    self.sct = mss.mss()
                screenshot = self.sct.grab(self.minecraft_bounds)
                img = np.array(screenshot)[:, :, :3].transpose(2,0,1)/255.0
                return img.astype(np.float32)
            except:
                return DEFAULT_IMAGE

    def save_screenshot_if_needed(self, screenshot):
        if not self.save_screenshots:
            return
        try:
            # Create screenshots directory if needed
            os.makedirs(self.screenshot_dir, exist_ok=True)
            
            # Get URI suffix and timestamp
            uri_suffix = self.uri.split(":")[-1]
            timestamp = int(time.time()*1000)
            
            # Prepare image
            img_save = (screenshot.transpose(1,2,0)*255).astype(np.uint8)
            
            # Create filename with full path
            filename = f"step_{self.steps:06d}_port{uri_suffix}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")

    def render(self, mode='human'):
        pass

    def close(self):
        try:
            if self.connected and self.websocket:
                self.connected = False
                asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
                self.websocket = None
            with self.screenshot_lock:
                if self.sct:
                    try: self.sct.close()
                    except: pass
                self.sct = None
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.connection_thread and self.connection_thread.is_alive():
                self.connection_thread.join(timeout=1)
        except:
            pass

    def __del__(self):
        self.close()
