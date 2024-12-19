import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress some warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize
import multiprocessing
import logging
from env_ppo_240 import MinecraftEnv
import pygetwindow as gw
import fnmatch
import time
import mss
import asyncio
import websockets
import json
import math
from pathlib import Path
import msvcrt
import time
from typing import Optional
import tkinter as tk
from tkinter import filedialog


# System configuration
PARALLEL_ENVS = 11          # Number of parallel TRAINING environments. 1 additional environment will be used for evaluation. So total environments = PARALLEL_ENVS + 1
WINDOW_CROP_WIDTH = 240
WINDOW_CROP_HEIGHT = 120   # Size of the captured image from each window. The model will currently train on half of this size.
MOVEMENT_DETECTION_THRESHOLD = 5       # shouldnt be needed anymore.
device = th.device("cuda" if th.cuda.is_available() else "cpu")     # Use GPU if available


# Filesystem configuration. Create uniqe subfolder for each run.
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(__file__).parent.parent                    # Go up one level from the script location
MODELS_DIR = BASE_DIR / "models"
#PPO_MODELS_DIR = MODELS_DIR / "ppo" / "v5"

LOG_DIR_BASE = BASE_DIR / "logs" / "tensorboard" / "ppo_v5_240"
#PPO_MODELS_DIR = r"E:\PPO_BC_MODELS\models_ppo_v5"
PPO_MODELS_DIR = Path(r"E:\PPO_BC_MODELS\models_ppo_240")
MODEL_PATH_PPO = str(PPO_MODELS_DIR) 

# Replace the original paths
MODEL_PATH_PPO = str(PPO_MODELS_DIR)     # Convert to string for compatibility
LOG_DIR = str(LOG_DIR_BASE / f"run_{RUN_TIMESTAMP}")


# Create necessary directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PPO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


# Training configuration
TOTAL_TIMESTEPS = 5_000_000     # Total timesteps. LR and Entropy coefficient will be scheduled based on this.
LEARNING_RATE = 5e-5            # Initial learning rate
N_STEPS = 1024             
BATCH_SIZE = 128
N_EPOCHS = 10
GAMMA = 0.99
EVAL_FREQ = 8192
EVAL_EPISODES = 1
SAVE_EVERY_STEPS = 20000
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
ENT_COEF = 0.01       # Initial entropy coefficient
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
USE_SDE = False
SDE_SAMPLE_FREQ = -1
TARGET_KL = 0.02
VERBOSE = 1
SEED = None


# Constants for schedulers
MIN_LR = 5e-7              # Minimum learning rate
MIN_ENT_COEF = 0.001       # Minimum entropy coefficient
DECAY_STEPS = 1_000_000      # Can be used instead of the total timesteps, if needed.



def get_scheduled_lr(initial_lr, progress_remaining: float):
    """Linear decay based on remaining progress"""
    decay_rate = -5 * (1 - progress_remaining)
    return max(MIN_LR, initial_lr * np.exp(decay_rate))

def get_scheduled_ent_coef(initial_ent, progress_remaining: float):
    """Exponential decay based on remaining progress""" 
    decay_rate = -5 * (1 - progress_remaining)
    return max(MIN_ENT_COEF, initial_ent * np.exp(decay_rate))



#  Custom CNN + dense layer feature extractor for PPO. Combines image, scalar and matrix inputs.

#  Input Shapes:
#    - image: (B, 3, 120, 120) - RGB game view
#    - surrounding_blocks: (B, X, Y, Z) - 3D matrix of block types
#    - blocks: (B, N) - Current block states
#    - hand: (B, 5) - Item in hand
#    - target_block: (B, 3) - Target block info
#    - player_state: (B, 8) - Position, rotation, health etc.

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        
        self.img_head = nn.Sequential(
            # Initial layers
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),

            # First downsampling
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            
            # Second downsampling
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            
            # Final downsampling
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten()
        )


        # Calculate CNN output size
        with th.no_grad():
            sample_input = th.zeros(1, 3, 120, 240)
            cnn_output = self.img_head(sample_input)
            self.cnn_features = cnn_output.shape[1]

        self.image_fusion = nn.Sequential(
            nn.Linear(self.cnn_features, 256),
            nn.ReLU(inplace=True),
        )


        # Fusion network for all scalar inputs
        scalar_dim = (
            observation_space.spaces["blocks"].shape[0] +  # blocks
            observation_space.spaces["hand"].shape[0] +    # hand
            observation_space.spaces["mobs"].shape[0] +  # target block
            observation_space.spaces["player_state"].shape[0]  # player state
        )

        # Simplified surrounding blocks processing
        self.surrounding_fusion = nn.Sequential(
            nn.Linear(12, 128),  # 12 directional values -> 64 features
            nn.ReLU(inplace=True),
        )

        self.scalar_fusion = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(inplace=True),
        )

        # Final fusion (adjusted for new dimensions)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128, features_dim),  # Updated input size
            nn.ReLU(inplace=True),
        )

    def forward(self, observations):
        # Process image through CNN and image fusion
        img_features = self.img_head(observations["image"])
        img_features = self.image_fusion(img_features)

        # Process directional values
        surrounding_features = self.surrounding_fusion(
            observations["surrounding_blocks"]
        )

        # Process scalar inputs
        scalar_input = th.cat([
            observations["blocks"],
            observations["hand"],
            observations["mobs"],
            observations["player_state"]
        ], dim=1)
        scalar_features = self.scalar_fusion(scalar_input)

        # Combine all features
        combined = th.cat([
            img_features,
            surrounding_features,
            scalar_features
        ], dim=1)

        # Final fusion to output dimension
        return self.fusion(combined)

def create_new_model(env):
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[1024, 1024]
    )

    # Learning rate scheduler
    def scheduled_lr(step):
        return get_scheduled_lr(LEARNING_RATE, step)

    # Initial entropy coefficient will be scheduled by callback
    initial_ent_coef = ENT_COEF

    model = PPO(
        policy=ActorCriticPolicy,
        env=env,
        learning_rate=scheduled_lr,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        clip_range_vf=CLIP_RANGE_VF,
        ent_coef=initial_ent_coef,  # Initial value that will be scheduled
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        tensorboard_log=LOG_DIR,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=VERBOSE,
    )

    return model

# Simple callback for entropy coefficient scheduling
class EntCoefScheduleCallback(BaseCallback):
    def __init__(self, initial_ent_coef):
        super().__init__()
        self.initial_ent_coef = initial_ent_coef

    def _on_step(self) -> bool:
        # Calculate progress remaining (1.0 -> 0.0)
        progress_remaining = 1.0 - (self.num_timesteps / TOTAL_TIMESTEPS)
        
        # Update entropy coefficient
        new_ent_coef = get_scheduled_ent_coef(
            self.initial_ent_coef,
            progress_remaining
        )
        self.model.ent_coef = new_ent_coef

        # Log to tensorboard
        self.logger.record("train/ent_coef", new_ent_coef)
        
        return True
    


def attempt_window_connection(port, retries=2):
    """Try to connect to a window/server multiple times"""
    async def get_window_info(port):
        uri = f"ws://localhost:{port}"
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({"action": "monitor"}))
                response = await ws.recv()
                window_data = json.loads(response)
                
                center_x = window_data["x"] + window_data["width"] // 2
                center_y = window_data["y"] + window_data["height"] // 2
                half_width = WINDOW_CROP_WIDTH // 2
                half_height = WINDOW_CROP_HEIGHT // 2

                return {
                    "left": center_x - half_width,
                    "top": center_y - half_height,
                    "width": WINDOW_CROP_WIDTH,
                    "height": WINDOW_CROP_HEIGHT,
                    "port": port
                }
        except Exception as e:
            print(f"Failed to connect to {uri}: {e}")
            return None

    for i in range(retries):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(get_window_info(port))
            loop.close()
            if result:
                return result
            print(f"Retry {i+1}/{retries} for port {port}")
            time.sleep(2)
        except Exception as e:
            print(f"Error on try {i+1}: {e}")
    return None

def find_available_windows():
    """Find all available windows with retries"""
    print("\n=== Starting Minecraft Window Detection ===")
    windows = []
    
    # Try eval window first (port 8080)
    eval_window = attempt_window_connection(8080)
    if eval_window:
        windows.append(eval_window)
    
    # Try training windows
    for i in range(PARALLEL_ENVS):
        port = 8081 + i
        window = attempt_window_connection(port)
        if window:
            windows.append(window)
    
    print(f"\nFound {len(windows)} working windows")
    return windows

def make_env(rank, is_eval=False, minecraft_windows=None):
    def _init():
        try:
            if is_eval:
                uri = "ws://localhost:8080"
                window_bounds = minecraft_windows[0]
            else:
                uri = f"ws://localhost:{8081 + rank}"
                window_bounds = minecraft_windows[rank + 1]

            env = MinecraftEnv(uri=uri, window_bounds=window_bounds)
            return env
        except Exception as e:
            print(f"Error creating environment {rank}: {e}")
            raise
    return _init

def load_ppo_model(model_path, env, device):
    try:
        print(f"Loading model from {model_path}...")
        new_model = create_new_model(env)
        old_model = PPO.load(model_path, env=env, device=device)
        new_model.policy.load_state_dict(old_model.policy.state_dict())
        return new_model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def get_latest_model(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    if not models:
        return None
    return os.path.join(model_dir, 
                       max(models, key=lambda x: os.path.getctime(os.path.join(model_dir, x))))

class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq, save_path):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save_step = 0

    def _on_step(self):
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            path = os.path.join(self.save_path, 
                              f"model_step_{self.num_timesteps}.zip")
            self.model.save(path)
            self.last_save_step = self.num_timesteps
        return True

def get_models_list(model_dir: str) -> list[str]:
    """Get sorted list of model files"""
    models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    return sorted(models, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))

def input_with_timeout(prompt: str, timeout: int = 60) -> Optional[str]:
    """Get input with timeout, return None if no input"""
    print(prompt)
    print(f"Waiting {timeout} seconds for input...")
    
    response = []
    start_time = time.time()
    
    while True:
        if msvcrt.kbhit():
            char = msvcrt.getwche()
            if char == '\r':  # Enter key
                print()  # New line after input
                return ''.join(response)
            elif char == '\b':  # Backspace
                if response:
                    response.pop()
                    # Clear last character
                    print('\b \b', end='', flush=True)
            else:
                response.append(char)
        
        if time.time() - start_time > timeout:
            print("\nTimeout - using latest model")
            return None
            
        time.sleep(0.1)

def select_model(model_dir: str) -> str:
    """Let user select a model or default to latest"""
    models = get_models_list(model_dir)
    
    if not models:
        print("No models found in directory")
        return None
        
    # Print available models
    print("\nAvailable models:")
    for idx, model in enumerate(models, 1):
        print(f"{idx}. {model}")
    
    # Get user selection with timeout
    selection = input_with_timeout("\nEnter model number or press Enter for latest: ")
    
    if selection is None or selection.strip() == "":
        return os.path.join(model_dir, models[-1])
    
    try:
        idx = int(selection) - 1
        if 0 <= idx < len(models):
            return os.path.join(model_dir, models[idx])
        else:
            print("Invalid selection, using latest model")
            return os.path.join(model_dir, models[-1])
    except ValueError:
        print("Invalid input, using latest model")
        return os.path.join(model_dir, models[-1])

def select_training_mode(model_dir: str) -> tuple[str, bool, str]:
    """
    Ask user for training mode:
    1. Continue previous training
    2. Start new training
    3. Load weights from another model
    
    Returns:
    - model_path: Path to model to load from (or None)
    - new_training: Boolean indicating if this is new training
    - source_weights: Path to source weights for option 3 (or None)
    """
    print("\n=== Training Mode Selection ===")
    print("1. Continue previous training")
    print("2. Start new training from scratch")
    print("3. Load weights from another model (e.g., combat model)")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            # Continue previous training - use existing model selection
            model_path = select_model(model_dir)
            return model_path, False, None
            
        elif choice == "2":
            # Start fresh
            return None, True, None
            
        elif choice == "3":
            # Create hidden tkinter root window
            root = tk.Tk()
            root.withdraw()
            
            # Open file dialog
            print("\nPlease select the source model file (.zip)...")
            source_weights = filedialog.askopenfilename(
                title="Select Source Model",
                filetypes=[("ZIP files", "*.zip")],
                initialdir=os.path.dirname(model_dir)
            )
            
            if source_weights and os.path.exists(source_weights):
                return None, False, source_weights
            else:
                print("No valid model selected, defaulting to new training")
                return None, True, None
        
        print("Invalid choice, please enter 1, 2, or 3")

def load_source_weights(source_path: str, target_model: PPO, env, device) -> bool:
    """
    Manually load matching weights layer by layer from source model.
    """
    try:
        print(f"Loading source weights from: {source_path}")
        
        # Load source model directly without environment
        source_model = PPO.load(source_path, device=device)
        
        # Get state dictionaries
        source_state = source_model.policy.state_dict()
        target_state = target_model.policy.state_dict()
        
        # Track transfers
        transferred = []
        skipped = []
        
        # New state dict to build
        new_state = {}
        
        # Manually check and transfer each layer
        for key in target_state.keys():
            try:
                if key in source_state:
                    if source_state[key].shape == target_state[key].shape:
                        # Shapes match - copy weights
                        new_state[key] = source_state[key]
                        transferred.append(key)
                    else:
                        # Shapes don't match - keep original
                        new_state[key] = target_state[key]
                        skipped.append(f"{key}: {source_state[key].shape} vs {target_state[key].shape}")
                else:
                    # Key not in source - keep original
                    new_state[key] = target_state[key]
                    skipped.append(f"{key}: not found in source")
            except Exception as e:
                print(f"Error transferring layer {key}: {e}")
                new_state[key] = target_state[key]
                skipped.append(f"{key}: error")
        
        # Load the combined weights
        target_model.policy.load_state_dict(new_state, strict=False)
        
        # Print detailed transfer report
        print("\nTransfer Report:")
        print(f"Successfully transferred {len(transferred)} layers:")
        for key in transferred:
            print(f"✓ {key}")
        print(f"\nSkipped {len(skipped)} layers:")
        for msg in skipped:
            print(f"✗ {msg}")
            
        return len(transferred) > 0
        
    except Exception as e:
        print(f"Error during weight transfer: {e}")
        return False

def main_training_loop():
    """Main training loop with automatic restart on failure"""
    while True:
        try:
            # Find available windows
            minecraft_windows = find_available_windows()
            if len(minecraft_windows) < 2:  # Need at least eval + 1 training
                print("Not enough windows found. Need at least 2 windows.")
                return

            print(f"Starting training with {len(minecraft_windows)-1} training environments")
            
            # Adjust parallel envs based on available windows
            actual_parallel_envs = len(minecraft_windows) - 1
            
            # Initialize environments with available windows
            train_env_fns = [make_env(i, minecraft_windows=minecraft_windows) 
                           for i in range(actual_parallel_envs)]
            train_env = SubprocVecEnv(train_env_fns)
            train_env = VecMonitor(train_env)
            #train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, 
            #                       clip_obs=10.0)
            
            eval_env_fn = [make_env(0, is_eval=True, minecraft_windows=minecraft_windows)]
            eval_env = SubprocVecEnv(eval_env_fn)
            eval_env = VecMonitor(eval_env)
            #eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, 
            #                      clip_obs=10.0, training=False)

            # Load latest model or create new
            model_path, new_training, source_weights = select_training_mode(MODEL_PATH_PPO)

            if new_training:
                print("Starting new training from scratch")
                model = create_new_model(train_env)
            elif source_weights:
                print(f"Loading weights from: {source_weights}")
                model = create_new_model(train_env)
                if not load_source_weights(source_weights, model, train_env, device):
                    print("Weight transfer failed, starting new training instead")
            elif model_path:
                print(f"Continuing training from: {model_path}")
                model = load_ppo_model(model_path, train_env, device)
            else:
                print("No valid model selected, starting new training")
                model = create_new_model(train_env)

            # Setup callbacks
            callbacks = [
                SaveOnStepCallback(SAVE_EVERY_STEPS, MODEL_PATH_PPO),
                EvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=MODEL_PATH_PPO,
                    log_path=LOG_DIR,
                    eval_freq=EVAL_FREQ,
                    n_eval_episodes=EVAL_EPISODES,
                    deterministic=False
                ),
                EntCoefScheduleCallback(ENT_COEF)
            ]

            # Training loop
            timesteps_done = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
            while timesteps_done < TOTAL_TIMESTEPS:
                try:
                    remaining_timesteps = TOTAL_TIMESTEPS - timesteps_done
                    model.learn(
                        total_timesteps=remaining_timesteps,
                        callback=callbacks,
                        tb_log_name=f"ppo_training_{RUN_TIMESTAMP}",
                        reset_num_timesteps=False
                    )
                    timesteps_done = TOTAL_TIMESTEPS

                except Exception as e:
                    print(f"Training error: {e}")
                    # Save recovery model
                    recovery_path = os.path.join(
                        MODEL_PATH_PPO,
                        f"recovery_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    )
                    model.save(recovery_path)
                    raise  # Re-raise to trigger restart

            # Successful completion
            final_path = os.path.join(
                MODEL_PATH_PPO, 
                f"final_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )
            model.save(final_path)
            print("Training completed successfully!")
            break

        except Exception as e:
            print(f"\nTraining failed: {e}")
            print("Waiting 10 seconds before restart...")
            time.sleep(10)
            continue
        
        finally:
            try:
                train_env.close()
                eval_env.close()
            except:
                pass

if __name__ == "__main__":
    main_training_loop()