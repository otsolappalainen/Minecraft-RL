import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress some warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import numpy as np
import torch as th
import torch.nn as nn
# Changed import from stable_baselines3.PPO to sb3_contrib.RecurrentPPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize
import multiprocessing
import logging
from env_rec import MinecraftEnv
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
PARALLEL_ENVS = 12
WINDOW_CROP_WIDTH = 240
WINDOW_CROP_HEIGHT = 120
MOVEMENT_DETECTION_THRESHOLD = 5
device = th.device("cuda" if th.cuda.is_available() else "cpu")
FREEZE_CNN_WEIGHTS = False

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
LOG_DIR_BASE = BASE_DIR / "logs" / "tensorboard" / "recurrent_ppo"
PPO_MODELS_DIR = Path(r"E:\PPO_BC_MODELS\models_recurrent")
MODEL_PATH_PPO = str(PPO_MODELS_DIR)
LOG_DIR = str(LOG_DIR_BASE / f"run_{RUN_TIMESTAMP}")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PPO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 5_000_000
LEARNING_RATE = 1e-5
N_STEPS = 4096
BATCH_SIZE = 256
N_EPOCHS = 5
GAMMA = 0.99
EVAL_FREQ = 2000000
EVAL_EPISODES = 1
SAVE_EVERY_STEPS = 20000
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
ENT_COEF = 0.02
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
USE_SDE = False
SDE_SAMPLE_FREQ = -1
TARGET_KL = 0.05
VERBOSE = 1
SEED = None

MIN_LR = 1e-6
MIN_ENT_COEF = 0.001
DECAY_STEPS = 1_000_000

def get_scheduled_lr(initial_lr, progress_remaining: float):
    decay_rate = -5 * (1 - progress_remaining)
    return max(MIN_LR, initial_lr * np.exp(decay_rate))

def get_scheduled_ent_coef(initial_ent, progress_remaining: float):
    decay_rate = -5 * (1 - progress_remaining)
    return max(MIN_ENT_COEF, initial_ent * np.exp(decay_rate))

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        self.img_head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten()
        )

        with th.no_grad():
            sample_input = th.zeros(1, 3, 120, 240)
            cnn_output = self.img_head(sample_input)
            self.cnn_features = cnn_output.shape[1]

        self.image_fusion = nn.Sequential(
            nn.Linear(self.cnn_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        scalar_dim = (
            observation_space.spaces["hand"].shape[0] +
            observation_space.spaces["mobs"].shape[0] +
            observation_space.spaces["player_state"].shape[0]
        )

        self.surrounding_fusion = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(inplace=True),
        )

        self.scalar_fusion = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations):
        img_features = self.img_head(observations["image"])
        img_features = self.image_fusion(img_features)

        surrounding_features = self.surrounding_fusion(
            observations["surrounding_blocks"]
        )

        scalar_input = th.cat([
            observations["hand"],
            observations["mobs"],
            observations["player_state"]
        ], dim=1)
        scalar_features = self.scalar_fusion(scalar_input)

        combined = th.cat([
            img_features,
            surrounding_features,
            scalar_features
        ], dim=1)

        return self.fusion(combined)


def create_new_model(env):
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[1024, 1024],
        lstm_hidden_size=512,  # This defines the LSTM memory size
        n_lstm_layers=1        # (optional) number of LSTM layers
    )

    def scheduled_lr(step):
        return get_scheduled_lr(LEARNING_RATE, step)

    initial_ent_coef = ENT_COEF

    # Key Change: Use RecurrentPPO and RecurrentActorCriticPolicy
    model = RecurrentPPO(
        policy=RecurrentActorCriticPolicy,
        env=env,
        learning_rate=scheduled_lr,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        clip_range_vf=CLIP_RANGE_VF,
        ent_coef=initial_ent_coef,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        tensorboard_log=LOG_DIR,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=VERBOSE,
        target_kl=TARGET_KL
    )

    if FREEZE_CNN_WEIGHTS:
        for param in model.policy.features_extractor.img_head.parameters():
            param.requires_grad = False
        print("CNN weights frozen")

    return model


class EntCoefScheduleCallback(BaseCallback):
    def __init__(self, initial_ent_coef):
        super().__init__()
        self.initial_ent_coef = initial_ent_coef

    def _on_step(self) -> bool:
        progress_remaining = 1.0 - (self.num_timesteps / TOTAL_TIMESTEPS)
        new_ent_coef = get_scheduled_ent_coef(self.initial_ent_coef, progress_remaining)
        self.model.ent_coef = new_ent_coef
        self.logger.record("train/ent_coef", new_ent_coef)
        return True


def attempt_window_connection(port, retries=2):
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
    print("\n=== Starting Minecraft Window Detection ===")
    windows = []
    for i in range(PARALLEL_ENVS):
        port = 8080 + i
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
                uri = f"ws://localhost:{8080 + rank}"
                window_bounds = minecraft_windows[rank]

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
        # Load PPO weights directly into RecurrentPPO if compatible
        # If not, you may need to do manual layer transfers
        old_model = RecurrentPPO.load(model_path, env=env, device=device)
        new_model.policy.load_state_dict(old_model.policy.state_dict())
        
        if FREEZE_CNN_WEIGHTS:
            for param in new_model.policy.features_extractor.img_head.parameters():
                param.requires_grad = False

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
    models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    return sorted(models, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))

def input_with_timeout(prompt: str, timeout: int = 60) -> Optional[str]:
    print(prompt)
    print(f"Waiting {timeout} seconds for input...")

    response = []
    start_time = time.time()

    while True:
        if msvcrt.kbhit():
            char = msvcrt.getwche()
            if char == '\r':
                print()
                return ''.join(response)
            elif char == '\b':
                if response:
                    response.pop()
                    print('\b \b', end='', flush=True)
            else:
                response.append(char)

        if time.time() - start_time > timeout:
            print("\nTimeout - using latest model")
            return None

        time.sleep(0.1)

def select_model(model_dir: str) -> str:
    models = get_models_list(model_dir)
    if not models:
        print("No models found in directory")
        return None

    print("\nAvailable models:")
    for idx, model in enumerate(models, 1):
        print(f"{idx}. {model}")

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
    print("\n=== Training Mode Selection ===")
    print("1. Continue previous training")
    print("2. Start new training from scratch")
    print("3. Load weights from another model (e.g., combat model)")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            model_path = select_model(model_dir)
            return model_path, False, None
        elif choice == "2":
            return None, True, None
        elif choice == "3":
            root = tk.Tk()
            root.withdraw()
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


def load_source_weights(source_path: str, target_model: RecurrentPPO, env, device) -> bool:
    import traceback  # Add import
    
    try:
        print(f"Loading source weights from: {source_path}")
        try:
            from stable_baselines3 import PPO
            source_model = PPO.load(source_path, device=device)
            print("Detected source as PPO model")
            source_type = "PPO"
        except:
            source_model = RecurrentPPO.load(source_path, device=device)
            print("Detected source as RecurrentPPO model") 
            source_type = "RecurrentPPO"

        # Get state dictionaries
        source_state = source_model.policy.state_dict()
        target_state = target_model.policy.state_dict()
        new_state = dict(target_state)

        # Track transfers
        transferred = []
        skipped = []

        # Layer mapping
        layer_groups = {
            'cnn': ['features_extractor.img_head'],
            'features': [
                'features_extractor.image_fusion',
                'features_extractor.surrounding_fusion', 
                'features_extractor.scalar_fusion',
                'features_extractor.fusion'
            ],
            'policy': ['mlp_extractor', 'action_net', 'value_net'],
            'lstm': ['lstm']
        }

        # Process each target layer
        for key in target_state.keys():
            try:
                # Skip LSTM for PPO source
                if source_type == "PPO" and any(x in key for x in layer_groups['lstm']):
                    skipped.append(f"{key} (LSTM layer skipped for PPO)")
                    continue

                # Get layer type
                layer_type = 'other'
                for group, patterns in layer_groups.items():
                    if any(pattern in key for pattern in patterns):
                        layer_type = group
                        break

                # Try direct transfer first
                if key in source_state and source_state[key].shape == target_state[key].shape:
                    new_state[key] = source_state[key]
                    transferred.append(f"{key} ({layer_type})")
                    continue

                # Try policy layer name variants
                if layer_type == 'policy':
                    old_style = key.replace('mlp_extractor', 'pi')
                    if old_style in source_state and source_state[old_style].shape == target_state[key].shape:
                        new_state[key] = source_state[old_style]
                        transferred.append(f"{key} (policy remapped)")
                        continue

                skipped.append(f"{key} (shape mismatch or not found)")

            except Exception as e:
                print(f"Error on layer {key}: {e}")
                skipped.append(f"{key} (error)")

        # Print summary by category
        print("\nTransfer Summary:")
        for group in layer_groups.keys():
            group_transfers = [t for t in transferred if group in t.lower()]
            if group_transfers:
                print(f"\n{group.upper()} layers transferred:")
                for t in group_transfers:
                    print(f"✓ {t}")

        print(f"\nSkipped {len(skipped)} layers:")
        for s in skipped:
            print(f"✗ {s}")

        # Load the filtered state dict
        target_model.policy.load_state_dict(new_state, strict=False)
        
        if FREEZE_CNN_WEIGHTS:
            for param in target_model.policy.features_extractor.img_head.parameters():
                param.requires_grad = False
            print("\nCNN weights frozen")

        return len(transferred) > 0

    except Exception as e:
        print(f"Error during weight transfer: {e}")
        traceback.print_exc()
        return False
    

def main_training_loop():
    while True:
        try:
            minecraft_windows = find_available_windows()
            if len(minecraft_windows) < 1:
                print("Not enough windows found. Need at least 2 windows.")
                return

            print(f"Starting training with {len(minecraft_windows)-1} training environments")
            
            actual_parallel_envs = len(minecraft_windows)
            train_env_fns = [make_env(i, minecraft_windows=minecraft_windows) for i in range(actual_parallel_envs)]
            train_env = SubprocVecEnv(train_env_fns)
            train_env = VecMonitor(train_env)

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

            callbacks = [
                SaveOnStepCallback(SAVE_EVERY_STEPS, MODEL_PATH_PPO),
                EntCoefScheduleCallback(ENT_COEF)
            ]

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
                    recovery_path = os.path.join(
                        MODEL_PATH_PPO,
                        f"recovery_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    )
                    model.save(recovery_path)
                    raise

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
            except:
                pass

if __name__ == "__main__":
    main_training_loop()
