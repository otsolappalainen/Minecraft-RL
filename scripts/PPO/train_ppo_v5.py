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
from env_ppo_v5 import MinecraftEnv
import pygetwindow as gw
import fnmatch
import time
import mss
import asyncio
import websockets
import json
import math
from pathlib import Path


# System configuration
PARALLEL_ENVS = 8          # Number of parallel TRAINING environments. 1 additional environment will be used for evaluation. So total environments = PARALLEL_ENVS + 1
WINDOW_CROP_SIZE = 240     # Size of the captured image from each window. The model will currently train on half of this size.
MOVEMENT_DETECTION_THRESHOLD = 5       # Used in mapping windows to URIs. Lower values are more sensitive to movement, but might assign wrong URIs.
device = th.device("cuda" if th.cuda.is_available() else "cpu")     # Use GPU if available


# Filesystem configuration. Create uniqe subfolder for each run.
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path(__file__).parent.parent                    # Go up one level from the script location
MODELS_DIR = BASE_DIR / "models"
PPO_MODELS_DIR = MODELS_DIR / "ppo" / "v5"
LOG_DIR_BASE = BASE_DIR / "logs" / "tensorboard" / "ppo_v5"


# Replace the original paths
MODEL_PATH_PPO = str(PPO_MODELS_DIR)     # Convert to string for compatibility
LOG_DIR = str(LOG_DIR_BASE / f"run_{RUN_TIMESTAMP}")


# Create necessary directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PPO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


# Training configuration
TOTAL_TIMESTEPS = 5_000_000     # Total timesteps. LR and Entropy coefficient will be scheduled based on this.
LEARNING_RATE = 1e-5            # Initial learning rate
N_STEPS = 2048                  
BATCH_SIZE = 512
N_EPOCHS = 5
GAMMA = 0.99
EVAL_FREQ = 4096
EVAL_EPISODES = 1
SAVE_EVERY_STEPS = 10000
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
ENT_COEF = 0.01       # Initial entropy coefficient
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
USE_SDE = False
SDE_SAMPLE_FREQ = -1
TARGET_KL = 0.01
VERBOSE = 1
SEED = None


# Constants for schedulers
MIN_LR = 1e-7              # Minimum learning rate
MIN_ENT_COEF = 0.001       # Minimum entropy coefficient
DECAY_STEPS = 600_000      # Can be used instead of the total timesteps, if needed.



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

        # CNN for image processing (3x120x120 input)
        self.img_head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten()
        )

        # Calculate CNN output size
        with th.no_grad():
            sample_input = th.zeros(1, 3, 120, 120)
            cnn_output = self.img_head(sample_input)
            cnn_features = cnn_output.shape[1]


        self.image_fusion = nn.Sequential(
            nn.Linear(cnn_features, 512),
            nn.ReLU(inplace=True),
        )


        # Fusion network for all scalar inputs
        scalar_dim = (
            observation_space.spaces["blocks"].shape[0] +  # blocks
            observation_space.spaces["hand"].shape[0] +    # hand
            observation_space.spaces["target_block"].shape[0] +  # target block
            observation_space.spaces["player_state"].shape[0]  # player state
        )

        matrix_dim = (
            observation_space.spaces["surrounding_blocks"].shape[0] * 
            observation_space.spaces["surrounding_blocks"].shape[1] * 
            observation_space.spaces["surrounding_blocks"].shape[2]   # surrounding blocks
        )
        
        self.matrix_fusion = nn.Sequential(
            nn.Linear(matrix_dim, 128),
            nn.ReLU(inplace=True),
        )


        self.scalar_fusion = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(inplace=True),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 512, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations):
        # Process image through CNN and image fusion
        img_features = self.img_head(observations["image"])
        img_features = self.image_fusion(img_features)

        # Process surrounding blocks through matrix fusion
        matrix_features = self.matrix_fusion(
            observations["surrounding_blocks"].flatten(1)
        )

        # Process scalar inputs
        scalar_input = th.cat([
            observations["blocks"],
            observations["hand"],
            observations["target_block"],
            observations["player_state"]
        ], dim=1)
        scalar_features = self.scalar_fusion(scalar_input)

        # Combine all features
        combined = th.cat([
            img_features,  # 512 features from image_fusion
            matrix_features,  # 128 features from matrix_fusion
            scalar_features  # 128 features from scalar_fusion
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
        progress_remaining = 1.0 - (self.num_timesteps / DECAY_STEPS)
        
        # Update entropy coefficient
        new_ent_coef = get_scheduled_ent_coef(
            self.initial_ent_coef,
            progress_remaining
        )
        self.model.ent_coef = new_ent_coef

        # Log to tensorboard
        self.logger.record("train/ent_coef", new_ent_coef)
        
        return True
    


# Minecraft window detection and mapping to URIs. Very bad code, but works for now.
# Sends actions to each client to detect movement in the game view. And maps URIs to windows based on that.

def find_minecraft_windows():
    print("\n=== Starting Minecraft Window Detection ===")
    
    patterns = ["Minecraft*", "*1.21.3*", "*Singleplayer*"]
    windows = []
    seen_handles = set()

    # Get all windows
    for title in gw.getAllTitles():
        for pattern in patterns:
            if (fnmatch.fnmatch(title, pattern)):
                for window in gw.getWindowsWithTitle(title):
                    if window._hWnd not in seen_handles:
                        center_x = window.left + window.width // 2
                        center_y = window.top + window.height // 2
                        crop_size = 240
                        half = crop_size // 2
                        
                        window_info = {
                            "left": center_x - half,
                            "top": center_y - half,
                            "width": crop_size,
                            "height": crop_size,
                            "uri": None,
                            "movement_score": 0
                        }
                        windows.append(window_info)
                        seen_handles.add(window._hWnd)

    if len(windows) < (PARALLEL_ENVS + 1):
        raise ValueError(f"Found only {len(windows)} Minecraft windows, need 9")

    print(f"\nFound {len(windows)} Minecraft windows")
    sct = mss.mss()

    async def get_stable_screenshot(bounds, num_samples=1, delay=0.1):
        screens = []
        for _ in range(num_samples):
            screen = np.array(sct.grab(bounds))[:,:,:3]
            screens.append(screen)
            await asyncio.sleep(delay)
        return np.mean(screens, axis=0)

    async def test_uri(uri, window_idx, max_retries=2):
        for attempt in range(max_retries):
            try:
                # Get stable before screenshots
                bounds = {
                    "left": windows[window_idx]["left"],
                    "top": windows[window_idx]["top"],
                    "width": windows[window_idx]["width"],
                    "height": windows[window_idx]["height"]
                }
                before = await get_stable_screenshot(bounds)
                
                # Send turn command
                async with websockets.connect(uri) as ws:
                    await ws.send(json.dumps({"action": "reset 2"}))
                    await asyncio.sleep(0.3)  # Wait longer for movement
                
                # Get stable after screenshots
                after = await get_stable_screenshot(bounds)
                
                # Calculate movement score
                diff = np.mean(np.abs(after - before))
                print(f"Movement score for window {window_idx} (attempt {attempt+1}): {diff}")
                
                if diff > windows[window_idx]["movement_score"]:
                    windows[window_idx]["movement_score"] = diff
                
                if diff > 5:  # Found significant movement
                    return True
                    
            except Exception as e:
                print(f"Error testing {uri} on window {window_idx}: {str(e)}")
                await asyncio.sleep(0.5)
                
        return False

    async def map_windows():
        unmapped_uris = [f"ws://localhost:{8080+i}" for i in range(PARALLEL_ENVS + 1)]
        mapped_windows = set()
        
        while unmapped_uris and len(mapped_windows) < (PARALLEL_ENVS + 1):
            for uri in unmapped_uris[:]:
                print(f"\nTesting {uri}...")
                
                # Reset movement scores
                for w in windows:
                    w["movement_score"] = 0
                
                # Test all unmapped windows
                tasks = []
                for idx, window in enumerate(windows):
                    if idx not in mapped_windows:
                        tasks.append(test_uri(uri, idx))
                
                results = await asyncio.gather(*tasks)
                
                # Find window with highest movement
                available_windows = [i for i, w in enumerate(windows) 
                                  if i not in mapped_windows]
                if available_windows:
                    max_idx = max(available_windows,
                                key=lambda i: windows[i]["movement_score"])
                    
                    if windows[max_idx]["movement_score"] > 5:
                        windows[max_idx]["uri"] = uri
                        mapped_windows.add(max_idx)
                        unmapped_uris.remove(uri)
                        print(f"Mapped {uri} to window at ({windows[max_idx]['left']}, {windows[max_idx]['top']})")
                    else:
                        print(f"No clear movement detected for {uri}, will retry")
                        
                await asyncio.sleep(0.5)

    # Run mapping
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(map_windows())
    
    # Sort and verify
    sorted_windows = sorted(
        [w for w in windows if w["uri"] is not None],
        key=lambda w: int(w["uri"].split(":")[-1])
    )

    if len(sorted_windows) != (PARALLEL_ENVS + 1):
        raise ValueError(f"Could only map {len(sorted_windows)} windows")

    # Visual verification in reading order
    async def verify_windows():
        print("\nVisual verification - turning windows left to right, top to bottom...")
        visual_order = sorted(sorted_windows, key=lambda w: (w["top"], w["left"]))
        
        for window in visual_order:
            async with websockets.connect(window["uri"]) as ws:
                await ws.send(json.dumps({"action": "turn_left"}))
                print(f"Turned window at ({window['left']}, {window['top']})")
                await asyncio.sleep(0.5)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(verify_windows())
    loop.close()

    sct.close()
    
    print("\nWindow mapping complete:")
    print(f"Eval window (8080): ({sorted_windows[0]['left']}, {sorted_windows[0]['top']})")
    print("Training windows:")
    for i, window in enumerate(sorted_windows[1:], 1):
        print(f"Env {i} (808{i}): ({window['left']}, {window['top']})")

    return [
        {k: v for k, v in w.items() if k not in ['uri', 'movement_score']} 
        for w in sorted_windows
    ]


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

def main():
    # Initialize environments
    minecraft_windows = find_minecraft_windows()
    
    train_env_fns = [make_env(i, minecraft_windows=minecraft_windows) 
                     for i in range(PARALLEL_ENVS)]
    train_env = SubprocVecEnv(train_env_fns)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, 
                            clip_obs=10.0)
    
    eval_env_fn = [make_env(0, is_eval=True, minecraft_windows=minecraft_windows)]
    eval_env = SubprocVecEnv(eval_env_fn)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, 
                           clip_obs=10.0, training=False)

    # Model initialization options
    print("Select an option:")
    print("1. Load from previous model")
    print("2. Start from scratch")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        models = [f for f in os.listdir(MODEL_PATH_PPO) if f.endswith('.zip')]
        if not models:
            print("No models found. Starting from scratch.")
            model = create_new_model(train_env)
        else:
            print("\nAvailable models:")
            for idx, name in enumerate(models, 1):
                print(f"{idx}. {name}")
            try:
                idx = int(input("Select model number: ")) - 1
                model_path = os.path.join(MODEL_PATH_PPO, models[idx])
                model = load_ppo_model(model_path, train_env, device)
            except Exception as e:
                print(f"Error loading model: {e}")
                return
    else:
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
        EntCoefScheduleCallback(ENT_COEF)  # Moved to end, removed verbose flag
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
            recovery_path = os.path.join(
                MODEL_PATH_PPO,
                f"recovery_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )
            model.save(recovery_path)
            
            # Try to reload latest successful model
            latest_model = get_latest_model(MODEL_PATH_PPO)
            if latest_model and latest_model != recovery_path:
                try:
                    model = load_ppo_model(latest_model, train_env, device)
                    continue
                except Exception:
                    break
            break

    # Save final model
    final_path = os.path.join(
        MODEL_PATH_PPO, 
        f"final_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    )
    model.save(final_path)
    
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()