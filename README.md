# Minecraft AI Training System

A deep reinforcement learning system that trains an AI agent to play Minecraft using PPO (Proximal Policy Optimization). The system uses parallel environments and WebSocket communication between a custom Minecraft mod and Python training scripts.

## System Overview

### Components
1. **Minecraft Mod** ([minecraft_client](./minecraft_client))
   - WebSocket server for receiving commands
   - Action execution in game
   - State observation and transmission
   - Block breaking event handling
   - Player state management

2. **Training System** ([PPO](./scripts/PPO))
   - PPO implementation
   - Parallel environment management
   - Screenshot processing
   - State observation handling
   - Model training and evaluation

## Requirements

### Minecraft Setup
- Minecraft 1.21.3
- Fabric Mod Loader
- Java JDK 21+

### Key Dependencies
- PyTorch 2.0.1+cu117 (CUDA 11.7)
- Stable-Baselines3 2.4.0
- Gymnasium 1.0.0
- OpenCV 4.10.0
- Websockets 14.1

## Quick Start

1. **Setup Minecraft**
   - Install Minecraft with Fabric
   - Build the fabric example mod with this client code ([minecraft_client](./minecraft_client))
   - Launch multiple Minecraft instances (8 training + 1 eval)

2. **Setup Python**

        python -m venv .venv
        .venv\Scripts\activate
        pip install -r requirements.txt

3. **Start Training**

        cd scripts/PPO
        python train_ppo_v5.py

4. **Monitor Progress**

        tensorboard --logdir=./logs/tensorboard/ppo_v5

## Training Process

1. **Environment Setup**
   - Multiple Minecraft clients for paraller training
   - Each runs mod with WebSocket server to receive actions and return observations

2. **Training Loop**
   - Agent receives state (screenshots, block data, player data)
   - PPO network selects actions
   - Actions executed in Minecraft
   - Rewards calculated from mining/exploration
   - Network updated with experiences

3. **State Observation**
   - Game screenshots (240x240 RGB, downscaled to 120x120 RGB for the training)
   - Block information (13x13x4 Matrix of surrounding block types)
   - Player position and orientation
   - Mining progress, tools, target block and broken block type

4. **Action Space**
   - Movement (forward, backward, strafe)
   - Looking (up, down, left, right)
   - Interactive actions
   - Item management



## Acknowledgments
   - Fabric
   - Stable Baselines3
   - Gymnasium