# 🌊 Project DeepDive: Autonomous Atari Seaquest Agent

**Project DeepDive** is a deep reinforcement learning framework that trains an autonomous AI agent to master the classic 1983 Atari 2600 game, *Seaquest*, using only raw pixel inputs and in-game rewards. 

In Seaquest, players must pilot a submarine to shoot enemies, dodge obstacles, rescue divers, and most importantly, surface frequently to replenish a strictly limited oxygen supply. This multi-layered objective presents a notoriously difficult challenge for standard AI agents, known as the "Long-Horizon Temporal Credit Assignment" problem—specifically, the agent must remember to surface for oxygen long after rescuing a diver.

This repository documents the architectural journey and final neural network design used to solve this exact problem.

## 🎮 Gameplay Demo

Check out the trained agent in action!

![Gameplay Demo](screenrecordings/gameplay.gif)

---

## The Breakthrough: Memory Solves the Oxygen Bottleneck

Through iterative development, the agent evolved from a standard Convolutional Neural Network (CNN) using Double DQN, to a sophisticated **Recurrent Double DQN (DRQN)** featuring Long Short-Term Memory (LSTM) to maintain an internal understanding of time. It interfaces with the Arcade Learning Environment (ALE) via Gymnasium.

## The Evolution of the Agent

### 1. Untrained Baseline
Without training, a random agent acts erratically, failing to avoid enemies or manage oxygen. Average score: **0-20 points**.

### 2. Double DQN (The "Oxygen Bottleneck")
I successfully trained a standard **Double DQN** (CNN-only) for 1,000,000 steps. 
- **Achievements:** It learned to shoot enemies and dodge obstacles, averaging **300-350 points**.
- **The Fatal Flaw:** The agent only "saw" a stack of the last 4 frames (a fraction of a second). To survive in Seaquest, you must collect a diver and then surface for oxygen. Because it had no long-term memory, it would collect a diver, fight sharks for 10 seconds, and completely forget it had a diver. It would try to surface for air and die instantly. It **never** survived the second oxygen cycle.

### 3. Recurrent Double DQN (The Solution)
To make the agent smarter, I implemented a **Recurrent Double DQN (DRQN)**. By injecting an LSTM layer after the CNN, the agent gained an internal "hidden state" that persists over time. 
- **The Result:** The agent successfully learned to remember when it collected a diver! During a rigorous 20-episode evaluation, it reached low oxygen 161 times, and successfully resurfaced 81 times. 
- **Score:** Because it survived multiple oxygen cycles, its average score skyrocketed to **516 points**, with peaks hitting **900 points**.

![DRQN vs DDQN Reward Comparison](assets/drqn_vs_ddqn_reward.png)

## 🧠 Architecture Details

For a deep, technical breakdown of the Recurrent Neural Network architecture, the LSTM memory mechanisms, and the Double Q-Learning Bellman math, please refer to the full **[System Architecture Documentation](architecture.md)**.

## ⚠️ Limitations & Known Issues

Reinforcement Learning is an inherently volatile field. The agent suffers from catastrophic forgetting, hardware bottlenecks, and extreme visual fragility. For a highly detailed and realistic explanation of this model's limitations, please read the **[Limitations & Known Issues](limitations.md)** document.

## Installation & Setup (For a Fresh Clone)

Follow these exact steps if you have just cloned the repository and want to run the pre-trained smart agent.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/seaquest-rl.git
cd seaquest-rl
```

### 2. Set Up the Python Environment
I highly recommend using Conda to keep dependencies isolated:
```bash
conda create -n aimltrain python=3.10
conda activate aimltrain
```

### 3. Install Dependencies
Install PyTorch, Gymnasium, the Atari emulator (Arcade Learning Environment), and other requirements:
```bash
pip install -r requirements.txt
```

### 4. Install the Atari ROMs (Emulator Setup)
To comply with copyright law, the actual Atari game files (ROMs) are **not included** in this GitHub repository. You must legally download them using the `AutoROM` tool (which was installed in the previous step):
```bash
AutoROM --accept-license
```
*This command will automatically download the required Seaquest emulator files into your python environment.*

---

## Usage

### Watching the Pre-Trained Smart Agent
You don't need to train the agent yourself! I have provided the fully trained, lightweight DRQN brain in the `models/` folder. 

To watch the agent play the game with its LSTM memory active:
```bash
python -m src.watch_recurrent_agent \
    --checkpoint models/drqn_final.pt \
    --episodes 3
```

### Evaluating the Agent (No Rendering)
To rigorously evaluate the agent's oxygen management metrics silently across 20 episodes:
```bash
python -m src.evaluate_recurrent \
    --checkpoint models/drqn_final.pt \
    --episodes 20 \
    --track-oxygen
```

### Training from Scratch
If you want to train the memory-enabled agent yourself from scratch (Warning: training LSTMs sequentially takes longer than standard CNNs. Expect 3-5 hours on an Apple Silicon M-series chip):
```bash
python -m src.train_recurrent \
    --steps 1000000 \
    --batch-size 32 \
    --log-dir logs/training_drqn \
    --checkpoint-dir models/checkpoints_drqn \
    --device mps
```
*(Note: Change `--device mps` to `--device cuda` for Nvidia GPUs or `--device cpu` if you do not have a dedicated GPU.)*

## Testing
Run the deterministic unit test suite to verify the recurrent architecture and sequence buffers:
```bash
PYTHONPATH=. pytest tests/
```
