# Seaquest RL Agent

This repository contains a Reinforcement Learning agent built to play the classic Atari 2600 game **Seaquest** from raw pixel observations. 

The agent is trained using a Convolutional Neural Network (CNN) combined with the **Double Deep Q-Network (Double DQN)** algorithm. It interfaces with the Arcade Learning Environment (ALE) via Gymnasium.

## Architecture and Methods

### Observation Processing
The agent learns directly from raw visual input without any hand-engineered game state features.
- **Grayscale Conversion & Resizing**: The raw RGB frames are converted to grayscale and downsampled to 84x84 pixels.
- **Frame Stacking**: To provide the agent with a sense of motion and velocity, the last 4 frames are stacked together, creating an input volume of `(4, 84, 84)`.
- **Normalization**: Pixel values (uint8) are scaled to `[0, 1]` inside the network for stability.

### Neural Network (CNN)
The Q-network follows the standard architecture introduced by DeepMind:
- **Conv1**: 32 filters, 8x8 kernel, stride 4
- **Conv2**: 64 filters, 4x4 kernel, stride 2
- **Conv3**: 64 filters, 3x3 kernel, stride 1
- **Fully Connected**: 512 units
- **Output**: 18 linear units corresponding to the 18 possible Atari actions.

### Double DQN & Experience Replay
We utilize **Double DQN (DDQN)** to mitigate the systematic overestimation of Q-values common in vanilla DQN. Action selection is decoupled from action evaluation using a frozen target network.

A **Replay Buffer** of capacity 1,000,000 stores recent transitions `(state, action, reward, next_state, done)`. Mini-batches are uniformly sampled from this buffer to break temporal correlations and stabilize the training gradients.

## Results and Current Capabilities

The agent was trained for **1,000,000 environment steps** using an epsilon-greedy exploration strategy. 

**Achievements:**
- The agent quickly learns basic survival: it successfully avoids enemies (sharks and submarines) and learns to shoot them to accumulate score.
- The average reward plateaus around **300-350 points**, significantly outperforming an untrained random agent.

**Limitations (The Oxygen Bottleneck):**
Seaquest requires the player to manage oxygen. To refill oxygen, the submarine must surface—but surfacing without having collected at least one diver causes instant death. 
Our rigorous evaluation shows that the current agent struggles with this long-horizon temporal credit assignment. It frequently learns to collect a diver and surface *once* per life, but fails to maintain the "collect diver -> surface" loop on the second oxygen cycle, consistently suffocating.

## Installation

We recommend using Conda to manage the environment:

```bash
conda create -n aimltrain python=3.10
conda activate aimltrain
pip install -r requirements.txt
```

### ROM Setup
To comply with copyright laws, the Atari ROMs are **not distributed** in this repository. 
You must legally obtain the Seaquest ROM and import it into ALE. Typically, this is done using the AutoROM utility (which downloads ROMs for academic research):
```bash
AutoROM --accept-license
```

## Usage

### Training
To train the Double DQN agent from scratch:
```bash
python -m src.train \
    --algorithm double-dqn \
    --steps 1000000 \
    --batch-size 32 \
    --log-dir logs/training \
    --checkpoint-dir models/checkpoints \
    --device mps
```
*(Note: Use `--device cuda` or `--device cpu` depending on your hardware. Apple Silicon MPS is fully supported.)*

### Evaluation
To evaluate a trained checkpoint:
```bash
python -m src.evaluate \
    --checkpoint models/checkpoints/dqn_step_1000000.pt \
    --episodes 20 \
    --track-oxygen
```

### Watching the Agent
To visually render the agent playing the game:
```bash
python -m src.watch_agent \
    --checkpoint models/checkpoints/dqn_step_1000000.pt \
    --episodes 3
```

### Testing
Run the deterministic unit test suite:
```bash
PYTHONPATH=. pytest tests/
```

## Future Work
To solve the long-horizon oxygen management bottleneck, future extensions could include:
- **Recurrent Memory (DRQN)**: Using LSTMs to allow the agent to remember if it has collected a diver outside of the 4-frame stack window.
- **Prioritized Experience Replay**: To sample rare but critical events (like suffocating or successfully surfacing) more frequently.
