# Project DeepDive: System Architecture

This document provides a deep, technical breakdown of the Recurrent Double Deep Q-Network (DRQN) built for this project.

## 1. Observation Preprocessing pipeline
Atari 2600 environments output raw RGB arrays at 210x160 resolution. Feeding this directly into a neural network is computationally expensive and noisy.
- **Grayscaling:** Color information is largely irrelevant to the core logic of Seaquest. The frames are converted to grayscale to reduce the channel dimension from 3 to 1.
- **Downsampling:** The frames are cropped and resized using bilinear interpolation to an 84x84 square.
- **Normalization:** Pixel values (0-255) are divided by 255.0 to scale them into the `[0, 1]` range, which stabilizes neural network gradients.

## 2. The Recurrent Neural Network (DRQN)
The core "brain" of the agent (`src/recurrent_network.py`) is divided into three distinct phases: **Perception**, **Memory**, and **Decision**.

### Phase A: Perception (The CNN)
The agent "sees" the game using a Convolutional Neural Network (CNN) based on the original DeepMind architecture, but optimized to feed into a recurrent layer.
- **Conv Layer 1:** 32 filters, 8x8 kernel, stride of 4. (Extracts broad shapes like the submarine and enemies).
- **Conv Layer 2:** 64 filters, 4x4 kernel, stride of 2. (Extracts finer details).
- **Conv Layer 3:** 64 filters, 3x3 kernel, stride of 1. (Extracts highly localized features).
- *Activation:* ReLU is applied after every convolutional layer.

### Phase B: Memory (The LSTM)
Standard DQNs take a stack of the last 4 frames to infer velocity. This DRQN removes the 4-frame stack bottleneck and instead uses a **Long Short-Term Memory (LSTM)** cell.
- The flattened output of the CNN is fed into an LSTM with a **512-dimensional hidden state**.
- The LSTM acts as the agent's working memory. It allows the agent to remember events that happened hundreds of frames ago (e.g., "I collected a diver 5 seconds ago, I must hold onto this memory until my oxygen gets low, then surface").

### Phase C: Decision (The Action Head)
The 512-dimensional output from the LSTM is passed through a fully connected linear layer.
- **Output:** An array of 18 values, representing the predicted Q-value (expected future reward) for each of the 18 possible Atari joystick actions.

---

## 3. Step-by-Step Simulation (Pixel to Action)
To understand exactly how the math works under the hood, here is a simulation of the tensor transformations for a single frame:

**Step 1: Raw Pixels (The Input)**
- The Atari emulator outputs an image shaped `[210, 160, 3]` (Height, Width, RGB Channels).
- The preprocessor crops it, turns it gray, and scales it, resulting in a single tensor of shape `[1, 84, 84]` representing normalized pixel intensities (e.g., `0.0` for black water, `1.0` for a white pixel of the submarine).

**Step 2: The CNN (Feature Extraction)**
- The `[1, 84, 84]` image is fed into Conv1. An 8x8 filter slides across the pixels taking dot products, reducing the shape to `[32, 20, 20]`.
- It passes through Conv2, reducing to `[64, 9, 9]`.
- It passes through Conv3, reducing to `[64, 7, 7]`.
- We now have 64 highly abstract "feature maps" of size 7x7. These maps represent learned concepts (e.g., Filter #12 might light up only when a shark is present).

**Step 3: Flattening**
- The `[64, 7, 7]` tensor is flattened into a 1-dimensional array of `3,136` numbers (`64 * 7 * 7`).

**Step 4: The LSTM Memory Update**
- The array of `3,136` numbers is fed into the LSTM layer. 
- *Crucially*, the LSTM also receives its **Previous Hidden State** (an array of `512` numbers generated from the *last* frame).
- The LSTM performs complex gate math (Forget Gate, Input Gate, Output Gate) to combine the new `3,136` numbers with the old `512` numbers, generating a brand new **Current Hidden State** of size `512`. 
- This `512` array now contains information about what is happening *now*, plus what happened *seconds ago* (like collecting a diver).

**Step 5: The Action Head**
- The `512` hidden state array is passed through a linear layer, which multiplies it against a learned weight matrix to output exactly `18` numbers.
- These 18 numbers are the Q-Values. For example:
  - Action 0 (NOOP): `14.2`
  - Action 1 (FIRE): `15.1`
  - Action 7 (RIGHTFIRE): `84.9`

**Step 6: The Decision**
- The agent takes the `argmax` (the highest number) of the 18 values. Since Action 7 has the highest expected reward (`84.9`), the submarine shoots a torpedo to the right!

---

## 4. Sequence-Aware Experience Replay
A standard DQN replay buffer stores isolated, randomized transitions `(State, Action, Reward, Next_State)`. 
Because our agent has an LSTM, it must be trained on **continuous sequences of time**, otherwise the LSTM cannot learn how time flows.
- Our `RecurrentReplayBuffer` stores individual frames but samples them in **contiguous sequences of length 8**.
- **Boundary Protection:** The buffer mathematically ensures that a sampled sequence never crosses the boundary of a "Game Over". (i.e., Frame 7 cannot be from Game 1, and Frame 8 from Game 2).

## 4. Double Q-Learning Math
Standard Q-Learning suffers from "maximization bias"—it overestimates how good certain actions are. I solved this using the **Double DQN** architecture.
- **Online Network:** Actively plays the game and is updated via gradient descent on every step.
- **Target Network:** A frozen copy of the Online Network. It provides stable "target" values for the math.
- **The Update:** The Online Network decides *which* action is best, but the Target Network evaluates *how much* that action is actually worth. Every 1,000 steps, the Target Network copies the weights from the Online Network.
