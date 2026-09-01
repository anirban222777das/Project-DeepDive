# Project DeepDive: Limitations & Known Issues

While the Recurrent Double DQN (DRQN) successfully solved the "Oxygen Bottleneck" in Seaquest, Reinforcement Learning is an inherently volatile field. The following are realistic, observed limitations of this specific implementation.

## 1. Hardware & Compute Bottlenecks
Despite the agent utilizing Apple Silicon (MPS) or Nvidia (CUDA) for blazing-fast matrix multiplication, the training speed is severely capped by the CPU.
- **Single-Threaded Emulation:** The Arcade Learning Environment (ALE) simulates the Atari 2600 CPU frame-by-frame on a single host CPU thread. The GPU spends the vast majority of its time sitting completely idle, waiting for the CPU to finish calculating the game's physics before it can process the next frame.
- **LSTM Unrolling:** Unlike standard Convolutional Neural Networks (CNNs) which can process batches of frames entirely in parallel, an LSTM *must* process time sequentially (Frame 1 -> Frame 2 -> Frame 3). This sequential requirement inherently slows down the backward pass (Backpropagation Through Time).

## 2. Exploration vs Exploitation Plateaus
The agent uses an **Epsilon-Greedy** exploration strategy. During training, it forces itself to take random actions (epsilon) a certain percentage of the time to discover new mechanics.
- **Deep Logical Mechanics:** Randomly mashing buttons is enough to accidentally discover that "shooting a shark = points". However, discovering complex logical sequences (e.g., "Wait for a diver to appear, align with it, collect it, wait 30 seconds, then surface without hitting a shark") is astronomically rare to achieve via pure randomness. 
- **The Epsilon Trap:** Once epsilon decays to a low number (e.g., 5%), the agent stops exploring and aggressively exploits whatever it currently knows. If it didn't stumble upon a deep mechanic early on, it will never learn it.

## 3. Catastrophic Forgetting
Neural Networks have a fixed capacity. In Reinforcement Learning, as the agent discovers new, highly rewarding strategies (like farming a specific spawn pattern of enemies), it fills its replay buffer with those specific memories.
- As older memories of basic survival (like dodging a specific obstacle) are pushed out of the buffer to make room for the new farming strategy, the network literally "forgets" how to dodge that obstacle. 
- This leads to sudden, catastrophic drops in performance during training, where an agent that was scoring 900 points suddenly drops back down to 200 points for a few dozen episodes until it relearns the basic mechanics.

## 4. Extreme Visual Fragility
The agent plays directly from raw pixel arrays. It does not actually understand what a "submarine" or a "shark" is. It only understands that a specific cluster of bright pixels moving horizontally correlates with danger.
- **Color Palettes:** If the game developer changed the color of the sharks from blue to green, the agent's performance would immediately drop to zero. It is entirely overfit to the exact visual rendering of the Atari 2600 emulator.
- **Flickering:** Atari games are famous for "sprite flickering" (due to hardware limitations, characters blink in and out of existence on alternating frames). If the agent happens to sample a sequence of frames where a shark is invisible due to flickering, it will crash directly into it.
