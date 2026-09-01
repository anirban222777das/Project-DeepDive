import os
import csv
import json
import random
import torch
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque

from src.config import parse_args, TrainingConfig
from src.preprocessing import AtariPreprocessor
from src.agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.dqn import DQN
from src.evaluate import evaluate

def calculate_epsilon(step: int, config: TrainingConfig) -> float:
    """Linear decay of epsilon."""
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end
    
    decay_rate = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay_steps
    return config.epsilon_start - (step * decay_rate)

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def save_checkpoint(path: str, dqn: DQN, step: int, episode: int, epsilon: float, learning_updates: int, config: TrainingConfig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'online_network_state_dict': dqn.online_network.state_dict(),
        'target_network_state_dict': dqn.target_network.state_dict(),
        'optimizer_state_dict': dqn.optimizer.state_dict(),
        'step': step,
        'episode': episode,
        'epsilon': epsilon,
        'learning_updates': learning_updates,
        'algorithm': config.algorithm,
        'rng_states': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()
        },
        # Save as dict for easier cross-compatibility
        'config': vars(config) 
    }
    torch.save(checkpoint, path)

def log_to_csv(filepath: str, headers: list, data: list):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

def train(config: TrainingConfig):
    set_seeds(config.seed)
    
    # Initialize Core Components
    env = gym.make("ALE/Seaquest-v5")
    preprocessor = AtariPreprocessor()
    
    # We pass the action count dynamically from the environment
    action_count = env.action_space.n
    
    dqn = DQN(
        action_count=action_count, 
        device=config.device, 
        gamma=config.gamma, 
        learning_rate=config.learning_rate,
        double_dqn=(config.algorithm == "double-dqn")
    )
    agent = DQNAgent(action_count=action_count, model=dqn.online_network, device=dqn.device)
    buffer = ReplayBuffer(capacity=config.replay_capacity)
    
    # Initialize Counters & Logging
    global_step = 0
    learning_updates = 0
    episodes_completed = 0
    
    rolling_rewards = deque(maxlen=100)
    rolling_lengths = deque(maxlen=100)
    
    training_log_path = os.path.join(config.log_dir, "training_metrics.csv")
    eval_log_path = os.path.join(config.log_dir, "evaluation_metrics.csv")
    
    print(f"--- Starting DQN Training on {dqn.device} ---")
    print(f"Total Steps: {config.total_steps}")
    
    if config.resume:
        print(f"\n--- Resuming from Checkpoint: {config.resume} ---")
        checkpoint = torch.load(config.resume, map_location=dqn.device, weights_only=False)
        
        # Verify algorithm matches to prevent cross-contamination
        if 'algorithm' in checkpoint and checkpoint['algorithm'] != config.algorithm:
            raise ValueError(f"Algorithm mismatch: Checkpoint is '{checkpoint['algorithm']}', but config is '{config.algorithm}'.")
            
        dqn.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        dqn.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        global_step = checkpoint['step']
        episodes_completed = checkpoint['episode']
        
        # Restore learning updates
        if 'learning_updates' in checkpoint:
            learning_updates = checkpoint['learning_updates']
        else:
            if global_step >= config.learning_starts:
                learning_updates = (global_step - config.learning_starts) // config.train_frequency
            else:
                learning_updates = 0
                
        if 'rng_states' in checkpoint:
            random.setstate(checkpoint['rng_states']['python'])
            np.random.set_state(checkpoint['rng_states']['numpy'])
            torch.set_rng_state(checkpoint['rng_states']['torch'])
            
        print(f"Resumed at Step: {global_step} | Episodes: {episodes_completed} | Updates: {learning_updates}")
        print("Note: Replay buffer is starting empty to avoid bloated checkpoint files.")
        
    else:
        # Baseline Pre-training Evaluation
        print("\n--- Baseline Pre-Training Evaluation ---")
        base_mean, base_std, base_len = evaluate(env, agent, preprocessor, config.evaluation_episodes)
        print(f"Mean Reward: {base_mean:.2f} ± {base_std:.2f} | Mean Length: {base_len:.1f}")
        log_to_csv(eval_log_path, ["step", "mean_reward", "std_reward", "mean_length"], [0, base_mean, base_std, base_len])
    
    # Outer Loop: Environment Steps
    raw_obs, _ = env.reset(seed=config.seed)
    state = preprocessor.reset(raw_obs)
    
    episode_reward = 0.0
    episode_length = 0
    
    for step in range(global_step + 1, config.total_steps + 1):
        global_step = step
        
        # 1. Select Action (Epsilon-Greedy)
        epsilon = calculate_epsilon(step, config)
        action = agent.select_action(state, epsilon=epsilon)
        
        # 2. Step Environment
        raw_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        
        # 3. Preprocess Next State
        next_state = preprocessor.step(raw_obs)
        
        # 4. Store Experience
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        
        # 5. Handle Episode End
        if done or episode_length >= config.max_episode_steps:
            episodes_completed += 1
            rolling_rewards.append(episode_reward)
            rolling_lengths.append(episode_length)
            
            # Log episode metrics
            log_to_csv(
                training_log_path, 
                ["step", "episode", "episode_reward", "episode_length", "epsilon", "loss", "mean_q", "mean_target", "mean_td_error", "buffer_size", "learning_updates"],
                [step, episodes_completed, episode_reward, episode_length, epsilon, "", "", "", "", len(buffer), learning_updates]
            )
            
            # Reset for next episode
            raw_obs, _ = env.reset()
            state = preprocessor.reset(raw_obs)
            episode_reward = 0.0
            episode_length = 0
            
        # 6. DQN Training Step
        if step >= config.learning_starts and len(buffer) >= config.batch_size and step % config.train_frequency == 0:
            batch = buffer.sample(config.batch_size)
            
            # Convert NumPy batch to PyTorch Tensors
            states = torch.tensor(batch[0], dtype=torch.float32)
            actions = torch.tensor(batch[1], dtype=torch.int64)
            rewards = torch.tensor(batch[2], dtype=torch.float32)
            next_states = torch.tensor(batch[3], dtype=torch.float32)
            dones = torch.tensor(batch[4], dtype=torch.bool)
            
            # Execute exactly ONE optimizer step
            diagnostics = dqn.train_step(states, actions, rewards, next_states, dones)
            learning_updates += 1
            
            # Log training diagnostics periodically
            if learning_updates % 1000 == 0:
                mean_r100 = np.mean(rolling_rewards) if rolling_rewards else 0.0
                mean_l100 = np.mean(rolling_lengths) if rolling_lengths else 0.0
                
                print(f"Step {step} | Episodes: {episodes_completed} | Epsilon: {epsilon:.3f} | "
                      f"Mean R(100): {mean_r100:.1f} | Mean L(100): {mean_l100:.1f} | "
                      f"Loss: {diagnostics['loss']:.4f} | Replay: {len(buffer)}")
                
                log_to_csv(
                    training_log_path, 
                    ["step", "episode", "episode_reward", "episode_length", "epsilon", "loss", "mean_q", "mean_target", "mean_td_error", "buffer_size", "learning_updates"],
                    [step, episodes_completed, "", "", epsilon, diagnostics["loss"], diagnostics["mean_q"], diagnostics["mean_target"], diagnostics["mean_td_error"], len(buffer), learning_updates]
                )
        
        # 7. Target Network Sync
        if step >= config.learning_starts and step % config.target_update_frequency == 0:
            dqn.update_target_network()
            
        # 8. Checkpoint Save
        if step % config.checkpoint_frequency == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"dqn_step_{step}.pt")
            save_checkpoint(ckpt_path, dqn, step, episodes_completed, epsilon, learning_updates, config)
            
        # 9. Periodic Evaluation
        if step % config.evaluation_frequency == 0:
            eval_mean, eval_std, eval_len = evaluate(env, agent, preprocessor, config.evaluation_episodes)
            print(f"\n--- Periodic Evaluation (Step {step}) ---")
            print(f"Mean Reward: {eval_mean:.2f} ± {eval_std:.2f} | Mean Length: {eval_len:.1f}\n")
            log_to_csv(eval_log_path, ["step", "mean_reward", "std_reward", "mean_length"], [step, eval_mean, eval_std, eval_len])
            
    # Final Checkpoint
    final_path = os.path.join(config.checkpoint_dir, f"dqn_step_{config.total_steps}.pt")
    save_checkpoint(final_path, dqn, config.total_steps, episodes_completed, epsilon, learning_updates, config)
    env.close()
    
    print("\n--- Training Complete! ---")
    print(f"Total Steps: {config.total_steps}")
    print(f"Total Learning Updates: {learning_updates}")

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
