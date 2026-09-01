import os
import csv
import json
import random
import torch
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque

gym.register_envs(ale_py)

from src.config import parse_args, TrainingConfig
from src.preprocessing import AtariPreprocessor
from src.recurrent_agent import DRQNAgent
from src.recurrent_replay_buffer import RecurrentReplayBuffer
from src.drqn import DRQN
from src.evaluate_recurrent import evaluate

def calculate_epsilon(step: int, config: TrainingConfig) -> float:
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

def save_checkpoint(path: str, drqn: DRQN, step: int, episode: int, epsilon: float, learning_updates: int, config: TrainingConfig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'online_network_state_dict': drqn.online_network.state_dict(),
        'target_network_state_dict': drqn.target_network.state_dict(),
        'optimizer_state_dict': drqn.optimizer.state_dict(),
        'step': step,
        'episode': episode,
        'epsilon': epsilon,
        'learning_updates': learning_updates,
        'algorithm': "drqn",
        'rng_states': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()
        },
        'config': {
            'batch_size': config.batch_size,
            'gamma': config.gamma,
            'learning_rate': config.learning_rate,
            'seq_len': 8
        }
    }
    torch.save(checkpoint, path)

def train(config: TrainingConfig):
    set_seeds(config.seed)
    
    env = gym.make("ALE/Seaquest-v5")
    preprocessor = AtariPreprocessor()
    
    num_actions = env.action_space.n
    
    seq_len = 8
    
    # Initialize Core DRQN Components
    buffer = RecurrentReplayBuffer(capacity=config.replay_capacity, seq_len=seq_len)
    drqn_engine = DRQN(num_actions=num_actions, device=config.device, learning_rate=config.learning_rate)
    agent = DRQNAgent(num_actions=num_actions, model=drqn_engine.online_network, device=config.device)
    
    # Setup logging
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    metrics_file = os.path.join(config.log_dir, "training_metrics.csv")
    with open(metrics_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "reward", "length", "epsilon", "loss", "mean_q", "max_q", "grad_norm", "learning_updates", "env_steps"])
        
    config_file = os.path.join(config.log_dir, "config.json")
    with open(config_file, "w") as f:
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        config_dict["algorithm"] = "drqn"
        config_dict["seq_len"] = seq_len
        json.dump(config_dict, f, indent=4)
        
    # Training state
    env_steps = 0
    episode_count = 0
    learning_updates = 0
    
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    
    print(f"Starting DRQN training for {config.total_steps} steps on {config.device}...")
    
    # Fill replay buffer with purely random actions first
    print(f"Prefilling replay buffer with {config.learning_starts} random transitions...")
    while len(buffer) < config.learning_starts:
        raw_obs, _ = env.reset()
        state = preprocessor.reset(raw_obs)
        agent.reset_hidden_state()
        
        while True:
            action = env.action_space.sample()
            raw_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = preprocessor.step(raw_obs)
            buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                break
                
    print("Buffer filled. Beginning training loop.")
    
    while env_steps < config.total_steps:
        raw_obs, _ = env.reset()
        state = preprocessor.reset(raw_obs)
        agent.reset_hidden_state()
        
        episode_reward = 0.0
        episode_length = 0
        
        while True:
            epsilon = calculate_epsilon(env_steps, config)
            action = agent.select_action(state, epsilon)
            
            raw_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = preprocessor.step(raw_obs)
            
            buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            env_steps += 1
            
            # Learning step
            if env_steps >= config.learning_starts and env_steps % config.train_frequency == 0:
                try:
                    states, actions, rewards_batch, next_states, dones = buffer.sample(config.batch_size)
                    loss, mean_q, max_q, grad_norm = drqn_engine.update(
                        states, actions, rewards_batch, next_states, dones, config.gamma
                    )
                    learning_updates += 1
                except ValueError:
                    # Not enough sequences yet
                    pass
            
            # Target Network Update
            if env_steps % config.target_update_frequency == 0:
                drqn_engine.update_target_network()
                
            # Evaluation
            if env_steps % config.evaluation_frequency == 0:
                print(f"[{env_steps}/{config.total_steps}] Evaluating agent...")
                eval_reward, eval_std, eval_length = evaluate(
                    env, agent, preprocessor, config.evaluation_episodes, return_detailed=False, track_oxygen=False
                )
                print(f"Evaluation: Mean Reward={eval_reward:.2f} ± {eval_std:.2f} | Mean Length={eval_length:.2f}")
                
                set_seeds(config.seed + env_steps)
                agent.reset_hidden_state()
                
            # Checkpointing
            if env_steps % config.checkpoint_frequency == 0:
                ckpt_path = os.path.join(config.checkpoint_dir, f"drqn_step_{env_steps}.pt")
                save_checkpoint(ckpt_path, drqn_engine, env_steps, episode_count, epsilon, learning_updates, config)
                print(f"Saved checkpoint to {ckpt_path}")
                
            if done or env_steps >= config.total_steps:
                break
                
        episode_count += 1
        recent_rewards.append(episode_reward)
        recent_lengths.append(episode_length)
        
        try:
            with open(metrics_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([env_steps, episode_count, episode_reward, episode_length, epsilon, 
                                 loss if 'loss' in locals() else "", 
                                 mean_q if 'mean_q' in locals() else "", 
                                 max_q if 'max_q' in locals() else "", 
                                 grad_norm if 'grad_norm' in locals() else "", 
                                 learning_updates, env_steps])
        except Exception:
            pass
            
        if episode_count % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode_count} | Steps: {env_steps} | Epsilon: {epsilon:.3f} | Avg Reward (100): {avg_reward:.1f}")

    final_ckpt = os.path.join(config.checkpoint_dir, "drqn_final.pt")
    save_checkpoint(final_ckpt, drqn_engine, env_steps, episode_count, epsilon, learning_updates, config)
    print("Training complete!")

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
