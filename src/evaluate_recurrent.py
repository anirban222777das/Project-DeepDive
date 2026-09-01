import argparse
import os
import csv
import torch
import numpy as np
import gymnasium as gym

from src.preprocessing import AtariPreprocessor
from src.recurrent_network import SeaquestDRQN
from src.recurrent_agent import DRQNAgent

def evaluate(env, agent, preprocessor, episodes, max_steps=27000, return_detailed=False, track_oxygen=False):
    rewards = []
    lengths = []
    actions_taken = []
    q_values = []
    max_q_values = []
    
    oxygen_at_death_list = []
    low_oxygen_reached_count = 0
    surface_after_low_oxygen_count = 0
    oxygen_recovery_events = 0
    
    with torch.no_grad():
        for _ in range(episodes):
            raw_obs, info = env.reset()
            state = preprocessor.reset(raw_obs)
            agent.reset_hidden_state()
            
            episode_reward = 0.0
            episode_length = 0
            
            current_lives = info.get('lives', 4)
            was_low_oxygen = False
            last_oxygen = 64
            
            while True:
                state_tensor = torch.tensor(state, dtype=torch.uint8, device=agent.device).unsqueeze(0).unsqueeze(0)
                q_vals, _ = agent.model(state_tensor, agent.hidden_state)
                max_q = torch.max(q_vals).item()
                
                action = agent.select_action(state, epsilon=0.0)
                q_value_taken = q_vals[0, 0, action].item()
                
                if return_detailed:
                    actions_taken.append(action)
                    q_values.append(q_value_taken)
                    max_q_values.append(max_q)
                
                raw_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if track_oxygen and hasattr(env.unwrapped, 'ale'):
                    ram = env.unwrapped.ale.getRAM()
                    oxy = ram[102]
                    
                    if oxy < 16 and not was_low_oxygen:
                        was_low_oxygen = True
                        low_oxygen_reached_count += 1
                        
                    if oxy == 64 and last_oxygen < 64:
                        if info.get('lives', current_lives) == current_lives:
                            oxygen_recovery_events += 1
                            if was_low_oxygen:
                                surface_after_low_oxygen_count += 1
                                was_low_oxygen = False
                                
                    if info.get('lives', current_lives) < current_lives or done:
                        oxygen_at_death_list.append(last_oxygen)
                        current_lives = info.get('lives', 0)
                        was_low_oxygen = False
                        
                    last_oxygen = oxy
                
                episode_reward += reward
                episode_length += 1
                
                if done or episode_length >= max_steps:
                    break
                    
                state = preprocessor.step(raw_obs)
                
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
    if return_detailed:
        stats = {
            "rewards": rewards,
            "lengths": lengths,
            "actions": actions_taken,
            "q_values": q_values,
            "max_q_values": max_q_values
        }
        if track_oxygen:
            stats["oxygen_metrics"] = {
                "oxygen_at_death": oxygen_at_death_list,
                "low_oxygen_reached_count": low_oxygen_reached_count,
                "surface_after_low_oxygen_count": surface_after_low_oxygen_count,
                "oxygen_recovery_events": oxygen_recovery_events
            }
        return stats
    return np.mean(rewards), np.std(rewards), np.mean(lengths)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Seaquest DRQN")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    parser.add_argument("--max-steps", type=int, default=27000, help="Max steps per episode")
    parser.add_argument("--track-oxygen", action="store_true", help="Track oxygen metrics via RAM (eval only)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        import ale_py
        gym.register_envs(ale_py)
    except:
        pass
        
    env = gym.make("ALE/Seaquest-v5")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    preprocessor = AtariPreprocessor()
    
    num_actions = env.action_space.n
    from src.drqn import DRQN
    drqn_engine = DRQN(num_actions, device=args.device)
    agent = DRQNAgent(num_actions, drqn_engine.online_network, device=args.device)
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        drqn_engine.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        drqn_engine.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        print(f"Loaded checkpoint from Step {checkpoint.get('step', 'unknown')}")
    else:
        print("Evaluating UNTRAINED network.")
        
    print(f"Evaluating for {args.episodes} episodes...")
    
    results = evaluate(env, agent, preprocessor, episodes=args.episodes, max_steps=args.max_steps, return_detailed=True, track_oxygen=args.track_oxygen)
    
    rewards = results["rewards"]
    lengths = results["lengths"]
    
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(lengths)
    
    print("\n--- Evaluation Results ---")
    print(f"Episodes: {args.episodes}")
    print(f"Mean Reward:   {mean_reward:.2f}")
    print(f"Median Reward: {median_reward:.2f}")
    print(f"Std Reward:    {std_reward:.2f}")
    print(f"Mean Length:   {mean_length:.2f}")
    
    if args.track_oxygen and "oxygen_metrics" in results:
        oxy_metrics = results["oxygen_metrics"]
        print("\n--- Oxygen Behavior Metrics ---")
        avg_oxy_death = np.mean(oxy_metrics["oxygen_at_death"]) if oxy_metrics["oxygen_at_death"] else 0
        print(f"Avg Oxygen at Death:            {avg_oxy_death:.2f}")
        print(f"Times Reached Low Oxygen (<16): {oxy_metrics['low_oxygen_reached_count']}")
        print(f"Successful Resurfaces from Low: {oxy_metrics['surface_after_low_oxygen_count']}")
        print(f"Total Oxygen Recovery Events:   {oxy_metrics['oxygen_recovery_events']}")
    
if __name__ == "__main__":
    main()
