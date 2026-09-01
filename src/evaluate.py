import argparse
import os
import csv
import torch
import numpy as np
import gymnasium as gym

from src.preprocessing import AtariPreprocessor
from src.network import SeaquestCNN
from src.agent import DQNAgent

def evaluate(env, agent, preprocessor, episodes, max_steps=27000, return_detailed=False, track_oxygen=False):
    """
    Evaluates the agent in the environment.
    Strictly uses epsilon = 0.
    Does not add to any replay buffer.
    Does not update any network weights.
    Returns: mean_reward, std_reward, mean_length (if return_detailed=False)
             OR dict with detailed stats (if return_detailed=True)
    """
    rewards = []
    lengths = []
    actions_taken = []
    q_values = []
    max_q_values = []
    
    # Oxygen metrics
    oxygen_at_death_list = []
    low_oxygen_reached_count = 0
    surface_after_low_oxygen_count = 0
    oxygen_recovery_events = 0
    
    # We must explicitly disable gradients to prevent memory buildup or accidental learning
    with torch.no_grad():
        for _ in range(episodes):
            raw_obs, info = env.reset()
            state = preprocessor.reset(raw_obs)
            
            episode_reward = 0.0
            episode_length = 0
            
            current_lives = info.get('lives', 4)
            was_low_oxygen = False
            last_oxygen = 64
            
            while True:
                # To collect Q-values, we do manual forward pass, or we can just let select_action do its thing
                # but we need to track Qs for detailed mode.
                
                state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0) / 255.0
                q_vals = agent.model(state_tensor)
                max_q = torch.max(q_vals).item()
                
                # Epsilon = 0 means purely greedy action selection
                action = agent.select_action(state, epsilon=0.0)
                
                q_value_taken = q_vals[0, action].item()
                
                if return_detailed:
                    actions_taken.append(action)
                    q_values.append(q_value_taken)
                    max_q_values.append(max_q)
                
                raw_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if track_oxygen and hasattr(env.unwrapped, 'ale'):
                    ram = env.unwrapped.ale.getRAM()
                    oxy = ram[102]
                    
                    # Track low oxygen state (< 25% capacity)
                    if oxy < 16 and not was_low_oxygen:
                        was_low_oxygen = True
                        low_oxygen_reached_count += 1
                        
                    # Track oxygen recovery (jumps to 64 without dying)
                    if oxy == 64 and last_oxygen < 64:
                        if info.get('lives', current_lives) == current_lives:
                            oxygen_recovery_events += 1
                            if was_low_oxygen:
                                surface_after_low_oxygen_count += 1
                                was_low_oxygen = False
                                
                    # Track death
                    if info.get('lives', current_lives) < current_lives or done:
                        oxygen_at_death_list.append(last_oxygen)
                        current_lives = info.get('lives', 0)
                        was_low_oxygen = False # Reset on death
                        
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
    parser = argparse.ArgumentParser(description="Evaluate Seaquest Baseline DQN")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (empty for untrained)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--max-steps", type=int, default=27000, help="Max steps per episode")
    parser.add_argument("--track-oxygen", action="store_true", help="Track oxygen metrics via RAM (eval only)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Fix for test namespace
    try:
        import ale_py
        gym.register_envs(ale_py)
    except:
        pass
        
    render_mode = "human" if args.render else None
    env = gym.make("ALE/Seaquest-v5", render_mode=render_mode)
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    preprocessor = AtariPreprocessor()
    
    num_actions = env.action_space.n
    from src.dqn import DQN
    dqn_engine = DQN(num_actions, device=args.device)
    agent = DQNAgent(num_actions, dqn_engine.online_network, device=args.device)
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        dqn_engine.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        dqn_engine.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        print(f"Loaded checkpoint from Step {checkpoint.get('step', 'unknown')}")
    else:
        print("Evaluating UNTRAINED network.")
        
    print(f"Evaluating for {args.episodes} episodes...")
    
    results = evaluate(env, agent, preprocessor, episodes=args.episodes, max_steps=args.max_steps, return_detailed=True, track_oxygen=args.track_oxygen)
    
    rewards = results["rewards"]
    lengths = results["lengths"]
    actions = results["actions"]
    q_values = results["q_values"]
    max_qs = results["max_q_values"]
    
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    
    print("\n--- Evaluation Results ---")
    print(f"Episodes: {args.episodes}")
    print(f"Mean Reward:   {mean_reward:.2f}")
    print(f"Median Reward: {median_reward:.2f}")
    print(f"Std Reward:    {std_reward:.2f}")
    print(f"Min Reward:    {min_reward:.2f}")
    print(f"Max Reward:    {max_reward:.2f}")
    print(f"Mean Length:   {mean_length:.2f}")
    print(f"Median Length: {median_length:.2f}")
    
    print("\n--- Action Distribution ---")
    action_counts = {}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    
    total_actions = len(actions)
    for a in sorted(action_counts.keys()):
        pct = (action_counts[a] / total_actions) * 100
        print(f"Action {a}: {pct:.2f}% ({action_counts[a]} times)")
        
    print("\n--- Q-Value Statistics ---")
    print(f"Mean Selected Q: {np.mean(q_values):.4f}")
    print(f"Mean Max Q:      {np.mean(max_qs):.4f}")
    print(f"Std Selected Q:  {np.std(q_values):.4f}")
    
    if args.track_oxygen and "oxygen_metrics" in results:
        oxy_metrics = results["oxygen_metrics"]
        print("\n--- Oxygen Behavior Metrics ---")
        avg_oxy_death = np.mean(oxy_metrics["oxygen_at_death"]) if oxy_metrics["oxygen_at_death"] else 0
        print(f"Avg Oxygen at Death:            {avg_oxy_death:.2f}")
        print(f"Times Reached Low Oxygen (<16): {oxy_metrics['low_oxygen_reached_count']}")
        print(f"Successful Resurfaces from Low: {oxy_metrics['surface_after_low_oxygen_count']}")
        print(f"Total Oxygen Recovery Events:   {oxy_metrics['oxygen_recovery_events']}")
    
    # Save results
    os.makedirs("logs/evaluation", exist_ok=True)
    
    ckpt_name = os.path.basename(args.checkpoint) if args.checkpoint else "untrained"
    summary_file = "logs/evaluation/evaluation_summary.csv"
    
    write_header = not os.path.exists(summary_file)
    with open(summary_file, "a") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["checkpoint", "episodes", "mean_reward", "median_reward", "std_reward", "min_reward", "max_reward", "mean_episode_length", "median_episode_length"])
        writer.writerow([ckpt_name, args.episodes, mean_reward, median_reward, std_reward, min_reward, max_reward, mean_length, median_length])
        
    results_file = f"logs/evaluation/{ckpt_name}_results.csv"
    with open(results_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "episode_length", "seed"])
        for i in range(args.episodes):
            writer.writerow([i+1, rewards[i], lengths[i], args.seed])
            
    action_file = "logs/evaluation/action_distribution.csv"
    write_header = not os.path.exists(action_file)
    with open(action_file, "a") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["checkpoint", "action_id", "count", "percentage"])
        for a in sorted(action_counts.keys()):
            pct = (action_counts[a] / total_actions) * 100
            writer.writerow([ckpt_name, a, action_counts[a], pct])
            
    print(f"\nResults saved to logs/evaluation/")
    
if __name__ == "__main__":
    main()
