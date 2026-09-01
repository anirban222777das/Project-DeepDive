import argparse
import time
import torch
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from src.preprocessing import AtariPreprocessor
from src.network import SeaquestCNN
from src.agent import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Watch Seaquest Baseline DQN Play")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint (empty for untrained)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second for rendering")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # We must explicitly use render_mode="human" to pop open the pygame window
    env = gym.make("ALE/Seaquest-v5", render_mode="human")
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
        
    print(f"Watching for {args.episodes} episodes...")
    
    # Get meaning of action IDs if available
    action_meanings = env.unwrapped.get_action_meanings()
    
    with torch.no_grad():
        for ep in range(args.episodes):
            raw_obs, _ = env.reset()
            state = preprocessor.reset(raw_obs)
            
            episode_reward = 0.0
            episode_length = 0
            
            print(f"\n--- Starting Episode {ep + 1} ---")
            
            while True:
                # Forward pass to get Q values for terminal HUD
                state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0) / 255.0
                q_vals = agent.model(state_tensor)
                
                action = agent.select_action(state, epsilon=0.0)
                q_value_taken = q_vals[0, action].item()
                
                raw_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                action_name = action_meanings[action] if action < len(action_meanings) else str(action)
                
                # Terminal HUD
                print(f"Ep: {ep+1} | Step: {episode_length:4d} | Action: {action_name:10s} (ID {action:2d}) | Q: {q_value_taken:7.3f} | Step Reward: {reward:3.0f} | Total: {episode_reward:5.0f}", end="\r")
                
                time.sleep(1.0 / args.fps)
                
                if done:
                    print(f"\nEpisode {ep + 1} finished with Total Reward: {episode_reward} after {episode_length} steps.")
                    break
                    
                state = preprocessor.step(raw_obs)
                
    env.close()

if __name__ == "__main__":
    main()
