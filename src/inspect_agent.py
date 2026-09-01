import argparse
import gymnasium as gym
import ale_py

from src.preprocessing import AtariPreprocessor
from src.network import SeaquestCNN
from src.agent import DQNAgent

def main():
    parser = argparse.ArgumentParser(description="Inspect DQN Agent action selection.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon for action selection (0.0=greedy, 1.0=random)")
    parser.add_argument("--steps", type=int, default=3, help="Number of steps to execute")
    args = parser.parse_args()

    print("Registering ALE environments...")
    gym.register_envs(ale_py)

    env_id = "ALE/Seaquest-v5"
    print(f"Creating environment: {env_id} (rendering human)")
    env = gym.make(env_id, render_mode="human")

    action_meanings = env.unwrapped.get_action_meanings()
    action_count = len(action_meanings)
    
    print(f"Action Count: {action_count}")

    # Initialize components
    processor = AtariPreprocessor()
    network = SeaquestCNN(num_actions=action_count)
    agent = DQNAgent(action_count=action_count, model=network)

    print(f"\n--- Running DQN Agent (Epsilon={args.epsilon}) ---")
    raw_obs, _ = env.reset()
    state = processor.reset(raw_obs)
    
    for step in range(1, args.steps + 1):
        print(f"\nStep: {step}")
        
        # We manually call _get_q_values just to print them for inspection
        q_values = agent._get_q_values(state).cpu().numpy()[0]
        print("Q-values:")
        for i in range(action_count):
            # Print a few to not spam too much, or just all since it's 18
            print(f"  {i}: {q_values[i]:.4f}")
            
        action = agent.select_action(state, epsilon=args.epsilon)
        
        print(f"\nSelected action:\n{action}")
        print(f"\nAction meaning:\n{action_meanings[action]}")
        
        # Take the action in the environment
        raw_obs, reward, terminated, truncated, _ = env.step(action)
        state = processor.step(raw_obs)
        
        if terminated or truncated:
            print("Episode ended early.")
            break

    env.close()
    print("\nInspection complete. The network is untrained; these values are arbitrary.")

if __name__ == "__main__":
    main()
