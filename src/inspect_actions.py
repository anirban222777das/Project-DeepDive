import argparse
import gymnasium as gym
import ale_py

def main():
    parser = argparse.ArgumentParser(description="Test a specific action in Seaquest.")
    parser.add_argument("--action", type=int, required=True, help="Action index to execute.")
    parser.add_argument("--steps", type=int, default=60, help="Number of steps to execute.")
    args = parser.parse_args()

    print("Registering ALE environments...")
    gym.register_envs(ale_py)

    env_id = "ALE/Seaquest-v5"
    print(f"Creating environment: {env_id} (rendering human)")
    env = gym.make(env_id, render_mode="human")

    # Get the meaning of the action
    meanings = env.unwrapped.get_action_meanings()
    if args.action < 0 or args.action >= len(meanings):
        print(f"Error: Action index {args.action} is out of bounds (0-{len(meanings)-1}).")
        env.close()
        return

    meaning = meanings[args.action]
    
    print("\n--- Action Inspection ---")
    print(f"Action Index: {args.action}")
    print(f"Meaning: {meaning}")
    print(f"Running for {args.steps} steps...")
    
    observation, info = env.reset()
    total_reward = 0
    actual_steps = 0
    
    for _ in range(args.steps):
        observation, reward, terminated, truncated, info = env.step(args.action)
        total_reward += reward
        actual_steps += 1
        
        if terminated or truncated:
            print(f"Episode ended early at step {actual_steps}")
            break

    print("\n--- Result ---")
    print(f"Steps executed: {actual_steps}")
    print(f"Total reward received: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
