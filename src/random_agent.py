import argparse
import gymnasium as gym
import ale_py

def main():
    parser = argparse.ArgumentParser(description="Run a random agent on Atari Seaquest.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    args = parser.parse_args()

    # Register ALE environments
    gym.register_envs(ale_py)

    # Create the environment with human rendering
    env_id = "ALE/Seaquest-v5"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id, render_mode="human")

    # Inspect the environment
    observation, info = env.reset()
    print("--------------------------------------------------")
    print(f"Environment: {env_id}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Shape: {env.observation_space.shape}")
    print(f"Action Space: {env.action_space}")
    
    # Check if action space is discrete to get the number of actions
    if isinstance(env.action_space, gym.spaces.Discrete):
        print(f"Number of Actions: {env.action_space.n}")
    else:
        print("Number of Actions: Not discrete")
    print("--------------------------------------------------")

    for episode in range(1, args.episodes + 1):
        observation, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (terminated or truncated):
            # Select a random action
            action = env.action_space.sample()

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1

        print(f"Episode: {episode}")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward}")
        print()

    # Clean up
    env.close()
    print("Environment closed cleanly.")

if __name__ == "__main__":
    main()
