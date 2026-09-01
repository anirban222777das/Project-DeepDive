import gymnasium as gym
import ale_py
import numpy as np

def main():
    print("Registering ALE environments...")
    gym.register_envs(ale_py)

    env_id = "ALE/Seaquest-v5"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id)

    episodes = 5
    
    total_rewards = []
    episode_lengths = []
    
    all_positive = 0
    all_negative = 0
    all_zero = 0
    all_nonzero = 0
    
    min_reward = float('inf')
    max_reward = float('-inf')
    
    termination_reasons = []

    print(f"\n--- Running {episodes} episodes to inspect rewards ---")
    
    for ep in range(1, episodes + 1):
        observation, info = env.reset()
        
        ep_reward = 0
        ep_length = 0
        
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            ep_length += 1
            
            if reward > 0:
                all_positive += 1
                all_nonzero += 1
            elif reward < 0:
                all_negative += 1
                all_nonzero += 1
            else:
                all_zero += 1
                
            if reward < min_reward:
                min_reward = reward
            if reward > max_reward:
                max_reward = reward
                
            if terminated or truncated:
                reason = "Terminated (Game Over / Life Loss)" if terminated else "Truncated (Time Limit)"
                termination_reasons.append(reason)
                break
                
        total_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(f"Episode {ep}: Reward = {ep_reward}, Length = {ep_length}, End Reason = {termination_reasons[-1]}")

    print("\n--- Reward Statistics ---")
    print(f"Episodes: {episodes}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print(f"Average Total Reward: {np.mean(total_rewards):.1f}")
    
    print(f"\nTotal steps across all episodes: {np.sum(episode_lengths)}")
    print(f"Positive rewards received: {all_positive}")
    print(f"Negative rewards received: {all_negative}")
    print(f"Zero rewards received: {all_zero}")
    print(f"Non-zero frequency: {all_nonzero / np.sum(episode_lengths) * 100:.2f}%")
    
    print(f"\nMinimum single reward observed: {min_reward}")
    print(f"Maximum single reward observed: {max_reward}")

    env.close()

if __name__ == "__main__":
    main()
