import gymnasium as gym
import ale_py
import numpy as np

def main():
    print("Registering ALE environments...")
    gym.register_envs(ale_py)

    env_id = "ALE/Seaquest-v5"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id)

    observation, info = env.reset()
    
    print("\n--- Observation Inspection ---")
    print(f"Python type: {type(observation)}")
    if isinstance(observation, np.ndarray):
        print(f"NumPy dtype: {observation.dtype}")
        print(f"Shape: {observation.shape}")
        print(f"Minimum pixel value: {np.min(observation)}")
        print(f"Maximum pixel value: {np.max(observation)}")
        print(f"Mean pixel value: {np.mean(observation):.2f}")
        print(f"Number of unique values: {len(np.unique(observation))}")
        
        # Verify against observation space
        space = env.observation_space
        print("\n--- Verification ---")
        print(f"Conforms to observation space shape? {observation.shape == space.shape}")
        print(f"Conforms to observation space dtype? {observation.dtype == space.dtype}")
        print(f"Is within space bounds? {space.contains(observation)}")
    else:
        print("Observation is not a numpy array.")

    env.close()

if __name__ == "__main__":
    main()
