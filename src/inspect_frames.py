import os
import gymnasium as gym
import ale_py
import numpy as np
from PIL import Image

def main():
    print("Registering ALE environments...")
    gym.register_envs(ale_py)

    env_id = "ALE/Seaquest-v5"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id)

    observation, info = env.reset()
    
    output_dir = "logs/observation_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Capturing Frames ---")
    frames = [observation]
    
    # Save frame 0
    img = Image.fromarray(observation)
    img.save(os.path.join(output_dir, "frame_0.png"))
    print(f"Saved frame 0 to {output_dir}/frame_0.png")
    
    # Run a few steps to capture consecutive frames
    for i in range(1, 5):
        # Taking a meaningful action (like going right or firing) to induce change
        action = 3  # RIGHT
        observation, _, _, _, _ = env.step(action)
        frames.append(observation)
        
        img = Image.fromarray(observation)
        img_path = os.path.join(output_dir, f"frame_{i}.png")
        img.save(img_path)
        print(f"Saved frame {i} to {img_path}")
        
    print("\n--- Investigating Frame Changes ---")
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i+1].astype(np.int16) - frames[i].astype(np.int16))
        changed_pixels = np.sum(diff > 0)
        mean_diff = np.mean(diff)
        print(f"Difference between Frame {i} and Frame {i+1}:")
        print(f"  Changed pixels: {changed_pixels} out of {diff.size}")
        print(f"  Mean absolute difference: {mean_diff:.4f}")

    env.close()

if __name__ == "__main__":
    main()
