import os
import gymnasium as gym
import ale_py
import numpy as np
from PIL import Image
from src.preprocessing import AtariPreprocessor

def main():
    print("Registering ALE environments...")
    gym.register_envs(ale_py)

    env_id = "ALE/Seaquest-v5"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id)

    processor = AtariPreprocessor()
    
    output_dir = "logs/preprocessing_samples"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Preprocessing Integration ---")
    observation, info = env.reset()
    
    # Run a few steps so the frame is more interesting than just the start screen
    for _ in range(20):
        observation, _, _, _, _ = env.step(1) # FIRE
        
    print(f"Raw shape: {observation.shape}")
    print(f"Raw dtype: {observation.dtype}")

    # Save RAW
    img_raw = Image.fromarray(observation)
    img_raw.save(os.path.join(output_dir, "1_raw.png"))
    
    # Process single frame to test intermediate steps
    gray_frame = np.dot(observation[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    img_gray = Image.fromarray(gray_frame)
    img_gray.save(os.path.join(output_dir, "2_grayscale.png"))
    
    # Test Full Processor
    state = processor.reset(observation)
    print(f"Processed frame shape (internal): (84, 84)")
    print(f"Stacked state shape: {state.shape}")
    print(f"Processed dtype: {state.dtype}")
    
    # Run a few more steps to show rolling history
    # Step 1
    obs1, _, _, _, _ = env.step(3) # RIGHT
    state = processor.step(obs1)
    
    # Step 2
    obs2, _, _, _, _ = env.step(3) # RIGHT
    state = processor.step(obs2)
    
    # Step 3
    obs3, _, _, _, _ = env.step(3) # RIGHT
    state = processor.step(obs3)
    
    print("\n--- Visualizing Stacked State ---")
    # State has shape (4, 84, 84)
    # Let's save each channel as an image to show the history
    for i in range(4):
        frame_img = Image.fromarray(state[i])
        out_path = os.path.join(output_dir, f"3_stacked_frame_{i}.png")
        frame_img.save(out_path)
        print(f"Saved state frame {i} to {out_path}")

    env.close()
    print("\nPreprocessing demonstration complete.")

if __name__ == "__main__":
    main()
