import torch
import gymnasium as gym
import ale_py
import numpy as np

from src.preprocessing import AtariPreprocessor
from src.network import SeaquestCNN

def main():
    print("Registering ALE environments...")
    gym.register_envs(ale_py)

    env_id = "ALE/Seaquest-v5"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id)

    # Initialize preprocessor and network
    processor = AtariPreprocessor()
    network = SeaquestCNN()

    print("\n--- Model Architecture ---")
    print(network)
    
    total_params = network.get_parameter_count()
    print(f"\nTotal trainable parameters: {total_params}")
    
    print("\n--- Connecting Pipeline ---")
    # Reset environment
    raw_obs, _ = env.reset()
    
    # Preprocess
    state = processor.reset(raw_obs)
    
    print("Preprocessed State:")
    print(f"  Shape: {state.shape}")
    print(f"  Dtype: {state.dtype}")
    
    # The preprocessed state is (4, 84, 84) in numpy. 
    # The CNN expects a batch dimension, so we add one: (1, 4, 84, 84)
    state_tensor = torch.tensor(state).unsqueeze(0)
    
    print("\nNetwork Input Tensor:")
    print(f"  Shape: {state_tensor.shape}")
    print(f"  Dtype: {state_tensor.dtype}")
    
    # In the forward pass, the tensor is converted to float32 and normalized (0.0 to 1.0)
    print("\nForward pass...")
    with torch.no_grad():
        output = network(state_tensor)
        
    print("\nNetwork Output:")
    print(f"  Shape: {output.shape}")
    print(f"  Dtype: {output.dtype}")
    print(f"  Values:\n{output.numpy()}")
    
    print("\nNote: These 18 outputs are generated from random, untrained weights.")
    print("They do not represent meaningful Q-values yet.")
    
    env.close()

if __name__ == "__main__":
    main()
