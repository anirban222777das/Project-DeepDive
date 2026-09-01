import torch
import numpy as np
from src.dqn import DQN
from src.replay_buffer import ReplayBuffer

def get_params_copy(network):
    return torch.cat([p.view(-1).detach().clone() for p in network.parameters()])

def main():
    print("--- Phase 8: One-Step Learning Update Inspection ---")
    
    dqn = DQN(action_count=18, gamma=0.99, learning_rate=1e-3)
    buffer = ReplayBuffer(capacity=100)
    
    print(f"Device: {dqn.device}")
    
    # 1. Synthesize minimal experiences to fill batch size
    batch_size = 8
    print(f"\nAdding {batch_size} synthetic experiences to Replay Buffer...")
    for i in range(batch_size):
        # We use dummy 84x84 screens
        state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
        next_state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
        action = i % 18
        reward = float(i)
        done = (i % 2 == 0) # Alternate dones
        
        buffer.add(state, action, reward, next_state, done)
        
    # 2. Sample the batch
    print("Sampling a batch...")
    states_np, actions_np, rewards_np, next_states_np, dones_np = buffer.sample(batch_size)
    
    # Convert numpy arrays to tensors for PyTorch (converting uint8 to float32 for NN input)
    states = torch.tensor(states_np, dtype=torch.float32)
    actions = torch.tensor(actions_np, dtype=torch.int64)
    rewards = torch.tensor(rewards_np, dtype=torch.float32)
    next_states = torch.tensor(next_states_np, dtype=torch.float32)
    dones = torch.tensor(dones_np, dtype=torch.bool)
    
    # 3. Snapshot weights before training
    print("Capturing pre-update weights...")
    initial_online = get_params_copy(dqn.online_network)
    initial_target = get_params_copy(dqn.target_network)
    
    # 4. Perform ONE Learning Update
    print("\nExecuting ONE Learning Update (Huber Loss -> Backward -> Optimizer Step)...")
    diagnostics = dqn.train_step(states, actions, rewards, next_states, dones)
    
    # 5. Print Diagnostics
    print("\n--- Diagnostics ---")
    print(f"Loss:          {diagnostics['loss']:.6f}")
    print(f"Mean Q:        {diagnostics['mean_q']:.6f}")
    print(f"Mean Target:   {diagnostics['mean_target']:.6f}")
    print(f"Mean TD Error: {diagnostics['mean_td_error']:.6f}")
    
    # 6. Verify parameter changes
    final_online = get_params_copy(dqn.online_network)
    final_target = get_params_copy(dqn.target_network)
    
    online_changed = not torch.equal(initial_online, final_online)
    target_changed = not torch.equal(initial_target, final_target)
    
    print("\n--- Parameter Update Boundary ---")
    print(f"Online Network parameters updated: {'YES' if online_changed else 'NO (ERROR)'}")
    print(f"Target Network parameters updated: {'YES (ERROR)' if target_changed else 'NO (FROZEN)'}")
    
    print("\nInspection Complete! The network can now physically learn from data.")

if __name__ == "__main__":
    main()
