import torch
from src.dqn import DQN

def check_sync(dqn):
    for p_on, p_tar in zip(dqn.online_network.parameters(), dqn.target_network.parameters()):
        if not torch.equal(p_on, p_tar):
            return False
    return True

def main():
    print("--- Phase 7 DQN Mathematical Core Inspection ---")
    
    dqn = DQN(action_count=18, gamma=0.99)
    print(f"Device: {dqn.device}")
    print(f"Gamma: {dqn.gamma}")
    
    sync_status = check_sync(dqn)
    print(f"Initial Synchronization: {'Match' if sync_status else 'Mismatch'}")
    
    # 1. Create a synthetic batch
    print("\n--- Synthetic Batch (Size 3) ---")
    states = torch.zeros((3, 4, 84, 84), dtype=torch.uint8)
    actions = torch.tensor([5, 10, 15], dtype=torch.int64)
    rewards = torch.tensor([1.0, 0.0, 50.0], dtype=torch.float32)
    next_states = torch.zeros((3, 4, 84, 84), dtype=torch.uint8)
    dones = torch.tensor([False, False, True], dtype=torch.bool)
    
    # Moving tensors to device
    states = states.to(dqn.device)
    actions = actions.to(dqn.device)
    rewards = rewards.to(dqn.device)
    next_states = next_states.to(dqn.device)
    dones = dones.to(dqn.device)
    
    # 2. Extract Current Q-values
    current_q_values = dqn.get_current_q_values(states, actions)
    print("\n[Online] Current Q-values Q(s, a):")
    print(current_q_values.cpu().detach().numpy())
    
    # 3. Calculate Bellman Targets
    bellman_targets = dqn.compute_bellman_targets(rewards, next_states, dones)
    print("\n[Target] Bellman Targets (r + γ max Q'):")
    print(bellman_targets.cpu().numpy())
    print("Notice the 3rd index (done=True) is exactly the reward (50.0) without gamma future contribution.")
    
    # 4. Demonstrate synchronization
    print("\n--- Target Synchronization Demonstration ---")
    print("Modifying online network weights (simulating training)...")
    with torch.no_grad():
        dqn.online_network.fc2.bias += 100.0
        
    print(f"Synchronization before update: {'Match' if check_sync(dqn) else 'Mismatch (Independent)'}")
    
    dqn.update_target_network()
    print(f"Synchronization after update(): {'Match' if check_sync(dqn) else 'Mismatch'}")
    
    print("\nInspection complete! The mathematics are ready for loss calculation.")

if __name__ == "__main__":
    main()
