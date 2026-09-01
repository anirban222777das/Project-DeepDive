import torch
import torch.nn as nn
from src.dqn import DQN

def get_params(network):
    """Helper to extract a flattened tensor of all parameters"""
    return torch.cat([p.view(-1) for p in network.parameters()])

def test_dqn_creation_and_initialization():
    dqn = DQN(action_count=18, device="cpu")
    
    assert dqn.online_network is not None
    assert dqn.target_network is not None
    
    # Test 2 - Same initial parameters
    online_params = get_params(dqn.online_network)
    target_params = get_params(dqn.target_network)
    assert torch.equal(online_params, target_params), "Networks did not start synchronized!"
    print("Test 1 & 2 - DQN Creation & Synchronization: Passed")

def test_independent_networks_and_update():
    dqn = DQN(action_count=18, device="cpu")
    
    # Modify online parameter manually (we modify the first bias slightly)
    with torch.no_grad():
        dqn.online_network.fc2.bias[0] += 10.0
        
    online_params = get_params(dqn.online_network)
    target_params = get_params(dqn.target_network)
    
    # Test 3 - Independent Networks
    assert not torch.equal(online_params, target_params), "Modifying online modified target!"
    print("Test 3 - Independent Networks: Passed")
    
    # Test 4 - Target Update
    dqn.update_target_network()
    online_params_updated = get_params(dqn.online_network)
    target_params_updated = get_params(dqn.target_network)
    assert torch.equal(online_params_updated, target_params_updated), "Update failed to synchronize!"
    print("Test 4 - Target Update: Passed")

def test_current_q_values():
    dqn = DQN(action_count=18, device="cpu")
    
    # Synthetic Batch B=2
    states = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
    actions = torch.tensor([5, 12], dtype=torch.int64)
    
    # Deterministic output verification
    # We will manually replace the final layer weights to known values
    with torch.no_grad():
        dqn.online_network.fc2.weight.fill_(0.0)
        dqn.online_network.fc2.bias.fill_(0.0)
        dqn.online_network.fc2.bias[5] = 55.5
        dqn.online_network.fc2.bias[12] = -12.3
    
    current_q = dqn.get_current_q_values(states, actions)
    
    assert current_q.shape == (2,), "Current Q-values shape incorrect"
    assert torch.allclose(current_q[0], torch.tensor(55.5))
    assert torch.allclose(current_q[1], torch.tensor(-12.3))
    print("Test 5 - Current Q-values Calculation: Passed")

def test_bellman_targets():
    dqn = DQN(action_count=18, device="cpu", gamma=0.99)
    
    # We need to test terminal (done=True) and non-terminal (done=False)
    # Batch size B=2
    next_states = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
    rewards = torch.tensor([10.0, 50.0], dtype=torch.float32)
    dones = torch.tensor([False, True], dtype=torch.bool)
    
    # Deterministic output setup for TARGET network
    with torch.no_grad():
        dqn.target_network.fc2.weight.fill_(0.0)
        dqn.target_network.fc2.bias.fill_(0.0)
        # We set the maximum action value strictly to 100.0
        dqn.target_network.fc2.bias[0] = 100.0
        
    targets = dqn.compute_bellman_targets(rewards, next_states, dones)
    
    assert targets.shape == (2,), "Targets shape incorrect"
    assert not targets.requires_grad, "Targets should not require gradients"
    
    # Item 0 (done=False): reward(10) + 0.99 * maxQ(100) = 10 + 99 = 109
    assert torch.allclose(targets[0], torch.tensor(109.0)), f"Expected 109.0, got {targets[0]}"
    print("Test 8 - Non-terminal transition: Passed")
    
    # Item 1 (done=True): reward(50) + 0.99 * 0 = 50
    assert torch.allclose(targets[1], torch.tensor(50.0)), f"Expected 50.0, got {targets[1]}"
    print("Test 7 - Terminal transition: Passed")
    
    print("Test 6 - Bellman targets shape & gradients: Passed")

def test_target_network_independence_in_computation():
    dqn = DQN(action_count=18, device="cpu", gamma=0.99)
    next_states = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
    rewards = torch.tensor([5.0], dtype=torch.float32)
    dones = torch.tensor([False], dtype=torch.bool)
    
    # Zero out target network so max Q is exactly 0.0
    with torch.no_grad():
        dqn.target_network.fc2.weight.fill_(0.0)
        dqn.target_network.fc2.bias.fill_(0.0)
        
    # Heavily modify online network so max Q is 5000.0
    with torch.no_grad():
        dqn.online_network.fc2.weight.fill_(0.0)
        dqn.online_network.fc2.bias.fill_(5000.0)
        
    # Calculate target. It should use TARGET network (maxQ=0), thus target=5.0
    # If it accidentally used online network, target would be 5 + 0.99 * 5000
    targets = dqn.compute_bellman_targets(rewards, next_states, dones)
    
    assert torch.allclose(targets[0], torch.tensor(5.0)), "Bellman target incorrectly used Online Network!"
    print("Test 9 - Target Network Exclusively Used: Passed")

def run_all_tests():
    print("Running DQN Mathematical Engine Tests...")
    test_dqn_creation_and_initialization()
    test_independent_networks_and_update()
    test_current_q_values()
    test_bellman_targets()
    test_target_network_independence_in_computation()
    print("All DQN mathematical tests passed successfully!\n")

if __name__ == "__main__":
    run_all_tests()
