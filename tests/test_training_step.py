import torch
import torch.nn as nn
import torch.optim as optim
import math
from src.dqn import DQN

def get_params(network):
    """Helper to extract a flattened tensor of all parameters"""
    return torch.cat([p.view(-1) for p in network.parameters()])

def test_optimizer_and_loss_exist():
    dqn = DQN(action_count=18, device="cpu")
    assert dqn.optimizer is not None, "Optimizer is missing"
    assert isinstance(dqn.optimizer, optim.Adam), "Optimizer should be Adam"
    assert dqn.loss_fn is not None, "Loss function is missing"
    assert isinstance(dqn.loss_fn, nn.SmoothL1Loss), "Loss should be Huber/SmoothL1Loss"
    print("Test 1 & 2 - Optimizer and Loss exist: Passed")

def test_one_update_and_diagnostics():
    dqn = DQN(action_count=18, device="cpu")
    
    states = torch.zeros((4, 4, 84, 84), dtype=torch.float32)
    actions = torch.tensor([0, 5, 10, 15], dtype=torch.int64)
    rewards = torch.tensor([1.0, 0.0, 10.0, -1.0], dtype=torch.float32)
    next_states = torch.zeros((4, 4, 84, 84), dtype=torch.float32)
    dones = torch.tensor([False, False, True, False], dtype=torch.bool)
    
    diagnostics = dqn.train_step(states, actions, rewards, next_states, dones)
    
    assert "loss" in diagnostics
    assert "mean_q" in diagnostics
    assert "mean_target" in diagnostics
    assert "mean_td_error" in diagnostics
    
    assert not math.isnan(diagnostics["loss"]) and not math.isinf(diagnostics["loss"]), "Loss is not finite"
    
    print("Test 3, 4, 11 - One update executes with diagnostics and finite loss: Passed")

def test_gradients_exist_correctly():
    dqn = DQN(action_count=18, device="cpu")
    
    states = torch.zeros((2, 4, 84, 84), dtype=torch.float32)
    actions = torch.tensor([0, 1], dtype=torch.int64)
    rewards = torch.tensor([1.0, 1.0], dtype=torch.float32)
    next_states = torch.zeros((2, 4, 84, 84), dtype=torch.float32)
    dones = torch.tensor([False, False], dtype=torch.bool)
    
    dqn.train_step(states, actions, rewards, next_states, dones)
    
    # Check online network gradients (should exist)
    online_grads_exist = any(p.grad is not None and torch.any(p.grad != 0) for p in dqn.online_network.parameters())
    assert online_grads_exist, "Online network did not receive gradients"
    
    # Check target network gradients (should be None)
    target_grads_exist = any(p.grad is not None for p in dqn.target_network.parameters())
    assert not target_grads_exist, "Target network illegally received gradients"
    
    print("Test 5 & 6 - Gradient Boundary Verified: Passed")

def test_weights_update_correctly():
    dqn = DQN(action_count=18, device="cpu")
    
    states = torch.zeros((2, 4, 84, 84), dtype=torch.float32)
    actions = torch.tensor([0, 1], dtype=torch.int64)
    rewards = torch.tensor([1.0, 1.0], dtype=torch.float32)
    next_states = torch.zeros((2, 4, 84, 84), dtype=torch.float32)
    dones = torch.tensor([False, False], dtype=torch.bool)
    
    initial_online = get_params(dqn.online_network).clone()
    initial_target = get_params(dqn.target_network).clone()
    
    dqn.train_step(states, actions, rewards, next_states, dones)
    
    final_online = get_params(dqn.online_network)
    final_target = get_params(dqn.target_network)
    
    assert not torch.equal(initial_online, final_online), "Online weights did not change after step!"
    assert torch.equal(initial_target, final_target), "Target weights were illegally modified!"
    print("Test 7 & 8 - Weight Update Boundary Verified: Passed")

def test_learning_direction():
    # Large learning rate to force an observable step
    dqn = DQN(action_count=18, device="cpu", learning_rate=0.1)
    
    states = torch.zeros((1, 4, 84, 84), dtype=torch.float32)
    actions = torch.tensor([0], dtype=torch.int64)
    rewards = torch.tensor([10.0], dtype=torch.float32)
    next_states = torch.zeros((1, 4, 84, 84), dtype=torch.float32)
    dones = torch.tensor([True], dtype=torch.bool)
    
    # Force online network to predict exactly 0.0 initially
    with torch.no_grad():
        dqn.online_network.fc2.weight.fill_(0.0)
        dqn.online_network.fc2.bias.fill_(0.0)
        
    initial_q = dqn.get_current_q_values(states, actions).item()
    assert initial_q == 0.0, "Setup failed"
    
    # Target is reward=10.0
    # Thus, Q should increase towards 10
    dqn.train_step(states, actions, rewards, next_states, dones)
    
    final_q = dqn.get_current_q_values(states, actions).item()
    
    assert final_q > initial_q, f"Learning direction failed! Initial: {initial_q}, Final: {final_q}, Target: 10.0"
    print("Test 12 - Learning Direction Verified: Passed")

def run_all_tests():
    print("Running Training Step Unit Tests...")
    test_optimizer_and_loss_exist()
    test_one_update_and_diagnostics()
    test_gradients_exist_correctly()
    test_weights_update_correctly()
    test_learning_direction()
    print("All training step tests passed successfully!\n")

if __name__ == "__main__":
    run_all_tests()
