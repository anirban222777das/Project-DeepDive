import torch
import pytest
from src.dqn import DQN

class MockNetwork(torch.nn.Module):
    def __init__(self, q_values):
        super().__init__()
        # q_values should be a tensor of shape (B, num_actions)
        self.q_values = q_values
        
    def forward(self, x):
        # Ignore input x, just return the mocked q_values
        return self.q_values

def test_double_dqn_vs_vanilla_dqn_targets():
    """
    Test explicitly showing the mathematical difference between 
    Vanilla DQN and Double DQN using mock networks.
    
    Setup:
    Online Network Q-values: [10, 20, 15]  -> max action is 1 (value 20)
    Target Network Q-values: [100, 30, 50] -> max action is 0 (value 100)
    
    Vanilla DQN should select max target Q-value = 100.
    Double DQN should select online max action (1) and evaluate on target (30).
    """
    # Create fake tensors
    # Batch size 1, 3 actions
    online_q = torch.tensor([[10.0, 20.0, 15.0]])
    target_q = torch.tensor([[100.0, 30.0, 50.0]])
    
    # Fake inputs (B, 4, 84, 84)
    dummy_states = torch.zeros((1, 4, 84, 84))
    rewards = torch.tensor([5.0])
    dones = torch.tensor([False])
    
    # Initialize Vanilla DQN
    vanilla_dqn = DQN(action_count=3, device="cpu", gamma=0.9, double_dqn=False)
    # Inject mock networks
    vanilla_dqn.online_network = MockNetwork(online_q)
    vanilla_dqn.target_network = MockNetwork(target_q)
    
    vanilla_targets = vanilla_dqn.compute_bellman_targets(rewards, dummy_states, dones)
    
    # Vanilla expected: 5.0 + 0.9 * 100.0 = 95.0
    assert torch.allclose(vanilla_targets, torch.tensor([95.0])), f"Vanilla DQN failed, got {vanilla_targets}"
    
    # Initialize Double DQN
    ddqn = DQN(action_count=3, device="cpu", gamma=0.9, double_dqn=True)
    # Inject mock networks
    ddqn.online_network = MockNetwork(online_q)
    ddqn.target_network = MockNetwork(target_q)
    
    ddqn_targets = ddqn.compute_bellman_targets(rewards, dummy_states, dones)
    
    # Double DQN expected: 5.0 + 0.9 * 30.0 = 32.0
    assert torch.allclose(ddqn_targets, torch.tensor([32.0])), f"Double DQN failed, got {ddqn_targets}"
    
def test_double_dqn_terminal_state():
    """
    Test that Double DQN ignores future Q-values when done=True.
    """
    online_q = torch.tensor([[10.0, 20.0, 15.0]])
    target_q = torch.tensor([[100.0, 30.0, 50.0]])
    
    dummy_states = torch.zeros((1, 4, 84, 84))
    rewards = torch.tensor([5.0])
    dones = torch.tensor([True])  # Terminal state!
    
    ddqn = DQN(action_count=3, device="cpu", gamma=0.9, double_dqn=True)
    ddqn.online_network = MockNetwork(online_q)
    ddqn.target_network = MockNetwork(target_q)
    
    ddqn_targets = ddqn.compute_bellman_targets(rewards, dummy_states, dones)
    
    # Expected: 5.0 + 0.9 * 30.0 * 0 = 5.0
    assert torch.allclose(ddqn_targets, torch.tensor([5.0])), f"DDQN terminal failed, got {ddqn_targets}"

def test_target_network_gradients_disabled():
    """
    Verify that Double DQN doesn't accidentally enable gradients on target network.
    """
    ddqn = DQN(action_count=3, device="cpu", double_dqn=True)
    for param in ddqn.target_network.parameters():
        assert param.requires_grad == False, "Target network should not track gradients"
        
def test_bellman_target_shape():
    """
    Verify the shape of Bellman targets is correctly (B,).
    """
    ddqn = DQN(action_count=18, device="cpu", double_dqn=True)
    B = 32
    dummy_states = torch.zeros((B, 4, 84, 84))
    rewards = torch.ones((B,))
    dones = torch.zeros((B,), dtype=torch.bool)
    
    targets = ddqn.compute_bellman_targets(rewards, dummy_states, dones)
    
    assert targets.shape == (B,), f"Expected shape {(B,)}, got {targets.shape}"
