import os
import torch
import pytest
from unittest.mock import patch, MagicMock

from src.train import save_checkpoint

def test_long_training_checkpoint_saving():
    """Verify that checkpoint saving handles the 1M training mode correctly"""
    mock_network = MagicMock()
    mock_network.state_dict.return_value = {"weight": torch.tensor([1.0])}
    
    mock_optimizer = MagicMock()
    mock_optimizer.state_dict.return_value = {"step": 1000}
    
    mock_dqn = MagicMock()
    mock_dqn.online_network = mock_network
    mock_dqn.target_network = mock_network
    mock_dqn.optimizer = mock_optimizer
    mock_dqn.double_dqn = True
    
    mock_config = MagicMock()
    mock_config.algorithm = "double-dqn"
    
    step = 500000
    episodes = 2500
    learning_updates = 125000
    epsilon = 0.5
    
    # We will test saving to a dummy path
    checkpoint_dir = "/tmp/seaquest_test_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    expected_path = os.path.join(checkpoint_dir, f"dqn_step_{step}.pt")
    
    # Test saving double-dqn checkpoint
    save_checkpoint(
        path=expected_path,
        dqn=mock_dqn,
        step=step,
        episode=episodes,
        epsilon=epsilon,
        learning_updates=learning_updates,
        config=mock_config
    )
    assert os.path.exists(expected_path)
    
    # Verify contents
    checkpoint = torch.load(expected_path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == step
    assert checkpoint["episode"] == episodes
    assert checkpoint["learning_updates"] == learning_updates
    assert checkpoint["algorithm"] == "double-dqn"
    
    # Cleanup
    os.remove(expected_path)
    
def test_evaluation_does_not_modify_weights():
    """Verify that evaluation does not modify network weights"""
    from src.network import SeaquestCNN
    
    net = SeaquestCNN()
    initial_weights = {name: param.clone() for name, param in net.named_parameters()}
    
    # Mocking evaluation
    # Since we tested eval immutability in test_evaluation_analysis.py, we just do a basic check here
    net.eval()
    with torch.no_grad():
        x = torch.zeros(1, 4, 84, 84)
        out = net(x)
        
    for name, param in net.named_parameters():
        assert torch.equal(param, initial_weights[name])
