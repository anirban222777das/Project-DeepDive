import os
import torch
import pytest
from src.config import TrainingConfig
from src.dqn import DQN
from src.train import save_checkpoint

def test_checkpoint_resume_structure(tmp_path):
    """Test that a checkpoint saves correctly and contains all necessary restore state."""
    # 1. Setup mock checkpoint
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = str(checkpoint_dir / "test_ckpt.pt")
    
    config = TrainingConfig(
        device="cpu",
        total_steps=100000,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    dqn = DQN(action_count=18, device="cpu")
    
    # Simulate a trained state
    start_step = 100000
    episodes = 200
    epsilon = 0.5
    learning_updates = 20000
    
    # 2. Save
    save_checkpoint(checkpoint_path, dqn, start_step, episodes, epsilon, learning_updates, config)
    
    # 3. Verify file exists
    assert os.path.exists(checkpoint_path)
    
    # 4. Load & verify structure
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    assert 'online_network_state_dict' in checkpoint
    assert 'target_network_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'step' in checkpoint
    assert 'episode' in checkpoint
    assert 'epsilon' in checkpoint
    assert 'learning_updates' in checkpoint
    
    assert checkpoint['step'] == 100000
    assert checkpoint['episode'] == 200
    assert checkpoint['learning_updates'] == 20000
