import os
import torch
import numpy as np
from collections import deque
import gymnasium as gym

from src.config import TrainingConfig
from src.train import calculate_epsilon, save_checkpoint
from src.dqn import DQN
from src.agent import DQNAgent
from src.preprocessing import AtariPreprocessor
from src.evaluate import evaluate

def get_params_copy(network):
    return torch.cat([p.view(-1).detach().clone() for p in network.parameters()])

def test_epsilon_decay():
    config = TrainingConfig(
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=1000
    )
    
    # Test Boundaries
    assert calculate_epsilon(0, config) == 1.0
    assert calculate_epsilon(500, config) == 0.55
    assert calculate_epsilon(1000, config) == 0.1
    
    # Test capping at the end
    assert calculate_epsilon(5000, config) == 0.1
    print("Test: Epsilon Decay Logic - Passed")

def test_checkpointing():
    dqn = DQN(18, device="cpu")
    config = TrainingConfig(seed=99)
    
    # Mutate slightly to ensure it's not just the default zeroes
    with torch.no_grad():
        dqn.online_network.fc2.bias.fill_(3.14)
        dqn.target_network.fc2.bias.fill_(2.71)
        
    path = "/tmp/dqn_test_checkpoint.pt"
    save_checkpoint(path, dqn, 500, 10, 0.5, 125, config)
    
    # Load into a new DQN
    dqn_new = DQN(18, device="cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    dqn_new.online_network.load_state_dict(checkpoint['online_network_state_dict'])
    dqn_new.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    dqn_new.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    assert checkpoint['step'] == 500
    assert checkpoint['episode'] == 10
    assert checkpoint['epsilon'] == 0.5
    
    assert torch.allclose(dqn_new.online_network.fc2.bias, torch.tensor(3.14))
    assert torch.allclose(dqn_new.target_network.fc2.bias, torch.tensor(2.71))
    
    os.remove(path)
    print("Test: Checkpoint Serialization - Passed")

def test_evaluation_boundary():
    env = gym.make("ALE/Seaquest-v5")
    preprocessor = AtariPreprocessor()
    
    dqn = DQN(action_count=18, device="cpu")
    agent = DQNAgent(action_count=18, model=dqn.online_network, device="cpu")
    
    initial_online = get_params_copy(dqn.online_network)
    initial_target = get_params_copy(dqn.target_network)
    
    # Run evaluation for a tiny amount of steps to ensure it doesn't crash or update weights
    mean_r, std_r, mean_l = evaluate(env, agent, preprocessor, episodes=1, max_steps=10)
    
    final_online = get_params_copy(dqn.online_network)
    final_target = get_params_copy(dqn.target_network)
    
    assert torch.equal(initial_online, final_online), "Evaluation illegally modified online weights!"
    assert torch.equal(initial_target, final_target), "Evaluation illegally modified target weights!"
    
    env.close()
    print("Test: Evaluation Safety Boundary - Passed")

def run_all_tests():
    print("Running Full Training Integration Tests...")
    test_epsilon_decay()
    test_checkpointing()
    test_evaluation_boundary()
    print("All training integration tests passed successfully!\n")

if __name__ == "__main__":
    run_all_tests()
