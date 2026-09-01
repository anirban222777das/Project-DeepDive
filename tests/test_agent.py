import torch
import numpy as np
from src.network import SeaquestCNN
from src.agent import DQNAgent

def test_agent_creation():
    model = SeaquestCNN(num_actions=18)
    agent = DQNAgent(action_count=18, model=model, device="cpu")
    assert agent is not None
    print("Test 1 - Agent Creation: Passed")

def test_greedy_action():
    model = SeaquestCNN(num_actions=18)
    agent = DQNAgent(action_count=18, model=model, device="cpu")
    
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    action = agent.select_greedy_action(state)
    
    assert isinstance(action, int)
    assert 0 <= action < 18
    print("Test 2 - Greedy Action: Passed")

def test_random_action():
    model = SeaquestCNN(num_actions=18)
    agent = DQNAgent(action_count=18, model=model, device="cpu")
    
    for _ in range(100):
        action = agent.select_random_action()
        assert 0 <= action < 18
    print("Test 3 - Random Action: Passed")

def test_epsilon_actions():
    model = SeaquestCNN(num_actions=18)
    agent = DQNAgent(action_count=18, model=model, device="cpu")
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    
    # Epsilon = 1.0 (Random)
    valid = True
    for _ in range(20):
        action = agent.select_action(state, epsilon=1.0)
        if not (0 <= action < 18):
            valid = False
    assert valid
    print("Test 4 - Epsilon = 1.0: Passed")
    
    # Epsilon = 0.0 (Greedy)
    # The greedy action for this fixed state should be deterministic
    greedy_expected = agent.select_greedy_action(state)
    for _ in range(20):
        action = agent.select_action(state, epsilon=0.0)
        assert action == greedy_expected
    print("Test 5 - Epsilon = 0.0: Passed")

def test_batch_handling():
    model = SeaquestCNN(num_actions=18)
    batch_state = torch.zeros((8, 4, 84, 84), dtype=torch.uint8)
    
    # Verify CNN can process batches natively
    out = model(batch_state)
    assert out.shape == (8, 18)
    
    # Verify agent handles single state correctly (adds batch dim automatically)
    agent = DQNAgent(action_count=18, model=model, device="cpu")
    single_state = np.zeros((4, 84, 84), dtype=np.uint8)
    q_values = agent._get_q_values(single_state)
    assert q_values.shape == (1, 18)
    print("Test 6 - Batch Handling: Passed")

def test_device_support():
    model_cpu = SeaquestCNN(num_actions=18)
    agent_cpu = DQNAgent(action_count=18, model=model_cpu, device="cpu")
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    
    _ = agent_cpu.select_greedy_action(state)
    print("Test 7 - Device CPU: Passed")
    
    if torch.backends.mps.is_available():
        model_mps = SeaquestCNN(num_actions=18)
        agent_mps = DQNAgent(action_count=18, model=model_mps, device="mps")
        _ = agent_mps.select_greedy_action(state)
        print("Test 7 - Device MPS: Passed")
    else:
        print("Test 7 - Device MPS: Skipped (MPS not available)")

def test_no_weight_modification():
    model = SeaquestCNN(num_actions=18)
    agent = DQNAgent(action_count=18, model=model, device="cpu")
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    
    # Get a copy of the initial weights
    initial_params = [p.clone() for p in agent.model.parameters()]
    
    # Perform several action selections
    for _ in range(50):
        agent.select_action(state, epsilon=0.5)
        
    # Check that weights haven't changed
    for initial, current in zip(initial_params, agent.model.parameters()):
        assert torch.equal(initial, current), "Model parameters changed during inference!"
        assert current.grad is None, "Gradients were calculated during inference!"
        
    print("Test 8 - No Weight Modification: Passed")

def run_all_tests():
    print("Running Agent Unit Tests...")
    test_agent_creation()
    test_greedy_action()
    test_random_action()
    test_epsilon_actions()
    test_batch_handling()
    test_device_support()
    test_no_weight_modification()
    print("All agent tests passed successfully!\n")

if __name__ == "__main__":
    run_all_tests()
