import pytest
import numpy as np
from src.replay_buffer import ReplayBuffer

def create_mock_state(fill_value=0):
    return np.full((4, 84, 84), fill_value, dtype=np.uint8)

def test_buffer_capacity():
    buffer = ReplayBuffer(capacity=5)
    
    for i in range(10):
        buffer.add(
            state=create_mock_state(i),
            action=i,
            reward=i * 1.5,
            next_state=create_mock_state(i+1),
            done=False
        )
        
    # Capacity is 5, so we should only have 5 experiences
    assert len(buffer) == 5
    
    # The oldest should have been dropped. The buffer should hold actions 5, 6, 7, 8, 9
    actions_in_buffer = [exp[1] for exp in buffer.buffer]
    assert actions_in_buffer == [5, 6, 7, 8, 9]
    print("Test Capacity: Passed")

def test_sample_shapes_and_types():
    buffer = ReplayBuffer(capacity=100)
    
    for i in range(50):
        buffer.add(
            state=create_mock_state(),
            action=0,
            reward=1.0,
            next_state=create_mock_state(),
            done=False
        )
        
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=32)
    
    assert states.shape == (32, 4, 84, 84)
    assert states.dtype == np.uint8
    
    assert actions.shape == (32,)
    assert actions.dtype == np.int64
    
    assert rewards.shape == (32,)
    assert rewards.dtype == np.float32
    
    assert next_states.shape == (32, 4, 84, 84)
    assert next_states.dtype == np.uint8
    
    assert dones.shape == (32,)
    assert dones.dtype == bool
    
    print("Test Shapes & Dtypes: Passed")

def test_random_sampling():
    buffer = ReplayBuffer(capacity=100)
    
    # We add 100 distinct experiences, tracking them by reward
    for i in range(100):
        buffer.add(create_mock_state(), 1, float(i), create_mock_state(), False)
        
    # Sample 10 items twice
    _, _, rewards1, _, _ = buffer.sample(10)
    _, _, rewards2, _, _ = buffer.sample(10)
    
    # Very high probability that the batches are different and not sequentially ordered
    assert not np.array_equal(rewards1, rewards2)
    
    # Ensure they are not strictly the last 10 elements
    last_10 = np.arange(90, 100, dtype=np.float32)
    assert not np.array_equal(np.sort(rewards1), last_10)
    print("Test Random Sampling: Passed")

def test_state_immutability():
    buffer = ReplayBuffer(capacity=10)
    
    original_state = create_mock_state(0)
    
    buffer.add(original_state, 1, 1.0, original_state, False)
    
    # Mutate the original array
    original_state.fill(255)
    
    states, _, _, _, _ = buffer.sample(1)
    
    # The stored array should still be filled with 0, not 255
    assert np.all(states[0] == 0)
    print("Test State Immutability: Passed")

def test_empty_and_insufficient_buffer():
    buffer = ReplayBuffer(capacity=10)
    
    # Empty
    try:
        buffer.sample(5)
        assert False, "Should have raised ValueError on empty buffer"
    except ValueError:
        pass

    # Insufficient
    buffer.add(create_mock_state(), 1, 1.0, create_mock_state(), False)
    try:
        buffer.sample(5)
        assert False, "Should have raised ValueError on insufficient buffer"
    except ValueError:
        pass
        
    print("Test Empty Buffer Logic: Passed")

def run_all_tests():
    print("Running Replay Buffer Unit Tests...")
    test_buffer_capacity()
    test_sample_shapes_and_types()
    test_random_sampling()
    test_state_immutability()
    test_empty_and_insufficient_buffer()
    print("All replay buffer tests passed successfully!\n")

if __name__ == "__main__":
    run_all_tests()
