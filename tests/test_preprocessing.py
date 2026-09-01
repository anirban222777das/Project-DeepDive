import numpy as np
from src.preprocessing import AtariPreprocessor

def test_preprocessing_initialization():
    # Test 7 - Reset logic
    processor = AtariPreprocessor(target_size=(84, 84), stack_size=4)
    # Fake raw observation
    fake_obs = np.ones((210, 160, 3), dtype=np.uint8) * 100
    
    state = processor.reset(fake_obs)
    
    # Test 5 & 6 - Stack size and shape
    assert state.shape == (4, 84, 84), f"Expected shape (4, 84, 84), got {state.shape}"
    
    # Test 3 & 4 - Dtype and value range
    assert state.dtype == np.uint8, f"Expected dtype uint8, got {state.dtype}"
    assert np.min(state) >= 0 and np.max(state) <= 255, "Values out of bounds for uint8"
    
    # Check that all 4 frames in the stack are identical after reset
    assert np.array_equal(state[0], state[1])
    assert np.array_equal(state[1], state[2])
    assert np.array_equal(state[2], state[3])
    
    print("Test Initialization / Reset: Passed")
    print("Test Shape & Stack (4x84x84): Passed")
    print("Test Dtype & Bounds (uint8): Passed")

def test_rolling_history():
    # Test 8 - Rolling history
    processor = AtariPreprocessor(target_size=(84, 84), stack_size=4)
    
    # Create 5 distinct fake frames
    frames = [np.full((210, 160, 3), fill_value=i*50, dtype=np.uint8) for i in range(1, 6)]
    
    # Reset with Frame 1 (Stack: 1, 1, 1, 1)
    state = processor.reset(frames[0])
    
    # Step with Frame 2 (Stack: 1, 1, 1, 2)
    state = processor.step(frames[1])
    # The newest frame is at the end of the stack (index 3)
    # Wait, processor appends to deque. np.stack keeps order. Index 3 is newest.
    # The oldest is index 0.
    
    # Step with Frame 3 (Stack: 1, 1, 2, 3)
    state = processor.step(frames[2])
    
    # Step with Frame 4 (Stack: 1, 2, 3, 4)
    state = processor.step(frames[3])
    
    # Step with Frame 5 (Stack: 2, 3, 4, 5)
    state = processor.step(frames[4])
    
    # Let's verify the stack content visually by checking unique values
    # Each processed frame should only contain one value because input was uniform
    # We expect the stack to represent frames 2, 3, 4, 5 in that order
    val_0 = np.unique(state[0])[0] # should be derived from frames[1]
    val_1 = np.unique(state[1])[0] # should be derived from frames[2]
    val_2 = np.unique(state[2])[0] # should be derived from frames[3]
    val_3 = np.unique(state[3])[0] # should be derived from frames[4]
    
    assert val_0 < val_1 < val_2 < val_3, "Frames are not rolling in the correct chronological order"
    
    print("Test Rolling History: Passed")

def test_grayscale_and_resize():
    # Test 1 & 2 - Grayscale and Resize logic internally
    processor = AtariPreprocessor(target_size=(84, 84), stack_size=4)
    
    # Create a red frame: (R=255, G=0, B=0)
    fake_obs = np.zeros((210, 160, 3), dtype=np.uint8)
    fake_obs[:, :, 0] = 255
    
    processed = processor._process_frame(fake_obs)
    
    # Shape should be 84x84 (grayscale, 2D array)
    assert processed.shape == (84, 84), f"Expected processed shape (84, 84), got {processed.shape}"
    
    # Luminance of pure red (255 * 0.2989) is approx 76
    assert 70 < np.unique(processed)[0] < 80, "Grayscale conversion is incorrect"
    
    print("Test Grayscale & Resize (84x84): Passed")

def run_all_tests():
    print("Running Preprocessing Unit Tests...")
    test_preprocessing_initialization()
    test_rolling_history()
    test_grayscale_and_resize()
    print("All preprocessing tests passed successfully!\n")

if __name__ == "__main__":
    run_all_tests()
