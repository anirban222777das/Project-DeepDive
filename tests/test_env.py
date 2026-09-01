import gymnasium as gym
import ale_py
import numpy as np

def test_environment():
    print("Test 1: Registering ALE and Creating Environment...")
    gym.register_envs(ale_py)
    env = gym.make("ALE/Seaquest-v5")
    assert env is not None
    print("Test 1 Passed.")

    print("Test 2 & 3: Testing reset() and Observation...")
    observation, info = env.reset()
    assert observation is not None
    assert info is not None
    
    # Check observation shape and dtype
    obs_space = env.observation_space
    assert observation.shape == obs_space.shape
    assert observation.dtype == obs_space.dtype
    
    # Specific checks based on Seaquest
    if obs_space.shape == (210, 160, 3):
        assert observation.shape == (210, 160, 3)
    if obs_space.dtype == np.uint8:
        assert observation.dtype == np.uint8
        
    print("Test 2 & 3 Passed.")

    print("Test 4: Testing valid random action...")
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    
    # Check 0 <= action < n
    if isinstance(env.action_space, gym.spaces.Discrete):
        assert 0 <= action < env.action_space.n
        
    print("Test 4 Passed.")

    print("Test 5 & 6: Testing step() and returned values...")
    step_result = env.step(action)
    assert len(step_result) == 5, f"Expected 5 return values, got {len(step_result)}"
    next_obs, reward, terminated, truncated, step_info = step_result
    assert next_obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(step_info, dict)
    print("Test 5 & 6 Passed.")

    print("Test 7: Testing environment close...")
    env.close()
    print("Test 7 Passed.")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_environment()
