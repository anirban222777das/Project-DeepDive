import gymnasium as gym
import ale_py
from src.preprocessing import AtariPreprocessor
from src.network import SeaquestCNN
from src.agent import DQNAgent
from src.replay_buffer import ReplayBuffer

def main():
    print("--- Phase 6 Replay Buffer Inspection ---")
    
    # 1. Setup minimal integration components
    env = gym.make("ALE/Seaquest-v5")
    processor = AtariPreprocessor()
    network = SeaquestCNN(num_actions=18)
    agent = DQNAgent(action_count=18, model=network)
    
    # Tiny capacity to demonstrate overflow easily
    buffer = ReplayBuffer(capacity=50)
    
    raw_obs, _ = env.reset()
    state = processor.reset(raw_obs)
    
    print("\nSimulating 20 steps of interaction...")
    
    for _ in range(20):
        # We use a mix of random and greedy
        action = agent.select_action(state, epsilon=0.5)
        
        raw_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = processor.step(raw_obs)
        
        # Store in experience replay
        buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            raw_obs, _ = env.reset()
            state = processor.reset(raw_obs)
            
    print(f"Total experiences added to buffer: {len(buffer)}")
    
    print("\n--- Sampling a Batch of 8 ---")
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=8)
    
    print(f"States shape: {states.shape}, dtype: {states.dtype}")
    print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"Rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
    print(f"Next States shape: {next_states.shape}, dtype: {next_states.dtype}")
    print(f"Dones shape: {dones.shape}, dtype: {dones.dtype}")
    
    print("\nSampled Actions Array:")
    print(actions)
    
    print("\nSampled Rewards Array:")
    print(rewards)
    
    print("\nSampled Dones Array:")
    print(dones)

    env.close()
    print("\nInspection complete! Notice that the shapes are neatly batched NumPy arrays.")

if __name__ == "__main__":
    main()
