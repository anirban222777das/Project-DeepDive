import torch
import numpy as np
from src.recurrent_network import SeaquestDRQN
from src.recurrent_replay_buffer import RecurrentReplayBuffer
from src.drqn import DRQN
from src.recurrent_agent import DRQNAgent

def test_recurrent_network_shape():
    model = SeaquestDRQN(input_channels=4, num_actions=18, lstm_hidden_size=256)
    
    B, L, C, H, W = 2, 8, 4, 84, 84
    x = torch.randint(0, 256, (B, L, C, H, W), dtype=torch.uint8)
    
    q_vals, hidden_state = model(x)
    
    assert q_vals.shape == (2, 8, 18)
    assert hidden_state is not None
    assert hidden_state[0].shape == (1, 2, 256)
    
    q_vals_2, hidden_state_2 = model(x, hidden_state)
    assert q_vals_2.shape == (2, 8, 18)

def test_recurrent_replay_buffer():
    buffer = RecurrentReplayBuffer(capacity=100, seq_len=4)
    
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    for i in range(6):
        done = (i == 5)
        buffer.add(state, i, i * 1.0, state, done)
        
    assert len(buffer) == 6
    
    states, actions, rewards, next_states, dones = buffer.sample(2)
    
    assert states.shape == (2, 4, 4, 84, 84)
    assert actions.shape == (2, 4)
    assert rewards.shape == (2, 4)
    assert next_states.shape == (2, 4, 4, 84, 84)
    assert dones.shape == (2, 4)
    
    for i in range(2):
        assert not np.any(dones[i, :-1])

def test_recurrent_agent_hidden_state():
    model = SeaquestDRQN(num_actions=18, lstm_hidden_size=64)
    agent = DRQNAgent(18, model, device="cpu")
    
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    
    assert agent.hidden_state is None
    
    agent.select_action(state, epsilon=0.0)
    assert agent.hidden_state is not None
    
    h1 = agent.hidden_state[0].clone()
    
    agent.select_action(state, epsilon=0.0)
    h2 = agent.hidden_state[0].clone()
    
    assert not torch.allclose(h1, h2)
    
    agent.reset_hidden_state()
    assert agent.hidden_state is None

def test_drqn_bellman_update():
    drqn = DRQN(num_actions=18, device="cpu", learning_rate=1e-3)
    
    B, L, C, H, W = 2, 4, 4, 84, 84
    states = np.zeros((B, L, C, H, W), dtype=np.uint8)
    next_states = np.zeros((B, L, C, H, W), dtype=np.uint8)
    actions = np.zeros((B, L), dtype=np.int64)
    rewards = np.ones((B, L), dtype=np.float32)
    dones = np.zeros((B, L), dtype=bool)
    
    dones[0, -1] = True
    
    loss, mean_q, max_q, grad_norm = drqn.update(states, actions, rewards, next_states, dones, gamma=0.99)
    
    assert isinstance(loss, float)
    assert grad_norm >= 0.0

if __name__ == "__main__":
    test_recurrent_network_shape()
    test_recurrent_replay_buffer()
    test_recurrent_agent_hidden_state()
    test_drqn_bellman_update()
    print("All DRQN tests passed!")
