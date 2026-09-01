import random
import numpy as np
from collections import deque

class RecurrentReplayBuffer:
    """
    Sequence-Aware Experience Replay Buffer for DRQN.
    Stores flat transitions but samples valid temporal sequences of length `seq_len`.
    """
    def __init__(self, capacity: int, seq_len: int = 8):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.
        """
        state_copy = np.copy(state)
        next_state_copy = np.copy(next_state)
        
        experience = (
            state_copy, 
            int(action), 
            float(reward), 
            next_state_copy, 
            bool(done)
        )
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Randomly samples a batch of valid sequences.
        A valid sequence is one of length `seq_len` that does not cross an episode boundary
        (i.e., no `done=True` in the first `seq_len - 1` transitions).
        """
        if len(self.buffer) < batch_size + self.seq_len:
            raise ValueError(f"Not enough experiences to sample a batch of sequences.")

        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []

        valid_count = 0
        max_idx = len(self.buffer) - self.seq_len
        
        while valid_count < batch_size:
            idx = random.randint(0, max_idx)
            
            # Check if this sequence crosses an episode boundary
            is_valid = True
            for offset in range(self.seq_len - 1):
                if self.buffer[idx + offset][4]:  # index 4 is 'done'
                    is_valid = False
                    break
                    
            if is_valid:
                seq_states = []
                seq_actions = []
                seq_rewards = []
                seq_next_states = []
                seq_dones = []
                
                for offset in range(self.seq_len):
                    exp = self.buffer[idx + offset]
                    seq_states.append(exp[0])
                    seq_actions.append(exp[1])
                    seq_rewards.append(exp[2])
                    seq_next_states.append(exp[3])
                    seq_dones.append(exp[4])
                    
                states_batch.append(seq_states)
                actions_batch.append(seq_actions)
                rewards_batch.append(seq_rewards)
                next_states_batch.append(seq_next_states)
                dones_batch.append(seq_dones)
                
                valid_count += 1

        return (
            np.array(states_batch, dtype=np.uint8),
            np.array(actions_batch, dtype=np.int64),
            np.array(rewards_batch, dtype=np.float32),
            np.array(next_states_batch, dtype=np.uint8),
            np.array(dones_batch, dtype=bool)
        )

    def __len__(self):
        return len(self.buffer)
