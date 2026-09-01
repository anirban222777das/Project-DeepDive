import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling historical environment transitions.
    Designed for memory efficiency: states are stored exactly as received (uint8).
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # We use a deque with a maxlen. When full, appending automatically removes the oldest item.
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.
        State arrays are explicitly copied to guarantee immutability if the environment
        mutates the original arrays externally.
        """
        # Ensure deep copies of numpy arrays so external mutation doesn't corrupt history
        state_copy = np.copy(state)
        next_state_copy = np.copy(next_state)
        
        # Store basic Python types where applicable to save overhead
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
        Randomly samples a batch of experiences.
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough experiences to sample. Buffer has {len(self.buffer)}, requested {batch_size}.")

        # Randomly sample unique experiences
        batch = random.sample(self.buffer, batch_size)

        # Unpack the list of tuples into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert back to contiguous numpy arrays for network processing
        # States are maintained as uint8
        return (
            np.array(states, dtype=np.uint8),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.uint8),
            np.array(dones, dtype=bool)
        )

    def __len__(self):
        return len(self.buffer)
