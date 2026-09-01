import torch
import numpy as np
import random

class DRQNAgent:
    """
    Agent class for Recurrent Double DQN.
    Manages epsilon-greedy action selection and maintains the LSTM hidden state
    across consecutive environment steps.
    """
    def __init__(self, num_actions: int, model: torch.nn.Module, device: str = "cpu"):
        self.num_actions = num_actions
        self.model = model
        self.device = device
        self.hidden_state = None

    def reset_hidden_state(self):
        """
        Resets the internal hidden state of the LSTM.
        Must be called at the beginning of each episode.
        """
        self.hidden_state = None

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        Updates the internal hidden state.
        
        Args:
            state: The current preprocessed state array of shape (C, H, W)
            epsilon: The probability of taking a random action
        Returns:
            action_id: Integer representing the chosen action
        """
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            # We still need to do a forward pass to update the hidden state!
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0).unsqueeze(0)
                _, self.hidden_state = self.model(state_tensor, self.hidden_state)
            return random.randint(0, self.num_actions - 1)
            
        # Greedy exploitation
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0).unsqueeze(0)
            q_values, self.hidden_state = self.model(state_tensor, self.hidden_state)
            action = torch.argmax(q_values[0, 0]).item()
            
        return action
