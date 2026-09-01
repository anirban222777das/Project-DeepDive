import random
import torch
import numpy as np

class DQNAgent:
    """
    DQN Agent that wraps the CNN model to provide action selection functionality.
    Currently supports greedy, random, and epsilon-greedy action selection.
    No learning or training mechanisms are implemented yet.
    """
    def __init__(self, action_count, model, device=None):
        self.action_count = action_count
        self.model = model
        
        # Auto-detect device if not provided
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        # Set model to evaluation mode (good practice, though our model has no dropout/batchnorm)
        self.model.eval()

    def select_random_action(self):
        """Selects a random valid action."""
        return random.randrange(self.action_count)

    def select_greedy_action(self, state):
        """
        Passes the state through the CNN and selects the action with the highest Q-value.
        State should be a numpy array of shape (4, 84, 84).
        """
        q_values = self._get_q_values(state)
        # Select action with max Q-value
        action = q_values.argmax(dim=1).item()
        return action
        
    def select_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.
        With probability epsilon, selects a random action.
        Otherwise, selects the greedy action.
        """
        if random.random() < epsilon:
            return self.select_random_action()
        else:
            return self.select_greedy_action(state)

    def _get_q_values(self, state):
        """
        Helper method to run a forward pass on a single state without tracking gradients.
        Expects state shape (4, 84, 84) and returns Q-values shape (1, action_count).
        """
        # Ensure state is a torch tensor
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.uint8)
        else:
            state_tensor = state
            
        # Add batch dimension: (4, 84, 84) -> (1, 4, 84, 84)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
            
        # Move to the appropriate device
        state_tensor = state_tensor.to(self.device)
        
        # Perform inference with no gradient calculation
        with torch.no_grad():
            q_values = self.model(state_tensor)
            
        return q_values

