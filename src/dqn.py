import torch
import torch.nn as nn
import torch.optim as optim
from src.network import SeaquestCNN

class DQN:
    """
    The Core Mathematical DQN Engine.
    Manages the Online Network (θ) and Target Network (θ⁻).
    Calculates the Bellman targets and current Q-values.
    Now includes Phase 8: One-step Optimization Update.
    """
    def __init__(self, action_count, device=None, gamma=0.99, learning_rate=1e-4, double_dqn=False):
        self.action_count = action_count
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.double_dqn = double_dqn
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        # 1. Initialize Online Network
        self.online_network = SeaquestCNN(num_actions=action_count).to(self.device)
        
        # 2. Initialize Target Network
        self.target_network = SeaquestCNN(num_actions=action_count).to(self.device)
        
        # Synchronize immediately
        self.update_target_network()
        
        # Disable gradient tracking on the target network permanently for safety
        for param in self.target_network.parameters():
            param.requires_grad = False
            
        # 3. Setup Optimizer and Loss
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
            
    def update_target_network(self):
        """
        Performs a hard update: Target Network θ⁻ ← Online Network θ
        """
        self.target_network.load_state_dict(self.online_network.state_dict())
        
    def get_current_q_values(self, states, actions):
        """
        Computes Q_online(s, a).
        states: (B, 4, 84, 84) tensor
        actions: (B,) tensor
        Returns: (B,) tensor representing the predicted Q-value for the chosen actions.
        """
        # Get Q-values for all actions from online network
        q_values_all = self.online_network(states) # Shape: (B, 18)
        
        # Gather the Q-values corresponding to the specific actions taken
        current_q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        return current_q_values
        
    def compute_bellman_targets(self, rewards, next_states, dones):
        """
        Computes the target y = r + γ * (1 - done) * max_a' Q_target(s', a').
        Does not track gradients.
        
        rewards: (B,) tensor
        next_states: (B, 4, 84, 84) tensor
        dones: (B,) boolean tensor
        Returns: (B,) tensor representing the mathematical Bellman targets.
        """
        with torch.no_grad():
            if self.double_dqn:
                # 1. Use ONLINE network to SELECT the best action
                next_q_online = self.online_network(next_states) # Shape: (B, 18)
                next_actions = next_q_online.argmax(dim=1, keepdim=True) # Shape: (B, 1)
                
                # 2. Use TARGET network to EVALUATE the selected action
                next_q_target_all = self.target_network(next_states) # Shape: (B, 18)
                next_q = next_q_target_all.gather(1, next_actions).squeeze(1) # Shape: (B,)
            else:
                # Standard Vanilla DQN
                # Pass next_states through TARGET network
                next_q_target_all = self.target_network(next_states) # Shape: (B, 18)
                
                # Find the maximum predicted action value for each next state
                next_q, _ = next_q_target_all.max(dim=1) # Shape: (B,)
            
            # Apply terminal state mask. If done is True, future contribution is 0.
            targets = rewards + self.gamma * next_q * (~dones)
            
        return targets

    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Executes exactly ONE optimization update (loss computation, backprop, optimizer step).
        states: (B, 4, 84, 84) uint8/float tensor
        actions: (B,) int64 tensor
        rewards: (B,) float tensor
        next_states: (B, 4, 84, 84) uint8/float tensor
        dones: (B,) bool tensor
        
        Returns:
            dict containing training diagnostics.
        """
        # Ensure correct types and device placement
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 1. Compute Current Q-Values Q_online(s, a)
        current_q = self.get_current_q_values(states, actions)
        
        # 2. Compute Bellman Targets using Target Network
        # Targets are treated as constant values (requires_grad is already False)
        targets = self.compute_bellman_targets(rewards, next_states, dones)
        
        # 3. Calculate Huber Loss
        loss = self.loss_fn(current_q, targets)
        
        # 4. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # 5. Optimization step
        self.optimizer.step()
        
        # 6. Diagnostics
        td_error = (targets - current_q).detach()
        diagnostics = {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
            "mean_target": targets.mean().item(),
            "mean_td_error": td_error.mean().item()
        }
        
        return diagnostics
