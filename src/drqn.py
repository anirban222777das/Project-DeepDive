import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.recurrent_network import SeaquestDRQN

class DRQN:
    """
    Core Recurrent Double DQN learning algorithm logic.
    Maintains online and target networks and computes sequential Bellman updates.
    """
    def __init__(self, num_actions: int, device: str = "cpu", learning_rate: float = 1e-4):
        self.num_actions = num_actions
        self.device = device
        
        # Initialize networks
        self.online_network = SeaquestDRQN(num_actions=num_actions).to(self.device)
        self.target_network = SeaquestDRQN(num_actions=num_actions).to(self.device)
        
        # Target network is strictly for evaluation (frozen)
        self.target_network.load_state_dict(self.online_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)

    def compute_bellman_targets(self, rewards, next_states, dones, gamma):
        with torch.no_grad():
            # 1. Action selection using the online network
            next_q_values_online, _ = self.online_network(next_states)
            best_next_actions = torch.argmax(next_q_values_online, dim=2)
            
            # 2. Action evaluation using the target network
            next_q_values_target, _ = self.target_network(next_states)
            next_q_target = next_q_values_target.gather(2, best_next_actions.unsqueeze(2)).squeeze(2)
            
            target_q = rewards + gamma * next_q_target * (1.0 - dones.float())
            
        return target_q

    def update(self, states, actions, rewards, next_states, dones, gamma=0.99):
        self.online_network.train()
        
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)
        
        # 1. Compute current Q-values
        q_values, _ = self.online_network(states)
        current_q = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
        
        # 2. Compute Bellman targets using Double DQN logic
        target_q = self.compute_bellman_targets(rewards, next_states, dones, gamma)
        
        # 3. Compute Huber Loss over the entire sequence
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 4. Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        
        # Optional: gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        self.optimizer.step()
        
        with torch.no_grad():
            mean_q = current_q.mean().item()
            max_q = current_q.max().item()
            
            grad_norm = 0.0
            for p in self.online_network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            
        return loss.item(), mean_q, max_q, grad_norm

    def update_target_network(self):
        """Copies weights from the online network to the target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())
