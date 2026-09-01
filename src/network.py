import torch
import torch.nn as nn
import torch.nn.functional as F

class SeaquestCNN(nn.Module):
    """
    Classic DQN Convolutional Neural Network architecture for Atari environments.
    Expects input shape: (Batch_Size, Channels, Height, Width) -> (B, 4, 84, 84).
    Outputs a value for each of the 18 Seaquest actions.
    """
    def __init__(self, input_channels=4, num_actions=18):
        super(SeaquestCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Fully connected layers
        # The flattened output dimension from the conv layers is 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(in_features=3136, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        """
        Forward pass of the CNN.
        x: A torch.Tensor of shape (B, 4, 84, 84) and dtype torch.uint8
        """
        # Pixel Normalization:
        # Convert uint8 (0-255) from the replay buffer / preprocessor into float32 (0.0-1.0)
        # This keeps our replay buffer memory usage low, while providing the CNN with normalized inputs.
        x = x.float() / 255.0
        
        # Apply convolutions with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the spatial dimensions: (B, 64, 7, 7) -> (B, 3136)
        x = x.reshape(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        
        # Output layer representing the 18 action values (No activation applied here)
        out = self.fc2(x)
        
        return out

    def get_parameter_count(self):
        """Returns the total number of trainable parameters in the network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

