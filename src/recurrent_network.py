import torch
import torch.nn as nn
import torch.nn.functional as F

class SeaquestDRQN(nn.Module):
    """
    Recurrent Double DQN Convolutional Neural Network architecture.
    Expects input shape: (Batch_Size, Sequence_Length, Channels, Height, Width)
    e.g., (B, L, 4, 84, 84).
    Outputs a Q-value for each of the 18 actions for each step in the sequence.
    """
    def __init__(self, input_channels=4, num_actions=18, lstm_hidden_size=512):
        super(SeaquestDRQN, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        
        # Convolutional layers (same as baseline)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # The flattened output dimension from the conv layers is 64 * 7 * 7 = 3136
        self.cnn_out_size = 3136
        
        # LSTM layer
        # batch_first=True means input should be (Batch, Seq, Features)
        self.lstm = nn.LSTM(input_size=self.cnn_out_size, hidden_size=self.lstm_hidden_size, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=self.lstm_hidden_size, out_features=num_actions)

    def forward(self, x, hidden_state=None):
        """
        Forward pass of the DRQN.
        x: A torch.Tensor of shape (B, L, C, H, W) and dtype torch.uint8 or float
        hidden_state: Tuple of (h_n, c_n) from previous step, or None for initial state.
        Returns:
            out: Q-values of shape (B, L, num_actions)
            hidden_state: The new hidden state (h_n, c_n) after processing the sequence
        """
        B, L, C, H, W = x.shape
        
        # Flatten B and L for CNN processing: (B*L, C, H, W)
        x = x.view(B * L, C, H, W)
        
        # Pixel Normalization
        x = x.float() / 255.0
        
        # CNN forward
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten spatial dims: (B*L, 3136)
        x = x.reshape(x.size(0), -1)
        
        # Reshape back to sequence: (B, L, 3136)
        x = x.view(B, L, self.cnn_out_size)
        
        # LSTM forward
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        
        # Fully connected forward
        out = self.fc(lstm_out)
        
        return out, hidden_state

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
