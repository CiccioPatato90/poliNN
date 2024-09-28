import torch
from torch import nn

class FeedForwardNet(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[13, 17], num_classes=None):
        super(FeedForwardNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])     # Input to first hidden layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]) # First hidden to second hidden
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)     # Second hidden to output layer
        #self.relu = nn.ReLU()
        self.tansig = nn.Tanh()  # Tansig activation function (hyperbolic tangent)

    def forward(self, x):
        x = self.tansig(self.fc1(x))
        x = self.tansig(self.fc2(x))
        x = self.fc3(x)  # No activation function here if using CrossEntropyLoss
        return x