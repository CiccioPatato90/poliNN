import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[15, 20], num_classes=4):
        super(FeedForwardNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_sizes[0])
        self.act = nn.ReLU()
        
        self.hidden2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        
        self.output = nn.Linear(hidden_sizes[1], num_classes)  # Logits as outputs

        # No softmax in the model itself during training if using CrossEntropyLoss

    def forward(self, x):
        x = self.act(self.hidden1(x))
        
        x = self.act(self.hidden2(x))
        
        x = self.output(x)  # Logits output (do not apply softmax here)
        return x
    

    # Step 4: Define the neural network model
class ModelGPT(nn.Module):
    def __init__(self, input_dim, num_clusters):
        super(ModelGPT, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.hidden2 = nn.Linear(64, 32)         # Second hidden layer
        self.output = nn.Linear(32, num_clusters)  # Output layer for clusters

    def forward(self, x):
        x = torch.relu(self.hidden1(x))  # Use ReLU activation
        x = torch.relu(self.hidden2(x))  # Use ReLU activation
        x = self.output(x)  # Output raw logits for CrossEntropyLoss
        return x