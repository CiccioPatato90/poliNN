import torch
import torch.nn as nn
from netdebugger import torch_utils as utils

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim, capacity):
        super(FeedForwardNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = capacity

        # Generate the unit sequence for the layer sizes
        unit_sequence = utils.unit_sequence(self.input_dim, self.output_dim, self.n_hidden)

        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()

        # Set up the layers based on the unit sequence
        for ind in range(len(unit_sequence) - 1):
            layer = nn.Linear(unit_sequence[ind], unit_sequence[ind + 1])
            self.layers.append(layer)
        
    def forward(self, data):
        # Extract the Tensor from data.x.x
        #x = data.x # Since data.x is a DataBatch, and data.x.x is a Tensor
        x = data

        # Pass input through all layers except the last one with ReLU activation
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.relu(x)
        
        # Apply the final layer without activation (logits for multi-class classification)
        x = self.layers[-1](x)
        
        return x  # Return raw logits suitable for multi-class classification
    

class ClusteringModel(nn.Module):

    def __init__(self, input_dim, output_dim, capacity):
            super(ClusteringModel, self).__init__()
            
            # Generate the unit sequence for the layers using your function
            self.unit_sequence = utils.unit_sequence(input_dim, output_dim, capacity)
            self.layers = nn.ModuleList()
            self.relu = nn.ReLU()

            # Dynamically create the layers based on the unit sequence
            for i in range(len(self.unit_sequence) - 1):
                layer = nn.Linear(self.unit_sequence[i], self.unit_sequence[i + 1])
                self.layers.append(layer)

            # Adjust cluster centers to match the size of the last hidden layer
            print(self.unit_sequence)
            print(self.unit_sequence[-2])
            #exit()
            self.cluster_centers = nn.Parameter(torch.randn(output_dim, self.unit_sequence[-1]))
        
    def forward(self, x):
        # Pass the data through the dynamically created layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation function to all layers except the last
                x = self.relu(x)

        # Ensure that the dimensions of x and cluster centers match for torch.cdist
        distances = torch.cdist(x, self.cluster_centers)

        return distances, x  # Return distances to cluster centers and the learned features
