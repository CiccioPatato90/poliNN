import torch
import torch.nn as nn
#from netdebugger import torch_utils as utils
"""
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim, capacity):
        super(FeedForwardNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = capacity

        # Generate the unit sequence for the layer sizes
        unit_sequence = utils.unit_sequence(self.input_dim, self.output_dim, self.n_hidden)

        self.relu = nn.Softmax()
        self.layers = nn.ModuleList()

        # Set up the layers based on the unit sequence
        for ind in range(len(unit_sequence) - 1):
            layer = nn.Linear(unit_sequence[ind], unit_sequence[ind + 1])
            self.layers.append(layer)
        
    def forward(self, data):
        # Extract the Tensor from data.x
        x = data

        # Pass input through all layers except the last one with ReLU activation
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.relu(x)
        
        # Apply the final layer without activation (logits for multi-class classification)
        x = self.layers[-1](x)
        
        return x  # Return raw logits suitable for multi-class classification
"""

class ClusteringModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ClusteringModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        """
            CHEATSHEET:

            - SIGMOID: maps any real-valued number to a value between 0 and 1
                Sigmoid tends to saturate for very large positive or negative inputs

            - RELU: ReLU(x)=max⁡(0,x) returns the input value if it is positive, and zero otherwise
                does not saturate for positive inputs
                problem when zeros are input

            - LEAKY RELU:
                Might be best choice for us since it has all benefits of ReLU but 
                1st deriv > 0 for x < 0, meaning that it doesn't "switch off" with negative values.
                This might solve sparsity problem
            
            - SOFTMAX:
                best choice for multi-class classification but requires a more complex network to be useful


        """
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc3(x)  # Output layer does not use activation function
        return x
