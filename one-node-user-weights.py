# pylint: disable=all

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the FeedForwardNode
class FeedForwardNode(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedForwardNode, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        self.bias = nn.Parameter(torch.randn(output_size))
    
    def forward(self, x):
        x = torch.matmul(x, self.weights) + self.bias
        # Thresholding to get binary output
        x = (x >= 0.5).float()

        return x

# Define the SimpleBinaryClassifier
class SimpleBinaryClassifier(nn.Module):
    def __init__(self):
        super(SimpleBinaryClassifier, self).__init__()
        self.node = FeedForwardNode(2, 1)
        self.initialize_weights()

    def initialize_weights(self):
        # Manually set weights and biases
        with torch.no_grad():  # Disable gradient tracking
            self.node.weights.copy_(torch.tensor([[1.0], [1.0]]))  # Example weights
            self.node.bias.copy_(torch.tensor([0]))             # Corrected bias
    
    def forward(self, x):
        x = self.node(x)
        return x

# Example data: XOR problem (binary classification)
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

# Model
model = SimpleBinaryClassifier()

# Test the model with manually set weights and biases
with torch.no_grad():
    outputs = model(inputs)
    for input_pair, output in zip(inputs, outputs):
        print(f"Input: {input_pair.numpy()} -> Output: {output.item()}")