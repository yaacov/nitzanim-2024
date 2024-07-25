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
    
# Define the SimpleBinaryClassifier with 2 input nodes and 1 output node
class SimpleBinaryClassifier(nn.Module):
    def __init__(self):
        super(SimpleBinaryClassifier, self).__init__()
        self.input_node1 = FeedForwardNode(2, 1)
        self.input_node2 = FeedForwardNode(2, 1)
        self.output_node = FeedForwardNode(2, 1)
        self.initialize_weights()

    def initialize_weights(self):
        # Manually set weights and biases for all nodes
        with torch.no_grad():
            # Input node 1
            self.input_node1.weights.copy_(torch.tensor([[1.0], [1.0]]))
            self.input_node1.bias.copy_(torch.tensor([-0.4]))
            # Input node 2
            self.input_node2.weights.copy_(torch.tensor([[-1.0], [-1.0]]))
            self.input_node2.bias.copy_(torch.tensor([1.6]))
            # Output node
            self.output_node.weights.copy_(torch.tensor([[1.0], [1.0]]))
            self.output_node.bias.copy_(torch.tensor([-1.4]))
    
    def forward(self, x):
        # Get outputs from the two input nodes
        x1 = self.input_node1(x)
        x2 = self.input_node2(x)
        # Stack the outputs and pass them to the output node
        combined_input = torch.cat((x1, x2), dim=1)
        y = self.output_node(combined_input)
        return y

# Example data: XOR problem (binary classification)
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

# Model
model = SimpleBinaryClassifier()

# Test the model with manually set weights and biases
with torch.no_grad():
    outputs = model(inputs)
    for input_pair, output in zip(inputs, outputs):
        print(f"Input: {input_pair.numpy()} -> Output: {output.item()}")
