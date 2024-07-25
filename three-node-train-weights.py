# pylint: disable=all

import torch
import torch.nn as nn
import torch.optim as optim

# Define the FeedForwardNode
class FeedForwardNode(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedForwardNode, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        self.bias = nn.Parameter(torch.randn(output_size))
    
    def forward(self, x):
        x = torch.matmul(x, self.weights) + self.bias
        return x

# Define the SimpleBinaryClassifier with 2 input nodes and 1 output node
class SimpleBinaryClassifier(nn.Module):
    def __init__(self):
        super(SimpleBinaryClassifier, self).__init__()
        self.input_node1 = FeedForwardNode(2, 1)
        self.input_node2 = FeedForwardNode(2, 1)
        self.output_node = FeedForwardNode(2, 1)

    def forward(self, x):
        # Get outputs from the two input nodes
        x1 = self.input_node1(x)
        x2 = self.input_node2(x)
        # Apply sigmoid to get probability-like outputs
        x1 = torch.sigmoid(x1)
        x2 = torch.sigmoid(x2)
        # Stack the outputs and pass them to the output node
        combined_input = torch.cat((x1, x2), dim=1)
        y = self.output_node(combined_input)
        y = torch.sigmoid(y)  # Apply sigmoid to output
        return y

# Example data: XOR problem (binary classification)
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([[0], [1], [1], [1]], dtype=torch.float32)

# Model
model = SimpleBinaryClassifier()

# Loss function and optimizer
# Binary Classification: The XOR problem is a binary classification problem where each
# output can be either 0 or 1. 
# The nn.BCELoss() (Binary Cross Entropy Loss) is well-suited for this task because it 
# measures the difference between the predicted probabilities and the actual binary labels.
criterion = nn.BCELoss()  # Binary Cross Entropy loss
# Stochastic Gradient Descent (SGD): SGD is a popular optimization algorithm that updates
# model parameters using the gradient of the loss function with respect to the parameters.
# It is a simple yet effective method for finding the minimum of the loss function.
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    
    outputs = model(inputs)  # Forward pass
    loss = criterion(outputs, labels)  # Compute loss
    
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    if (epoch + 1) % 1000 == 0:
        print("node1 weights: ", model.input_node1.weights)
        print("node1 bias   : ", model.input_node1.bias)
        print("node2 weights: ", model.input_node2.weights)
        print("node2 bias   : ", model.input_node2.bias)
        print("node3 weights: ", model.output_node.weights)
        print("node3 bias   : ", model.output_node.bias)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model after training
with torch.no_grad():
    outputs = model(inputs)
    predicted = (outputs >= 0.5).float()
    for input_pair, output in zip(inputs, predicted):
        print(f"Input: {input_pair.numpy()} -> Predicted Output: {output.item()}")
