#Creating Tensors
import torch

# Create a tensor of size 2x3 filled with zeros
tensor_a = torch.zeros(2,3)
print(tensor_a)

# Create a tensor from a list
tensor_b = torch.tensor([1,2,3])
print(tensor_b)

# Create a random tensor
tensor_c = torch.rand(2,3)
print(tensor_c)

## Tensor Operations

# Add two tensors
result = tensor_b + tensor_c
print("Addition:", result)

# Multiply tensors element-wise
result = tensor_b * tensor_c
print("Multiplication:", result)

# Matrix multiplication
result = torch.matmul(tensor_b.float(), tensor_c.t())  # .t() is for transpose
print("Matrix Multiplication:", result)
