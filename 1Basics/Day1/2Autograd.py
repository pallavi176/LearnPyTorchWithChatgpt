#Autograd is PyTorch’s automatic differentiation engine that powers neural network training. 
#It tracks all operations on tensors. When you finish your computation, you can call .backward() 
#and have all the gradients computed automatically.

import torch

# Create tensors with requires_grad=True to track computation
x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
y = torch.tensor([4.0,5.0,6.0], requires_grad=True)

# Perform a simple computation
z = x * y

# Compute gradients
u = torch.tensor([1.0,1.0,1.0])
z.backward(u)

# Print gradients
print(x.grad)    # dz/dx
print(y.grad)    # dz/dy

#In this example, z is a tensor on which we perform an operation. Since x and y are created with requires_grad=True, 
#PyTorch will automatically calculate the gradients of z with respect to x and y.
