import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available and use it; otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the second-order neural network model
class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(1, 10)
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        layer_out = torch.sigmoid(self.hidden_layer(x))  # Sigmoid activation
        output = self.output_layer(layer_out)
        return output

# Initialize the model and move it to the chosen device
N2 = Network2().to(device)

# Define the example function for the second-order ODE
def f(x):
    return -torch.ones(x.shape[0], x.shape[1])

# Define the loss function for the second-order ODE
def loss(x):
    x.requires_grad = True
    y = N2(x)
    
    # First derivative (dy/dx)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    
    # Second derivative (d²y/dx²)
    y_double_prime = torch.autograd.grad(dy_dx.sum(), x, create_graph=True)[0]
    
    # Loss function with boundary conditions (y(0) = 0 and y(1) = 0)
    return torch.mean((y_double_prime - f(x)) ** 2) + 0.5 * (y[0, 0] - 0.) ** 2 + 0.5 * (y[-1, 0] - 0.) ** 2

# Optimizer: LBFGS
optimizer = optim.LBFGS(N2.parameters())

# Define the input range for the training
x = torch.linspace(0, 1, 100)[:, None].to(device)

# Closure function to be used in LBFGS
def closure():
    optimizer.zero_grad()
    l = loss(x)
    l.backward()
    return l

# Training loop
epochs = 10
for i in range(epochs):
    optimizer.step(closure)
    print(f"Epoch [{i+1}/{epochs}], Loss: {loss(x).item()}")

# After training, you can check the model's output (e.g., prediction at some point)
xx = torch.linspace(0, 1, 100)[:, None].to(device)
with torch.no_grad():
    yy = N2(xx)  # Predicted output for the range of x

# Save the trained model
torch.save(N2.state_dict(), "model_second_order.pth")
print("Model saved to model_second_order.pth")

# You can print the predictions for specific values of x (optional)
print(f"Predictions at x = [0.0, 0.5, 1.0]:")
print(yy[[0, 50, 99], :].cpu().numpy())
