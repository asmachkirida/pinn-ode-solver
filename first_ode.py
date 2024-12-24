import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available and use it; otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the first-order neural network model
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(1, 10)
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        layer_out = torch.sigmoid(self.hidden_layer(x))  # Sigmoid activation
        output = self.output_layer(layer_out)
        return output

# Initialize the model and move it to the chosen device
N = Network().to(device)

# Define the example function for the first-order ODE
def f(x):
    return torch.exp(x)

# Define the loss function for the first-order ODE
def loss(x):
    x.requires_grad = True
    y = N(x)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return torch.mean((dy_dx - f(x)) ** 2) + (y[0, 0] - 1.) ** 2  # Boundary condition y(0) = 1

# Optimizer: LBFGS (suitable for small-batch, high-precision optimization)
optimizer = optim.LBFGS(N.parameters())

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
    loss_value = optimizer.step(closure)  # Step through optimizer
    print(f'Epoch [{i+1}/{epochs}], Loss: {loss_value}')  # Print loss for each epoch

# After training, you can check the model's output (e.g., prediction at some point)
xx = torch.linspace(0, 1, 100)[:, None].to(device)
with torch.no_grad():
    yy = N(xx)  # Predicted output for the range of x

# Optionally, print some of the predictions to see the results
print("Predictions at x = [0.0, 0.5, 1.0]:")
print(yy[::50])  # Print predictions at the first, middle, and last points

# Save the trained model
model_path = "model_first_order.pth"
torch.save(N.state_dict(), model_path)  # Save the model's state_dict

print(f"Model saved to {model_path}")
