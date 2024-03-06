import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(42)
x = np.random.rand(100, 1)
y = 2 * x + 1 + 0.1 * np.random.randn(100, 1)

# Convert data to PyTorch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature, one output

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegression()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the results
predicted = model(x_tensor).detach().numpy()
plt.scatter(x, y, label='Original data')
plt.plot(x, predicted, label='Fitted line', color='r')
plt.legend()
plt.show()
