import numpy as np
import matplotlib.pyplot as plt

# Define the functions
x = np.linspace(0.01, 1, 1000)  # Avoid log(0) by starting from 0.01
logx = -np.log(x)
relu = np.maximum(0, 0.5 - x)

# Derivatives of the functions
logx_derivative = -1 / x  # Derivative of log(x)
relu_derivative = np.where(x < 0.5, -1, 0)  # Derivative of ReLU(0.5 - x)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(6, 3))

# Plot a: log(x) and ReLU(0.5 - x)
axs[0].plot(x, logx, label='log(x)', color='b')
axs[0].plot(x, relu, label='ReLU(0.5 - x)', color='r')
# axs[0].set_title('-log(x) and ReLU(0.5 - x)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
# axs[0].legend()

# Plot b: Derivative of log(x) and ReLU(0.5 - x)
axs[1].plot(x, logx_derivative, label="Derivative of log(x)", color='b')
axs[1].plot(x, relu_derivative, label="Derivative of ReLU(0.5 - x)", color='r')
# 值域
axs[1].set_ylim(-5, 2)  # Set y-axis limits for better visibility
# axs[1].set_title('Derivatives of -log(x) and ReLU(0.5 - x)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Derivative')
# axs[1].legend()

plt.tight_layout()
plt.show()
plt.close()