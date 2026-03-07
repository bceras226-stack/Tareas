import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

def softplus(x):
    return np.log(1 + np.exp(x))

y1 = sigmoid(x)
y2 = tanh(x)
y3 = relu(x)
y4 = leaky_relu(x)
y5 = softplus(x)
plt.figure()
plt.subplot(5,2,1)
plt.plot(x, y1, label="Sigmoid")
plt.title("Función de Activación Sigmoid")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid(True)
plt.legend()

plt.subplot(5,2,4)
plt.plot(x, y2, label="Tanh")
plt.title("Función de Activación Tanh")
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.grid(True)
plt.legend()

plt.subplot(5,2,5)
plt.plot(x, y3, label="ReLU")
plt.title("Función de Activación ReLU")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid(True)
plt.legend()

plt.subplot(5,2,8)
plt.plot(x, y4, label="Leaky ReLU")
plt.title("Función de Activación Leaky ReLU")
plt.xlabel("x")
plt.ylabel("LeakyReLU(x)")
plt.grid(True)
plt.legend()

plt.subplot(5,2,9)
plt.plot(x, y5, label="Softplus")
plt.title("Función de Activación Softplus")
plt.xlabel("x")
plt.ylabel("Softplus(x)")
plt.grid(True)
plt.legend()
plt.show()