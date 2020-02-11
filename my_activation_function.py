import numpy as np 
import matplotlib.pyplot as plt


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
    return x*(x>0)

def my_activation_func(x):
    return relu(x)*(1+ np.exp(-(5/max(x))*x))

x = np.linspace(-100, 1000)
plt.plot(x,my_activation_func(x),'r')
plt.plot(x,relu(x),'g')
plt.show()