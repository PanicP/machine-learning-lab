import numpy as np
import matplotlib.pyplot as plt

data = np.load("mnist.npz")
traind = data["arr_0"]
trainl = data["arr_2"]

# 1a)
print(np.argmax(traind[:, 0, 0]))  # --> 0
# 1b)
print(traind.mean(axis=2))  # long output! shape (60000,28)
# 1c)
print(traind.sum(axis=1))  # long output! shape (60000,28)

(_, ax) = plt.subplots(2, 2)
# 2a)
# f, ax = plt.subplots(1, 1)
x_data = np.linspace(1, 5, 100)
y_data = 1. / x_data
ax[0, 0].plot(x_data, y_data)
# plt.show()
# 2b)
# f, ax = plt.subplots(1, 1)
x_data = np.linspace(1, 5, 100)
y_data = 1. / x_data
ax[0, 1].bar(x_data, y_data)
# plt.show()
# 2c)
# f, ax = plt.subplots(1, 1)
x_data = np.linspace(1, 5, 100)
y_data_1 = 1. / x_data
y_data_2 = np.sqrt(x_data)
ax[1, 0].plot(x_data, y_data_1)
ax[1, 0].plot(x_data, y_data_2)
plt.show()
