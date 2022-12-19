import numpy as np
import matplotlib.pyplot as plt
import os
import sys

data = np.load("mnist.npz")
traind = data["arr_0"]
trainl = data["arr_2"]

# 1a)
print(np.argmax(traind[:, 0, 0]))  # --> 0
# 1b)
print(traind.mean(axis=2))  # long output! shape (60000,28)
# 1c)
print(traind.sum(axis=1))  # long output! shape (60000,28)

# 2a)
f, ax = plt.subplots(1, 1)
x_data = np.linspace(1, 5, 100)
y_data = 1. / x_data
ax.plot(x_data, y_data)
plt.show()
# 2b)
f, ax = plt.subplots(1, 1)
x_data = np.linspace(1, 5, 100)
y_data = 1. / x_data
ax.bar(x_data, y_data)
plt.show()
# 2c)
f, ax = plt.subplots(1, 1)
x_data = np.linspace(1, 5, 100)
y_data_1 = 1. / x_data
y_data_2 = np.sqrt(x_data)
ax.plot(x_data, y_data_1)
ax.plot(x_data, y_data_2)
plt.show()

# 3a)
indices = [5, 6, 7]
f, ax = plt.subplots(1, 3)

for i, index in enumerate(indices):
    ax[i].imshow(traind[index])
plt.show()

# 3b)
means = traind.mean(axis=(1, 2))
x_data = np.linspace(0, 60000, 60000)
f, ax = plt.subplots(1, 1)
ax.scatter(x_data, means)
plt.show()

# 3c)
means = traind.mean(axis=(1, 2))
mask = (means > 0.3)
strongImgs = traind[mask, :, :]

f, ax = plt.subplots(1, 3)
for i in range(0, 3):
    ax[i].imshow(strongImgs[i])
plt.show()

# 3d)
indices = np.random.randint(0, 60000, 10)
f, ax = plt.subplots(1, 10)

for i, index in enumerate(indices):
    ax[i].imshow(traind[index])
plt.show()

# 3e)
# create mask for images of class 5
numerical_classes = trainl.argmax(axis=1)
# gives a boolean array of 60000 entries
mask = (numerical_classes == 5)

# copy out all samples using the boolean array: mask indexing!
samples5 = traind[mask, :, :]

# display first 5 images of class 5
f, ax = plt.subplots(1, 10)
for i in range(0, 10):
    ax[i].imshow(samples5[i])
plt.show()
