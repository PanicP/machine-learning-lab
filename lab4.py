import numpy as np
import random

'''Array Creation'''

# a
arr1a = np.arange(0, 100, 2)
print('arr1a', arr1a)

# b
arr1b = np.array([(i, i, i) for i in range(1, 4, 1)])
print('arr1b', arr1b)

# c
arr1c = np.full((3, 5), 55)
print('arr1c', arr1c)

# d
arr1d = np.random.rand(5, 4, 3)
print('arr1d', arr1d)

'''Numpy basics and slicing'''
traind = np.random.uniform(0, 255, [2000, 20, 20])
# a
print('arr2a', traind[1000])
# b
print('arr2b', traind[0:1000:1, 0:5:1], traind[0:1000:1, 15:20:1])
# c
print('arr2c', np.amax(traind[10]), np.amin(traind[10]))
# d
print('arr2d')
# e
z = traind[10]
print('arr2e1', z[::2, ::])
print('arr2e2', z[::, ::2])
print('arr2e3', np.transpose(z))
print('arr2e4', np.flip(z, 0), np.flip(z, 1), z[::2, ::2])
# f
print('arr2f', 1-z)

'''Reduction'''

# a
print('arr3a', np.sum(traind[::, 0, 0]))
# b
print('arr3b', np.mean(traind[::, 0, 0]))
# c
print('arr3c', np.mean(traind, axis=(1, 2)))
# d
print('arr3d', np.mean(traind, axis=(1)))
# e
print('arr3e', np.sum(traind, axis=(1)))

'''Broadcasting'''

tempB = np.arange(1, 21, 1)
print(tempB)
# a
print('arr4a', traind + tempB[None, :])
# b
print('arr4b', traind + tempB[:, None])
# c
print('arr4c', traind + 0)

'''Fancy indexing and mask indexing'''

# a
# b
