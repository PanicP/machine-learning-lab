import numpy as np

a1 = np.arange(0, 101, 2)
# print(a1)

a2 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]).reshape(3, 3)
# print(a2)

a3 = np.full((3, 5), 55)
# print(a3)

a4 = np.random.rand(5, 4, 3)
# print(a4)


traind = np.random.uniform(0, 255, [2000, 20, 20])
# print(traind)
b1 = traind[1000, :, :]
# print(b1)
b2 = traind[0:1000:1, 0:5:1]
b2 = traind[0:1000:1, 15:20:1]
# print(b2)
# print('b3', np.amax(traind[10]), np.amin(traind[10]))
# e
z = traind[10]
# print('arr2e1', z[::2, ::])
# print('arr2e2', z[::, ::2])
# print('arr2e3', np.transpose(z))
# print('arr2e4', np.flip(z, 0), np.flip(z, 1), z[::2, ::2])
# f
# print('arr2f', 1-z)


#######

# print('arr3a', np.sum(traind[::, 0, 0]))
# print('arr3b', np.mean(traind[::, 0, 0]))
# print('arr3b', np.mean(traind, axis=(1, 2)))
# # d
# print('arr3d', np.mean(traind, axis=(1)))
# # e
# print('arr3e', np.sum(traind, axis=(1)))

####

bc = np.arange(1, 21, 1)
# print(bc)
# traind += bc.reshape(1, 1, 20)
# # b)
# traind += bc.reshape(1, 20, 1)
# # c)
# traind += traind[0].reshape(1, 20, 20)
# print(traind)
