import sys
import numpy as np

#  normal usage
arr0 = np.zeros([3, 3])
print("arr0 info: ", arr0.shape, arr0.dtype)

arr1 = np.array([1, 2, 3])
print("arr2 info: ", arr1.shape, arr1.dtype)

arr2 = np.array([1, 2, 3.])
print("arr2 info: ", arr2.shape, arr2.dtype)

#  reshape
arr3 = np.array([1, 1, 1, 2, 2, 2])
arr4 = np.reshape(arr3, [2, 3])
arr4[1, 0] = 5
print(arr4)

# vectorization
arr5 = np.array([0.1, 0.2, 0.3, 0.4])
arr5 += 2
print(arr5)

# comparison
arr6 = np.array([0.1, 0.2, 0.3, 0.4])
arr7 = np.array([0.1, 0.1, 0.1, 0.4])
print(arr6 == arr7)
print((arr6 == arr7).astype("int"))

# slicing
arr8 = np.array([(i, i+2, i+3) for i in range(0, 6, 3)])
print(arr8)
print(arr8[0:1:1, 0:3])
print(arr8[:1:, :])
print(arr8[0, :])

# fancy indexing
arr9 = np.arange(0, 10, 1)
print(arr9[[0, 5, 7]])
arr10 = np.arange(0, 21, 1).reshape(3, 7)
print(arr10)
print("slice out rows 0 and 2; ", arr10[[0, 2], 1:5])

# broadcasting
img = np.arange(0, 25).reshape(5, 5)
single_row = np.array([1, 2, 3, 4, 5])
# inefficient way:
for i in range(0, 5):
    img[i, :] += single_row
    print(img)
# efficient way:
img += single_row.reshape(1, 5)
