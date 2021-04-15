import numpy as np

w = np.arange(12).reshape(3, 4)
b = np.array([1, 2, 3, 4])
print(w)
print(w+b - w)


a = np.arange(12).reshape(3, 4)
print(a)
print(a.max())