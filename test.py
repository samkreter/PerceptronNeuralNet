import numpy as np


sigmoid = lambda x: 1 / (1 + np.exp(-x))

x = np.array([0.05,0.1,1])
w = np.array([.15,.2,.35])

print(w.dot(x))

