import numpy as np






# x = np.array([0.05,0.1,1])
# w = np.array([.15,.2,.35])

# print(sigmoid(w.dot(x)))

class Nueron:
    """docstring for Nueron"""
    def __init__(self,bias,numInputs):
        self.bias = bias
        self.weights = np.append(self.randomWeights(numInputs),bias)

    def sigmoid(self,x):
        print(x)
        return 1 / (1 + np.exp(-x))

    def output(self,x):
        x = np.append(x,1)
        print(self.weights)
        return (self.sigmoid(x.dot(self.weights)))

    def randomWeights(self,num):
        return np.random.uniform(low=0.5, high=13.3, size=(num))




