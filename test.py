import numpy as np






# x = np.array([0.05,0.1,1])
# w = np.array([.15,.2,.35])

# print(sigmoid(w.dot(x)))

class Layer(object):
    """docstring for Layer"""
    def __init__(self, numNuerons, bias = None):
        self.bais = bias or 1
        self.nuerons = []

        for i in range(numNuerons):
            nuerons.append(Nueron(self.bias))



class Nueron:
    """docstring for Nueron"""
    def __init__(self,bias):
        self.weights = []
        self.bias = bias


    def sigmoid(self,x):
        print(x)
        return 1 / (1 + np.exp(-x))

    def getOutput(self,input):
        input = np.append(input,1)
        self.output = self.sigmoid(x.dot(self.weights))
        return self.output

    def calcError(self,labels):
        return 0.5 * (label - self.output) ** 2

    def randomWeights(self,num):
        return np.random.uniform(low=0.5, high=13.3, size=(num))




