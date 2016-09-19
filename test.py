import numpy as np


# x = np.array([0.05,0.1,1])
# w = np.array([.15,.2,.35])

# print(sigmoid(w.dot(x)))


class NueralNet():
    """docstring for NueralNet"""
    def __init__(self,learningRate,numInputs,numHidden,numOutputs,hWeights = None, hBias = None, outputBias = None, outWeights = None):
        self.learningRate = learningRate or 1
        self.numInputs = numInputs
        self.hiddenLayer = Layer(numHidden, hBias)
        self.outputLayer = Layer(numOutputs, outputBias)

        self.setInputToHiddenWeights(hWeights)
        self.setHiddenToOutput(outWeights)

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.numInputs))
        print('------')
        print('Hidden Layer')
        self.hiddenLayer.inspect()
        print('------')
        print('* Output Layer')
        self.outputLayer.inspect()
        print('------')


    def train(self,inputs, labels):
        self.fullForwardPass(inputs);

        #Get output layer gradients
        outputGrad = [0] * len(self.outputLayer.nuerons)
        for i in range(len(self.outputLayer.nuerons)):
            outputGrad[i] = self.outputLayer.nuerons[i].calcGrad(labels[i])

        #Get hidden layer gradients
        hiddenGrad = [0] * len(self.hiddenLayer.nuerons)
        for i in range(len(self.hiddenLayer.nuerons)):

            outputErrorSum = 0

            for j in range(len(self.outputLayer.nuerons)):

                outputErrorSum += outputGrad[j] * self.outputLayer.nuerons[j].weights[i]

            hiddenGrad[i] = outputErrorSum * self.hiddenLayer.nuerons[i].activationFunDeriv()

        #Adjust output layer weights
        for i in range(len(self.outputLayer.nuerons)):
            for w in range(len(self.outputLayer.nuerons[i].weights)):
                self.outputLayer.nuerons[i].weights[w] -= self.learningRate * self.outputLayer.nuerons[i].getInputAtIndex(i) * outputGrad[i]

        #Adjust hidden layer weights
        for i in range(len(self.hiddenLayer.nuerons)):
            for w in range(len(self.hiddenLayer.nuerons[i].weights)):
                self.hiddenLayer.nuerons[i].weights[w] -= self.learningRate * hiddenGrad[i] * self.hiddenLayer.nuerons[i].getInputAtIndex(i)



    def fullForwardPass(self, inputs):

        return self.outputLayer.forwardPass(self.hiddenLayer.forwardPass(inputs))

    def setInputToHiddenWeights(self,hWeights):

        for nueron in self.hiddenLayer.nuerons:
            if hWeights:
                nueron.weights = np.append(hWeights,nueron.bias)
            else:
                nueron.weights = np.append(np.random.rand(self.numInputs),nueron.bias)




    def setHiddenToOutput(self,outWeights):

        for nueron in self.outputLayer.nuerons:

            if outWeights:
                nueron.weights = outWeights
            else:
                nueron.weight = np.random.rand(len(self.hiddenLayer.nuerons))

    def getOverallError(self, trainingData):

        error = 0

        for i in range(len(trainingData)):

            trainingInputs, trainingOutputs = trainingData[i]

            self.fullForwardPass(trainingInputs)

            for j in range(len(trainingOutputs)):

                error += self.outputLayer.nuerons[j].calcError(trainingOutputs[j])

        return error




class Layer():
    """docstring for Layer"""
    def __init__(self, numNuerons, bias = None):
        self.bias = bias or 1
        self.nuerons = []

        for i in range(numNuerons):
            self.nuerons.append(Nueron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.nuerons))
        for n in range(len(self.nuerons)):

            print(' Neuron', n)

            for w in range(len(self.nuerons[n].weights)):

                print('  Weight:', self.nuerons[n].weights[w])

            print('  Bias:', self.bias)

    def forwardPass(self,input):
        outputs = []

        for nueron in self.nuerons:

            outputs.append(nueron.getOutput(input))

        return outputs


class Nueron:
    """docstring for Nueron"""
    def __init__(self,bias):
        self.weights = []
        self.bias = bias


    def sigmoid(self,x):
        print(x)
        return 1 / (1 + np.exp(-x))

    def getOutput(self,inputs):
        # self.inputs = np.append(input,1)
        # exit(-1)
        # self.output = self.sigmoid(self.inputs.dot(self.weights))
        self.inputs = inputs
        output = 0
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]

        self.output = self.sigmoid(output + self.bias)
        return self.output

    def calcError(self,label):
        return 0.5 * (label - self.output) ** 2

    def randomWeights(self,num):
        return np.random.uniform(low=0.5, high=13.3, size=(num))

    def activationFunDeriv(self):
        return self.output * (1 - self.output)

    def calcGrad(self, targetOutput):
        return -(targetOutput - self.output) * self.activationFunDeriv();

    def getInputAtIndex(self,index):
        return self.inputs[index]

#learningRate,numInputs,numHidden,numOutputs
nn = NueralNet(0.5,2, 2, 2, hWeights=[0.15, 0.2, 0.25, 0.3], hBias=0.35, outputBias=0.6, outWeights=[0.4, 0.45, 0.5, 0.55])

nn.inspect();
# for i in range(10000):

#     nn.train([0.05, 0.1], [0.01, 0.99])

#     print(i, round(nn.getOverallError([[[0.05, 0.1], [0.01, 0.99]]]), 9))