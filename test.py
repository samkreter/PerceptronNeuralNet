import numpy as np


# x = np.array([0.05,0.1,1])
# w = np.array([.15,.2,.35])

# print(sigmoid(w.dot(x)))


class NueralNet():
    """docstring for NueralNet"""
    def __init__(self,learningRate,numInputs,numOutputs,numHLayers,numHiddenNodes = None,hWeights = None, hBias = None, outputBias = None, outWeights = None):

        #TODO Complete param check

        self.learningRate = learningRate or 1
        self.numInputs = numInputs
        self.numHiddenNodes = numHiddenNodes #Array
        self.numHLayers = numHLayers
        self.hiddenLayers = []
        self.outputLayer = Layer(numOutputs, outputBias)

        self.setUpHiddenLayers()
        self.setHiddenToOutput(outWeights)


    def setUpHiddenLayers(self):
        for i in range(self.numLayers):
            self.hiddenLayers.append(Layer(self.numHiddenNodes[i], hBias))
            self.setInputToHiddenWeights(i,hWeights[i])

    def train(self,inputs, labels):
        self.fullForwardPass(inputs);

        #Get output layer gradients
        outputGrad = [0] * len(self.outputLayer.nuerons)
        for i in range(len(self.outputLayer.nuerons)):
            outputGrad[i] = self.outputLayer.nuerons[i].calcGrad(labels[i])

        #####Get hidden layer gradients#####################################
        hiddenGrad = []
        #output to first hidden
        for i in range(len(self.hiddenLayers[-1].nuerons)):

            outputErrorSum = 0

            for j in range(len(self.outputLayer.nuerons)):

                outputErrorSum += outputGrad[j] * self.outputLayer.nuerons[j].weights[i]

            hiddenGrad.append(outputErrorSum * self.hiddenLayer.nuerons[i].activationFunDeriv())


        #TODO: Check math for counting, way to complicated here
        for i in range(self.numHLayers-2,-1,-1):
            hiddenGrad.insert(0,[0] * len(self.hiddenLayers[i].nuerons))

            for k in range(len(self.hiddenLayers[i].nuerons)):

                outputErrorSum = 0

                for j in range(len(self.hiddenLayers[i+1].nuerons)):

                    outputErrorSum += hiddenGrad[1][j] * self.hiddenLayers[i+1].nuerons[j].weights[k]

                hiddenGrad[0][k] = outputErrorSum * self.hiddenLayers[i].nuerons[k].activationFunDeriv()


        #Adjust output layer weights
        for i in range(len(self.outputLayer.nuerons)):
            for w in range(len(self.outputLayer.nuerons[i].weights)):
                # print("output: ",self.learningRate * self.outputLayer.nuerons[i].getInputAtIndex(i) * outputGrad[i])
                self.outputLayer.nuerons[i].weights[w] -= self.learningRate * self.outputLayer.nuerons[i].getInputAtIndex(i) * outputGrad[i]


        #Adjust hidden layer weights
        for k in range(self.numHLayers):
            for i in range(len(self.hiddenLayers[k].nuerons)):
                for w in range(len(self.hiddenLayers[k].nuerons[i].weights)):
                    self.hiddenLayers[k].nuerons[i].weights[w] -= self.learningRate * hiddenGrad[k][i] * self.hiddenLayers[k].nuerons[i].getInputAtIndex(i)
                    # print("hidden: ",self.learningRate * hiddenGrad[i] * self.hiddenLayer.nuerons[i].getInputAtIndex(i))

    #MultiSet
    def fullForwardPass(self, inputs):
        currOutput = self.hiddenLayers[0].forwardPass(inputs)

        for i in range(1,self.hiddenLayers):
            currOutput = self.hiddenLayers[i].forwardPass(currOutput)

        return self.outputLayer.forwardPass(currOutput)

    #multiSet
    def setInputToHiddenWeights(self,index,hWeights):
        if index == 0:
            numInputs = self.numInputs
        else:
            numInputs = len(self.hiddenLayer[index-1].nuerons)

        for nueron in self.hiddenLayers[index].nuerons:
            if hWeights:
                nueron.weights = hWeights[:numInputs]
            else:
                nueron.weights = np.random.rand(numInputs)

    #MultiSet
    def setHiddenToOutput(self,outWeights):

        for nueron in self.outputLayer.nuerons:

            if outWeights:
                nueron.weights = outWeights[:len(self.hiddenLayers[-1].nuerons)]
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

#nn.inspect();
for i in range(10000):

    nn.train([0.05, 0.1], [0.01, 0.99])

    print(i, round(nn.getOverallError([[[0.05, 0.1], [0.01, 0.99]]]), 9))