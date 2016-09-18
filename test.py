import numpy as np






# x = np.array([0.05,0.1,1])
# w = np.array([.15,.2,.35])

# print(sigmoid(w.dot(x)))


class NueralNet():
    """docstring for NueralNet"""
    def __init__(self,learningRate,numInputs,numHidden,numOutputs,hWeights = None, hBias = None,outBias = None, outWeights = None, ):
        self.learningRate = learningRate or 1
        self.numInputs = numInputs
        self.hiddenLayer = Layer(numHidden, hBias)
        self.outputLayer = Layer(numOutputs, outputBias)

        self.setInputToHiddenWeights(hWeights)
        self.setHiddenToOutput(outWeights)


    def train(self,inputs, labels):
        self.fullForwardPass(ipnuts);

        #Get output layer gradients
        outputGrad = [0] * len(self.outputLayer.neurons)
        for i in range(len(self.outputLayer.neurons)):
            outputGrad[i] = self.outputLayer.neurons[i].calcGrad(labels[i])

        #Get hidden layer gradients
        hiddenGrad = [0] * len(self.hiddenLayer.neurons)
        for i in range(len(self.hiddenLayer.neurons)):
            dErrorHiddenNeuronOutput = 0

            for j in range(len(self.output_layer.neurons)):

                outputErrorSum += outputGrad[j] * self.outputLayer.neurons[j].weights[i]

            hiddenGrad[i] = outputErrorSum * self.hiddenLayer.neurons[i].activationFunDeriv(inputs)

        #Adjust output layer weights
        for i in range(len(self.outputLayer.nuerons)):
            for w in range(len(self.outputLayer.nuerons[i].weights)):
                self.outputLayer.neurons[i].weights[w] -= self.learningRate * self.outputLayer.neurons[i].getInputAtIndex(i) * outputGrad[i]

        #Adjust hidden layer weights
        for i in range(len(self.hiddenLayer.nuerons)):
            for w in range(len(self.hiddenLayer.nuerons[i].weights)):
                self.hiddenLayer.nuerons[i].weights[w] -= self.learningRate * self.hiddenGrad[i] * self.hiddenLayer.nuerons[i].getInputAtIndex(i)



    def fullForwardPass(self, inputs):

        return self.outputLayer.forwardPass(self.hiddenLayer.forwardPass(inputs))

    def setInputToHiddenWeights(hWeights):

        for neuron in self.hiddenLayer.nuerons:
            if hWeights:
                nueron.weights = hWeights
            else:
                nueron.weights = np.random.rand(self.numInputs)




    def setHiddenToOutput(outWeights):

        for nueron in self.outputLayer:

            if outWeights:
                nueron.weights = outWeights
            else:
                nueron.weight = np.random.rand(len(self.hiddenLayer.nuerons))




class Layer():
    """docstring for Layer"""
    def __init__(self, numNuerons, bias = None):
        self.bais = bias or 1
        self.nuerons = []

        for i in range(numNuerons):
            nuerons.append(Nueron(self.bias))

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

    def getOutput(self,input):
        self.inputs = np.append(input,1)
        self.output = self.sigmoid(self.inputs.dot(self.weights))
        return self.output

    def calcError(self,labels):
        return 0.5 * (label - self.output) ** 2

    def randomWeights(self,num):
        return np.random.uniform(low=0.5, high=13.3, size=(num))

    def activationFunDeriv(self):
        return self.output * (1 - self.output)

    def calcGrad(self, targetOutput):
        return -(targetOutput - self.output) * self.calculate_pd_total_net_input_wrt_input();

    def getInputAtIndex(self,index):
        return self.inputs[index]


