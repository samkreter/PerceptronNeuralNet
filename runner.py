import numpy as np
import NueralNet as net



#Three clouds testing

#Load data
threeClouds = np.loadtxt(open("threeclouds.data","rb"),delimiter=",")

nn = net.NueralNet(learningRate=0.5,activFunc='sigmoid',outputActivFunc='linear',numInputs=2,numOutputs=1,numHLayers=2,numHiddenNodes=[4,4], hWeights=[[0.15, 0.2, 0.25, 0.3],[0.15, 0.2, 0.25, 0.3]], hBias=0.35, outputBias=0.6, outWeights=[0.4, 0.45, 0.5, 0.55])

#nn.inspect();



def convertLabel(num):
    if num == 1:
        return [1,0,0]
    elif num == 2:
        return [0,1,0]
    else:
        return [0,0,1]


for i in range(1000):
    for j in range(len(threeClouds)):
        nn.train([threeClouds[j][1],threeClouds[j][1]], [threeClouds[j][0]])
    index = np.random.randint(0,len(threeClouds)-1)
    print(i,threeClouds[index][0],nn.predict([threeClouds[index][1],threeClouds[index][1]]))
    #print(i, round(nn.getOverallError([[threeClouds[index][1:], convertLabel(threeClouds[index][0])]]), 9))





#First round testing
# nn = net.NueralNet(learningRate=0.5,activFunc='tanh',numInputs=2,numOutputs=2,numHLayers=2,numHiddenNodes=[2,3], hWeights=[[0.15, 0.2, 0.25, 0.3],[0.15, 0.2, 0.25, 0.3]], hBias=0.35, outputBias=0.6, outWeights=[0.4, 0.45, 0.5, 0.55])

# #nn.inspect();
# for i in range(10000):

#     nn.train([0.05, 0.1], [0.01, 0.99])

#     print(i, round(nn.getOverallError([[[0.05, 0.1], [0.01, 0.99]]]), 9))