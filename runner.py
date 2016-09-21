import numpy as np
import NueralNet as net



# #Three clouds testing###########################

# #Load data
# threeClouds = np.loadtxt(open("threeclouds.data","rb"),delimiter=",")

# nn = net.NueralNet(learningRate=10,activFunc='sigmoid',outputActivFunc='sigmoid',numInputs=2,numOutputs=3,numHLayers=2,numHiddenNodes=[2,3], hWeights=[[0.15, 0.2, 0.25, 0.3],[0.15, 0.2, 0.25, 0.3]], hBias=0.5, outputBias=0.5, outWeights=[0.4, 0.45, 0.5, 0.55])




# def convertLabel(num):
#     if num == 1:
#         return [1,0,0]
#     elif num == 2:
#         return [0,1,0]
#     else:
#         return [0,0,1]

# def getError(errors):
#     length = len(errors)
#     errorSum = 0.0
#     errorList = [0,0,0]
#     for i in range(length):
#         if errors[i][0] != errors[i][1]:
#             errorSum += 1
#             errorList[int(errors[i][0])-1] += 1
#     print(np.argmax(errorList)+1)
#     return errorSum / length


# for i in range(500):
#     srange = list(range(len(threeClouds)))
#     np.random.shuffle(srange)
#     for j in srange:
#         nn.train(threeClouds[j][1:], convertLabel(threeClouds[j][0]))
#     errors = []
#     trange = list(range(len(threeClouds)))
#     np.random.shuffle(trange)
#     for i in trange:
#         errors.append([threeClouds[i][0],np.argmax(nn.predict(threeClouds[i][1:]))+1])
#     print(getError(errors))
    #input()
    #print(errors[0])
    #print(i,threeClouds[index][0],nn.predict(threeClouds[index][1:]))
    #print(i, round(nn.getOverallError([[threeClouds[index][1:], convertLabel(threeClouds[index][0])]]), 9))


####Wine Data########################

#Load data
wines = np.loadtxt(open("wine.data","rb"),delimiter=",")

nn = net.NueralNet(learningRate=0.5,activFunc='tanh',outputActivFunc='tanh',numInputs=13,numOutputs=3,numHLayers=2,numHiddenNodes=[4,4], hWeights=[[0.15, 0.2, 0.25, 0.3],[0.15, 0.2, 0.25, 0.3]], hBias=0.35, outputBias=0.6, outWeights=[0.4, 0.45, 0.5, 0.55])



def convertLabel(num):
    if num == 1:
        return [1,0,0]
    elif num == 2:
        return [0,1,0]
    else:
        return [0,0,1]

def getError(errors):
    length = len(errors)
    errorSum = 0.0
    errorList = [0,0,0]
    for i in range(length):
        if errors[i][0] != errors[i][1]:
            errorSum += 1
            errorList[int(errors[i][0])-1] += 1
    print(np.argmax(errorList)+1)
    return errorSum / length

for i in range(50):
    for j in range(len(wines)):
        nn.train(wines[j][1:], convertLabel(wines[j][0]))

    errors = []
    trange = list(range(len(threeClouds)))
    np.random.shuffle(trange)
    for i in trange:
        errors.append([wines[i][0],np.argmax(nn.predict(wines[i][1:]))+1])
    print(getError(errors))


    #print(i, round(nn.getOverallError([[threeClouds[index][1:], convertLabel(threeClouds[index][0])]]), 9))




#First round testing
# nn = net.NueralNet(learningRate=0.5,activFunc='tanh',numInputs=2,numOutputs=2,numHLayers=2,numHiddenNodes=[2,3], hWeights=[[0.15, 0.2, 0.25, 0.3],[0.15, 0.2, 0.25, 0.3]], hBias=0.35, outputBias=0.6, outWeights=[0.4, 0.45, 0.5, 0.55])

# #nn.inspect();
# for i in range(10000):

#     nn.train([0.05, 0.1], [0.01, 0.99])

#     print(i, round(nn.getOverallError([[[0.05, 0.1], [0.01, 0.99]]]), 9))