import numpy as np
import NueralNet as net
from sklearn import preprocessing
from sklearn import cross_validation



### Helper functions

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
    return errorSum / length




######################Three clouds testing###########################

#Load data
# threeClouds = np.loadtxt(open("threeclouds.data","rb"),delimiter=",")
# cloudData = threeClouds[:,1:]
# cloudData = preprocessing.scale(threeClouds[:,1:])
# X_train, X_test, labels_train, labels_test = cross_validation.train_test_split(cloudData,threeClouds[:,0], test_size=.4,random_state=0)

# nn = net.NueralNet(learningRate=.5,activFunc='sigmoid',outputActivFunc='sigmoid',numInputs=2,numOutputs=3,numHLayers=1,numHiddenNodes=[2,3], hWeights=[[0.15, 0.2, 0.25, 0.3],[0.15, 0.2, 0.25, 0.3]], hBias=0.5, outputBias=0.5, outWeights=[0.4, 0.45, 0.5, 0.55])



# for i in range(500):
#     srange = list(range(len(X_train)))
#     np.random.shuffle(srange)
#     for j in srange:
#         nn.train(X_train[j], convertLabel(labels_train[j]))
#     errors = []
#     trange = list(range(len(X_test)))
#     np.random.shuffle(trange)
#     for k in trange:
#         errors.append([labels_test[k],np.argmax(nn.predict(X_test[k]))+1])
#     print(i,getError(errors))

    # print(errors[0])
    # print(i,threeClouds[index][0],nn.predict(threeClouds[index][1:]))
    # print(i, round(nn.getOverallError([[threeClouds[index][1:], convertLabel(threeClouds[index][0])]]), 9))

###########################################################################

####Wine Data##################################################################################

# #Load data
wines = np.loadtxt(open("wine.data","rb"),delimiter=",")
labels = wines[:,:1]
wdata = preprocessing.scale(wines[:,1:])
#wdata = wines[:,1:]
X_train, X_test, labels_train, labels_test = cross_validation.train_test_split(wdata,labels, test_size=.4,random_state=0)


nn = net.NueralNet(learningRate=10,activFunc='sigmoid',outputActivFunc='sigmoid',numInputs=13,numOutputs=3,numHLayers=1,numHiddenNodes=[5], hBias=0.35, outputBias=0.6, outWeights=[0.4, 0.45, 0.5, 0.55])


for i in range(250):
    for j in range(len(X_train)):
        nn.train(X_train[j], convertLabel(labels_train[j]))

    errors = []
    trange = list(range(len(X_test)))
    np.random.shuffle(trange)
    for k in trange:
        errors.append([labels_test[k],np.argmax(nn.predict(X_test[k]))+1])
    print(i,getError(errors))


    #print(i, round(nn.getOverallError([[threeClouds[index][1:], convertLabel(threeClouds[index][0])]]), 9))
##########################################################################

#############semeion Data ####################################


#Load data
# semeion = np.genfromtxt("semeion.data")
# labels = semeion[:,-10:]
# #sdata = preprocessing.scale(semeion[:,:-10])
# sdata = semeion[:,:-10]
# #wdata = preprocessing.scale(wines[:,1:])


# nn = net.NueralNet(learningRate=100,activFunc='tanh',outputActivFunc='tanh',numInputs=256,numOutputs=10,numHLayers=1,numHiddenNodes=[6,2], hBias=0.35, outputBias=.6, outWeights=[0.4, 0.45, 0.5, 0.55])


# # #index = 130
# # for i in range(50):
# #     index = int(input())
# #     nn.train(sdata[index],labels[index])
# #     print(np.argmax(labels[index])+1,np.argmax(nn.predict(sdata[index]))+1)


# #     print(getError(errors))

# for i in range(50):
#     for j in range(len(sdata)):
#         nn.train(sdata[j], labels[j])

#     errors = []
#     trange = list(range(len(sdata)))
#     np.random.shuffle(trange)
#     for k in trange:
#         errors.append([np.argmax(labels[k])+1,np.argmax(nn.predict(sdata[k]))+1])
#     print(getError(errors))
#######################################################################



###############First round testing#######################################
# nn = net.NueralNet(learningRate=0.5,activFunc='tanh',numInputs=2,numOutputs=2,numHLayers=2,numHiddenNodes=[2,3], hWeights=[[0.15, 0.2, 0.25, 0.3],[0.15, 0.2, 0.25, 0.3]], hBias=0.35, outputBias=0.6, outWeights=[0.4, 0.45, 0.5, 0.55])

# #nn.inspect();
# for i in range(10000):

#     nn.train([0.05, 0.1], [0.01, 0.99])

#     print(i, round(nn.getOverallError([[[0.05, 0.1], [0.01, 0.99]]]), 9))
############################################################################