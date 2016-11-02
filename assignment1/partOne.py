from scipy import io
import ploty
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split

randomSeed = 0

amps = np.load('fifth.npy')
dataLen = len(amps)

# t = np.arange(0, dataLen)
# ploty.plotFunc(t, dataset)
# ploty.plotNsave(t, amps, 'figs/amplitudes.png')

# plt.figure()
# plt.hist(amps, bins=20)
# plt.show()

columnCount = 21
howManyToBeDiscarded = dataLen % columnCount

C = dataLen - howManyToBeDiscarded

ampsTrimmed = amps[:C]

ampsMatrix = ampsTrimmed.reshape((C / columnCount, columnCount))

randomState = np.random.RandomState(seed=randomSeed)
# np.random.shuffle(ampsMatrix)   #this is not functional, affects the array directly
randomState.shuffle(ampsMatrix)

dataTrain, dataRest = train_test_split(ampsMatrix, train_size=0.7, test_size=0.3, random_state=randomSeed)

dataValidation, dataTest = train_test_split(dataRest, train_size=0.5, test_size=0.5, random_state=randomSeed)


def makeLastColumnOutput(matrix):
    return matrix[:, :matrix.shape[1] - 1], matrix[:, matrix.shape[1] - 1]


XshufTrain, yShufTrain = makeLastColumnOutput(dataTrain)
print "train shapes"
print XshufTrain.shape
print yShufTrain.shape

XshufVal, yShufVal = makeLastColumnOutput(dataValidation)
print "validation shapes"
print XshufVal.shape
print yShufVal.shape

XshufTest, yShufTest = makeLastColumnOutput(dataTest)
print "test shapes"
print XshufTest.shape
print yShufTest.shape

t = np.arange(0, 20) / 20.0

def plotInputOutput(t, input, output, index=0):
    fig = plt.figure()
    step_t = t[len(t) - 1] - t[len(t) - 2]
    next_t = t[len(t) - 1] + step_t
    plt.xlim([0, next_t + 3 * step_t])
    plt.hold(True)
    plt.plot(t, input[index], 'b-')
    plt.plot([next_t], output[index], 'r.')
    plt.hold(False)
    plt.show()


#plotInputOutput(t, XshufTrain, yShufTrain)


def coefficientOfDetermination(realTargets, predictions):
    matrix = np.corrcoef(realTargets, predictions)
    cc = matrix[1,0]
    assert round(matrix[0,1],12) == round(matrix[1,0],12)
    return cc**2

sampleInput = XshufTrain[0]
sampleOutput = yShufTrain[0]

w, residuals, rank, s = np.linalg.lstsq(t, sampleInput)
print w


# w, residuals, rank, s = np.linalg.lstsq(XshufTrain, yShufTrain)
# print "coefficient of determination seems very good for the training data"
# print coefficientOfDetermination(realTargets=yShufTrain, predictions=XshufTrain.dot(w))
# print "let's try for the validation set instead"
# print coefficientOfDetermination(realTargets=yShufVal, predictions=XshufVal.dot(w))
# print "and the testing set also"
# print coefficientOfDetermination(realTargets=yShufTest, predictions=XshufTest.dot(w))
# print "coefficient of determination is good after all"