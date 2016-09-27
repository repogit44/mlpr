import numpy as np
from ploty import plotFunc, startPlot
import matplotlib.pyplot as plt
import math

# X = np.tile(mu[:,None], (1, D)) + 0.01*np.random.randn(N, D)

# a = np.array([3, 5, 1, 6, 3])

# print np.tile(a, (3,2))

np.random.seed(0)


def noisySine(x):
    whatever = 2
    return np.sin(x) + np.random.rand(len(x)) / whatever


startX = 0
stopX = 10

samples_x = np.arange(startX, stopX, 0.5)
samples_y = noisySine(samples_x)

# plotFunc(samples_x, samples_y)
startPlot(samples_x, samples_y)


def getRadialBasisFunction(xx, cc):
    return np.exp(-((xx - cc) ** 2) / 2)


def phiRadialBasisFunction(Xin):
    return np.hstack([
        getRadialBasisFunction(Xin, 2),
        getRadialBasisFunction(Xin, 5),
        getRadialBasisFunction(Xin, 8),
    ])


def getPolynomial(degree):
    def getPolynomial(feature):
        # powers = np.zeros((len(feature), degree))
        powerOne = feature if len(feature.shape) > 1 else feature[np.newaxis].T

        if degree > 1:
            powers = []
            powers += [powerOne]
            # then loop over the remaining degrees:
            # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
            for curPower in range(2, degree + 1):
                powers += [powerOne ** curPower]

            return np.hstack(powers)
        else:
            return powerOne

    return getPolynomial


#print getPolynomial(4)(np.array([2, 5, 3]))


def fit_and_plot(phi_fn, X, yy):
    """phi_fn takes Nx1 inputs and returns NxD basis function values"""
    # w_fit = phi_fn(X) \ yy; % Dx1
    wFit = np.linalg.lstsq(phi_fn(X), yy)[0]
    X_grid = np.arange(startX, stopX, 0.1)[np.newaxis].T
    f_grid = phi_fn(X_grid).dot(wFit)
    # plot(X_grid, f_grid, 'LineWidth', 2);
    # plotFunc(X_grid, f_grid)
    plt.hold('on')
    # plt.plot(X_grid, f_grid, 'b-')
    plt.plot(X_grid, f_grid, 'g.')
    plt.show()


X = samples_x[np.newaxis].T
# this works well
#fit_and_plot(phiRadialBasisFunction, samples_x[np.newaxis].T, samples_y)

def phiPolynomial(degree):
    def myfunc(s):
        return degree + s

    return myfunc

#print
#print phiPolynomial(3)(5)

k = 12
#fit_and_plot(getPolynomial(k), X, samples_y)   # this is overfitting

#lets try and expand the tables as suggested
curLambda = 2

yExtra = np.zeros(k)
#print yExtra
featuresExtra = math.sqrt(curLambda) * np.eye(k)
#print featuresExtra

ourPolynomialGenerator = getPolynomial(k)
newY = np.concatenate([samples_y, yExtra])
newX = np.vstack([ourPolynomialGenerator(X), featuresExtra])
#print newY
#print newX.shape

def fitAndPlot2(features, yy, phi_fn):
    """phi_fn takes Nx1 inputs and returns NxD basis function values"""
    # w_fit = phi_fn(X) \ yy; % Dx1
    wFit = np.linalg.lstsq(features, yy)[0]
    X_grid = np.arange(startX, stopX, 0.1)[np.newaxis].T
    f_grid = phi_fn(X_grid).dot(wFit)
    # plot(X_grid, f_grid, 'LineWidth', 2);
    # plotFunc(X_grid, f_grid)
    plt.hold('on')
    # plt.plot(X_grid, f_grid, 'b-')
    plt.plot(X_grid, f_grid, 'g.')
    plt.show()

fitAndPlot2(newX, newY, ourPolynomialGenerator)