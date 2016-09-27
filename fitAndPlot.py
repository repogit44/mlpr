import numpy as np
import matplotlib.pyplot as plt
from ploty import plotFunc

yy = np.array([1.1, 2.3, 2.9])
X = np.array([0.8, 1.9, 3.1])[np.newaxis].T
plt.clf()
plt.plot(X, yy, 'x', MarkerSize=20, LineWidth=2)


# plt.show()

def phiLinear(Xin):
    assert Xin.shape[0] >= Xin.shape[1] and Xin.shape[1] == 1
    return np.hstack([np.ones((max(Xin.shape), 1)), Xin])


# print phiLinear(X)

def phiQuadratic(Xin):
    assert Xin.shape[0] >= Xin.shape[1] and Xin.shape[1] == 1
    ones = np.ones((max(Xin.shape), 1))
    Xin_squared = np.square(Xin)
    return np.hstack([ones, Xin, Xin_squared])


# print phiQuadratic(X)

def getRadialBasisFunction(xx, cc):
    return np.exp(-((xx - cc) ** 2) / 2)


# print getRadialBasisFunction(X, 1)

a = np.arange(-10, 10, 0.1)
y = getRadialBasisFunction(a, 5)
plotFunc(a, y)


def phiRadialBasisFunction(Xin):
    return np.hstack([
        getRadialBasisFunction(Xin, 1),
        getRadialBasisFunction(Xin, 2),
        getRadialBasisFunction(Xin, 3),
    ])


# print phiRadialBasisFunction(X)


def fit_and_plot(phi_fn, X, yy):
    """phi_fn takes Nx1 inputs and returns NxD basis function values"""
    # w_fit = phi_fn(X) \ yy; % Dx1
    wFit = np.linalg.lstsq(phi_fn(X), yy)[0]
    X_grid = np.arange(0, 4, 0.01)[np.newaxis].T
    f_grid = phi_fn(X_grid).dot(wFit)
    # plot(X_grid, f_grid, 'LineWidth', 2);
    plotFunc(X_grid, f_grid)


fit_and_plot(phiLinear, X, yy)

plt.plot(X, yy, 'x', MarkerSize=20, LineWidth=2)
fit_and_plot(phiQuadratic, X, yy)

plt.plot(X, yy, 'x', MarkerSize=20, LineWidth=2)
fit_and_plot(phiRadialBasisFunction, X, yy)
