import numpy as np
from ploty import plotFunc, plotMultiple
import math

a = np.arange(-1, 1, 0.01)

def getRadialBasisFunction(xx, cc):
    return np.exp(-((xx - cc) ** 2) / 2)


# print getRadialBasisFunction(X, 1)

y = getRadialBasisFunction(a, 5)
#plotFunc(a, y)


def getAnotherBasisFunction(xx, cc, hh):    #c affect the center, h affects how wide it is. The higher the h the more wide
    return np.exp(-((xx - cc) ** 2) / (hh**2))

#plotFunc(a, getAnotherBasisFunction(a, 5, 1))
#plotFunc(a, getAnotherBasisFunction(a, 5, 2))
#plotFunc(a, getAnotherBasisFunction(a, 5, 3))

def yetAnotherBasisFunction(xx, k, hh):    #c affect the center, h affects how wide it is. The higher the h the more wide
    cc = (k - 51)*hh / math.sqrt(2)
    return np.exp(-((xx - cc) ** 2) / (hh**2))

Y = [
    yetAnotherBasisFunction(a, 1, 0.02),
    yetAnotherBasisFunction(a, 50, 0.02),
    yetAnotherBasisFunction(a, 101, 0.02)
]

plotMultiple(a, Y)
