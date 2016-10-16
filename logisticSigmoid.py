import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ploty import plot3D

a = np.array([2, 5])

print a

grid_size = 0.1
x_grid = np.arange(-10, 10, grid_size)

#f_vals = np.cos(x_grid)

plt.clf()

def plotFunc(x, y):
    plt.hold('on')
    plt.plot(x, y, 'b-')
    plt.plot(x, y, 'r.')
    plt.show()
    plt.hold('off')

sigma = lambda v, x, b: 1 / (1 + math.exp(-v*x - b))

print sigma(1,3,5)

def getSigmoid(v, b, x):
    arr = (-v * x) - b
    result = np.exp(arr) + 1
    #result**(-1) == (1/result)
    return result**(-1)

plotFunc(x_grid, getSigmoid(1, 0, x_grid))
#plotFunc(x_grid, getSigmoid(-1, 0, x_grid))
#plotFunc(x_grid, getSigmoid(-10, 0, x_grid))    #v changes the bandwidth
#plotFunc(x_grid, getSigmoid(-1, 5, x_grid)) #b changes the center

xLen = len(x_grid)
X, Y = np.meshgrid(x_grid, x_grid)
temp = -(X + 2*Y + 5)
Z = (1 + np.exp(temp))**(-1)
print Z
plt.clf()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
plt.show()