import numpy as np
from numpy import random as rnd
import math

#related to tutorial week 4, ex. 3
m = 1
sigma = 4
a = 5
n = 9

N = 100000

x1 = rnd.normal(m, sigma, N)
#print x1.shape

v = rnd.normal(0, n, N) #rnd.randn(N)rnd.normal(m, sigma, N)

x2 = a * x1 + v
#print x2.shape

x = np.array([x1, x2]).T
print np.mean(x, axis=0)
print np.array([m, a*m])    #by theory

print

print np.cov(x1, x2)
#by theory:
print np.array([[math.pow(sigma, 2),    a * math.pow(sigma,2)],
                [a * math.pow(sigma,2), (math.pow(a*sigma,2) + math.pow(n,2))]])