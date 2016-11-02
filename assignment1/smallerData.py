from scipy import io
import ploty
import numpy as np
from matplotlib import pyplot as plt

ampData = io.loadmat('amp_data.mat')
amps = ampData['amp_data']
dataLen = len(amps)

t = np.arange(0, dataLen)

np.save('fifth', amps[:dataLen/10])