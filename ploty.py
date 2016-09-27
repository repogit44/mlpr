import matplotlib.pyplot as plt

def plotFunc(x, y):
    startPlot(x, y)
    plt.show()
    plt.hold('off')


def startPlot(x, y):
    plt.hold('on')
    plt.plot(x, y, 'b-')
    plt.plot(x, y, 'r.')