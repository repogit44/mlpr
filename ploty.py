import matplotlib.pyplot as plt

def plotFunc(x, y):
    startPlot(x, y)
    plt.show()
    plt.hold('off')


def startPlot(x, y):
    plt.hold('on')
    plt.plot(x, y, 'b-')
    plt.plot(x, y, 'r.')

def plotMultiple(x, Y):
    plt.clf()
    plt.hold('on')
    for i in range(len(Y)):
        plt.plot(x, Y[i], 'b-')
        plt.plot(x, Y[i], 'r.')
    plt.hold('off')
    plt.show()


def plot3D(x, y, z):
    fig = plt.figure(figsize=(8, 8))    #8 by 8 inches
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, 'r.', ms=2)
    #ax.plot(inputs[:, 0], inputs[:, 1], outputs[:, 0], 'b.', ms=2)

    ax.set_xlabel('Input x')
    ax.set_ylabel('Input y')
    ax.set_zlabel('Output')
    #ax.legend(['Targets', 'Predictions'], frameon=False)
    fig.tight_layout()
