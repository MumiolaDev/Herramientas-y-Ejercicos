import numpy as np
from matplotlib import pyplot as plt


def F(x):
    y = []
    y2 = []
    for var in x:
        tmp = var*np.exp(-var*var)
        y.append(tmp)

    return np.array(y)



X = np.linspace(0,1,100)
Y = F(X)



plt.plot(X,Y,".")
#plt.plot(test_x,test_y)
plt.show()
