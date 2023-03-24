
import numpy as np
import math
from matplotlib import pyplot as plt

p0 = 10

def F(x):
    return x/(x+p0)

xs = np.linspace(0,100,1000)


def Fprima(x):
    return p0/(x+p0)**2

plt.plot(xs,F(xs))
plt.show()
