import matplotlib.pyplot as plt
from mpmath import sech
import numpy as np
pi = np.pi

x = np.linspace(0.1,1, 100)
f = 9*pi**4/(16*x**2)*sech(3*pi**2/4*x)
plt.plot(x, f)
plt.show()