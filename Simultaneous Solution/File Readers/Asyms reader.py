import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 20
pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

file = open('/home/teerthal/Repository/Gauge_Evolution/Simul/Simul_Asyms/Asymptotes.npy', 'rb')
stack = np.load(file)
print(np.shape(stack))


b =1
plt.plot(stack[:,b,0,0],stack[:,b,0,1]*sqrt(2*stack[:,b,0,0]))
b=0
plt.plot(stack[:,b,0,0],stack[:,b,0,1]*sqrt(2*stack[:,b,0,0]))

plt.show()