import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 20
pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

file = open('/home/teerthal/Repository/Gauge_Evolution/Simul/Simul_Nk/Rescaled/0107/rescaled_3.npy', 'rb')
stack = np.load(file)
print(np.shape(stack))

scale = 0
f = 0.1
b=0
plt.semilogy(stack[:,0,b,0,scale],stack[:,0,b,0,2], label = ('bf:0'))
b=1
plt.semilogy(stack[:,0,b,0,scale],stack[:,0,b,0,2], label = ('bf:0.1'))
b=2
plt.semilogy(stack[:,0,b,0,scale],stack[:,0,b,0,2], label = ('bf:0.2'))
b=3
plt.semilogy(stack[:,0,b,0,scale],stack[:,0,b,0,2], label = ('bf:0.4'))
b=4
plt.semilogy(stack[:,0,b,0,scale],stack[:,0,b,0,2], label = ('bf:0.5'))

plt.title(['alpha/f:20', 'f:0.05','$N_i-N_k=6$'])
plt.xlabel(r'$\mathcal{N}_k$')
plt.ylabel(r'$\sqrt{2k}|\mathcal{A}_\pm|$')
plt.legend()
plt.show()