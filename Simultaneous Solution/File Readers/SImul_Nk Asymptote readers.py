import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 20
pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

file = open('/home/teerthal/Repository/Gauge_Evolution/Simul/Simul_Nk/2806/asyms_7.npy', 'rb')
stack = np.load(file)
print(np.shape(stack))


b=0
plt.semilogy(stack[:,b,0,0],stack[:,b,0,1], label = (0,'-'))
plt.semilogy(stack[:,b,1,0],stack[:,b,1,1], label = (0,'+'))
b=1
plt.semilogy(stack[:,b,0,0],stack[:,b,0,1], label = (0.5,'-'))
plt.semilogy(stack[:,b,1,0],stack[:,b,1,1], label = (0.5,'+'))
b=2
plt.semilogy(stack[:,b,0,0],stack[:,b,0,1], label = (0.3,'-'))
plt.semilogy(stack[:,b,1,0],stack[:,b,1,1], label = (0.3,'+'))


plt.title(['alpha/f:10', 'f:0.05','$N_i-N_k=6$'])
plt.xlabel(r'$\mathcal{N}_k$')
plt.ylabel(r'$\sqrt{2k}|\mathcal{A}_\pm|$')
plt.legend()
plt.show()