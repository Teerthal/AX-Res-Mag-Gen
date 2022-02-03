import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

p = 2
f = 0.05
b = 0
alpha = 10*f

N_initial = 0
N_final = 70
N_start = 0

#Scalar field initial conditions
phi_initial = sqrt(282)
phi_final = p / sqrt(2)
phi_prime_initial = -0.1

phi_start = phi_final + (N_start - N_final) * (phi_final - phi_initial) / (N_final - N_initial);print('phi_start:%str'%phi_start)
phi_prime_start = -p/phi_start;print('phi_prime_start:%str'%phi_prime_start)

epsilon_final = 1
phi_prime_final = sqrt(2*epsilon_final)

a_initial = 1
H_initial = 1
a_final = a_initial*exp(N_final)

k_min = alpha/f*a_initial*abs(phi_prime_initial)*H_initial;print(k_min)
k_max = alpha/f*a_final*abs(phi_prime_final)*H_initial;print(k_max)

Asyms = []
l = np.linspace(k_min*1e3,1e5*k_max/a_final, 24)
for k in [l[0]]:
    Asyms_h = []
    for h in [-1,1]:
        file = open('/home/teerthal/Repository/Gauge_Evolution/Simul/Simul/A_%d_%d.npy' % (k, h), 'rb')
        stack = np.load(file)
        Asyms_h.append([k,stack[1,len(stack)]])
        print(np.shape(stack))
        file.close()
        plt.title(k)
        plt.semilogy(stack[0,:],stack[1,:])
    plt.show()
    Asyms.append(Asyms_h)

Asyms = np.array(Asyms)
print(np.shape(Asyms))
plt.semilogy(Asyms[:,0,0],Asyms[:,0,1], 'r')
plt.semilogy(Asyms[:,1,0],Asyms[:,1,1], 'b')
plt.show()
