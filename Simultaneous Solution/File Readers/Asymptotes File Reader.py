import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 20
pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

file = open('/home/teerthal/Repository/Gauge_Evolution/Simul/Simul_Asyms/Asymptotes.npy', 'rb')
stack = np.load(file)
print(np.shape(stack))

f = np.geomspace(0.1,0.005, 24)
f_index = 0
f = f[f_index]
alpha = np.linspace(1.*f, 10*f, 4)
alpha_index = 2
alpha = alpha[alpha_index]
scale = 1
b = np.linspace(0,1./f, 3)

print('alpha/f:', alpha/f)


p = 2

N_initial = 0
N_final = 20
N_start = 0

#Scalar field initial conditions
phi_initial = sqrt(282.)
phi_final = p / sqrt(2.)
phi_prime_initial = -0.1

phi_start = phi_final + (N_start - N_final) * (phi_final - phi_initial) / (N_final - N_initial);print('phi_start:%str'%phi_start)
phi_prime_start = -p/phi_start;print('phi_prime_start:%str'%phi_prime_start)

epsilon_final = 1.
phi_prime_final = sqrt(2.*epsilon_final)

a_initial = 1.
H_initial = 1
a_final = a_initial*exp(N_final)
H_final = H_initial * sqrt(3. / 2. * (phi_final / phi_initial) ** 2);print(H_final/H_initial)

k_min = alpha / f * a_initial * abs(phi_prime_initial) * H_initial / a_final
k_max = alpha / f * a_final * abs(phi_prime_final) * H_final / a_final

def a(N):
    return exp(N_initial+N)

def phi(N):
    return phi_final + (N - N_final) * (phi_final - phi_initial) / (N_final - N_initial)

def phi_prime(N):
    return -p/phi(N)

def epsilon(N):
    return (phi_prime(N))**2 / 2.

def xi(N):
    return alpha/f*sqrt(epsilon(N))

k = np.geomspace(100. * k_min, 100. * k_max, 5000)
N_pivot = 70
print('xi:',xi(N_pivot))
Analytical = (1/sqrt(2*k))*exp(pi*abs(xi(N_pivot)))/sqrt(2.*pi*abs(xi(N_pivot)))



b_index = 0
plt.loglog(stack[f_index, alpha_index,:,b_index,0,0],stack[f_index, alpha_index,:,b_index,0,scale],label = 'b:%.2f'%b[b_index])
b_index = 1
plt.loglog(stack[f_index, alpha_index,:,b_index,0,0],stack[f_index, alpha_index,:,b_index,0,scale],label = 'b:%.2f'%b[b_index])
b_index = 2
plt.loglog(stack[f_index, alpha_index,:,b_index,0,0],stack[f_index, alpha_index,:,b_index,0,scale],label = 'b:%.2f'%b[b_index])
plt.title('f:%.3f'%f)
plt.loglog(k, Analytical, label = 'Analtical')

plt.legend()
plt.xlabel(r'$k/a_f$')
plt.ylabel(r'$|\mathcal{A}_-|$')
plt.show()