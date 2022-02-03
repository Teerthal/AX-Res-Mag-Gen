import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
plt.rcParams['axes.labelsize'] = 20
pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

file = open('/home/teerthal/Repository/Gauge_Evolution/Separate/0107/separate_1.npy', 'rb')
stack = np.load(file)
print(np.shape(stack))

p = 2
f = 0.1
alpha = 20*f

N_final = 70.
N_initial = 0.

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

def a(N):
    return exp(N-N_final)

def phi(N):
    return sqrt(2*p*(N_final-N_initial)+phi_final**2)

def phi_prime(N):
    return -p / phi(N)

def epsilon(N):
    return (phi_prime(N)) ** 2 / 2.

def H(N):
    return H_initial * sqrt((3. - epsilon(N)) / 2. * (phi(N) / phi_initial) ** 2)

def xi(N):
    return alpha/f*epsilon(N)

def k_max(N):

    return a(N)*H(N)*xi(N)

def V(N):
    return (1.9/1.2e-6)**2*phi(N)**2/2

def rho_phi(N):

    return phi_prime(N)**2/2 + V(N)

print(phi_prime(N_final))

for b_index in [0,1,2,3,4]:

    b = np.array([0, .1, .2, .3, .5])

    b = b[b_index]

    N_k = stack[:,b_index,0,0,0]

    k = a(N_final)*H(N_final)*exp(N_k-N_final)
    k_max = a(N_final)*H(N_final)*exp(62-N_final)

    A = stack[:,b_index,0,0,1]

    Integrand = 1/(4*pi**2)*exp(4*(N_k-N_final))*H(N_final)**4*A**2/k

    rho_B = trapz(Integrand, x=k)

    B_end = sqrt(2*rho_B)*exp(-70*2)
    B_max = 1/(4*pi**2)*exp(4*(62-N_final))*H(N_final)*max(A);print(B_max)
    lamda = trapz(pi*k**2*pi*A**2)/trapz(pi*k**2*k/2*A**2)*exp(-70);print('lamda end:%s'%lamda)

    Mpl = 1.2e19

    T0 = 2.3e-3
    TR = 1e14
    af = T0/Mpl*(TR/Mpl)**(1/3)*(H(N_final)*Mpl**2)**(-2/3);print('a final:%s'%af)
    z_rec = 1000

    B_present = ((1+z_rec)*T0**4/Mpl)**(1/3)*af*(B_max*Mpl**2)**(2/3)*(k_max*Mpl)**(-1/3)/(6.8e-20);print('B present:%s'%B_present)

    rho_rec = T0**4*(1+z_rec)**4

    lambda_present = af**2/(k_max*Mpl)*(B_max*Mpl**2)**(2/3)*(rho_rec/Mpl)**(-2/3)*6.4e-39;print('lambda now',lambda_present)
    print(rho_B/H(N_final)**2)
    plt.subplot(211)
    plt.loglog(k,A)
    plt.subplot(212)
    plt.semilogy(N_k,A)
plt.show()