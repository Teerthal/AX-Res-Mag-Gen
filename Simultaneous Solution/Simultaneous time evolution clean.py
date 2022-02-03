import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

p = 2
f = 0.1
alpha = 100*f

N_initial = 40
N_final = N_initial+30

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

def a(N):
    return exp(-N_initial+N)

def phi(N):
    return phi_final + (N - N_final) * (phi_final - phi_initial) / (N_final - N_initial)

def phi_prime(N):
    return -p/phi(N)

def epsilon(N):
    return (phi_prime(N))**2 / 2.

def H(N):
    return H_initial*sqrt((3.-epsilon(N)) / 2. * (phi(N) / phi_initial) ** 2)

def k_ins(N):
    return alpha/f*a(N)*abs(phi_prime(N))*H(N)/a(N_final)
print(a(N_initial))
def N_k(k):
    return N_initial + log(k/(a(N_initial)*H(N_initial)))

print(k_ins(N_initial),k_ins(N_final))
print(N_k(k_ins(N_initial)),N_k(k_ins(N_final)))
asyms_k = []
for k in np.geomspace(100,1000,3):
    print([k,N_k(k)])
    asyms_b = []
    for b in [0]:
        asyms_h = []
        for h in [-1]:

            def ODE(N, u):

                eps = 1 / 2 * u[1] ** 2
                return np.array(
                    [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p),
                     u[3],
                     -(1 - epsilon(0)) * u[3] - (
                         exp(-2 * (N - N_k(k)) * (1 - eps)) - exp(-(N - N_k(k)) * (1 - epsilon(0))) * (
                             1 - epsilon(0)) * h * alpha/f * u[
                             1]) * u[
                         2]])

            r = ode(ODE).set_integrator('zvode', method='bdf', order=5, rtol=1e-15, atol=1e-15, nsteps=1e6)

            # Initial Condition
            t_initial = -1
            A_initial =  exp(-1j * k * t_initial)/sqrt(2*exp(N_k(k)-N_initial))
            A_prime_initial = -1j * exp((N_k(k)-N_initial)*(1-epsilon(0))) * A_initial
            print(A_prime_initial)
            init = np.array([phi(N_initial), phi_prime(N_initial), A_initial, A_prime_initial])
            r.set_initial_value(init, N_initial)

            u = []
            t = []
            while r.successful() and r.t <= N_final:
                r.integrate(N_final, step=True)
                u.append(r.y)
                t.append(r.t)

            Phi = np.array([item[0] for item in u])
            Phi_prime = np.array([item[1] for item in u])
            A = absolute(np.array([item[2] for item in u]))*sqrt(2*exp(N_k(k)-N_initial))
            A_prime = np.array([item[3] for item in u])
            N = np.array(t)

            plt.plot(N,A, label = k)
            plt.legend()
            asyms_h.append(np.array([N_k(k),A[-1]]))
        asyms_b.append(np.array(asyms_h))
    asyms_k.append(np.array(asyms_b))
plt.show()
print(np.shape(asyms_k))
exit()
stack = np.array(asyms_k)
plt.semilogy(stack[:,0,0,0],stack[:,0,0,1])
plt.show()
