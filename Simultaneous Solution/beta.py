import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

p = 2
f = 0.025
alpha = 20*f

N_final = 70.
N_initial = 0.

def N_stop(b,f):
    return N_final + b*f*6

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

asyms_b = []
for b in [0,0.2/f,0.4/f]:

    def a(N):
        return exp( N)

    def phi(N):
        return sqrt(2*p*(N_final-N_initial)+phi_final**2)
    print('phi initial:',phi(N_initial))
    def phi_prime(N):
        return -p / phi(N)

    def epsilon(N):
        return (phi_prime(N)) ** 2 / 2.


    def H(N):
        return H_initial * sqrt((3. - epsilon(N)) / 2. * (phi(N) / phi_initial) ** 2)


    def k_ins(N):
        return alpha / f * a(N) * abs(phi_prime(N)) * H(N) / a(N_final)

    def Scalar_ODE(N, u):
        eps = 1 / 2 * u[1] ** 2
        return np.array(
            [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p)])

    solver = ode(Scalar_ODE).set_integrator('vode', method = 'adams', order = 12, atol=1e-15,rtol=1e-15)
    Scalar_init = np.array([phi(N_initial),phi_prime(N_initial)])
    solver.set_initial_value(Scalar_init,N_initial)

    #Scalar ODE solving steps

    steps = 2e3
    dt = (N_stop(b,f)-N_initial)/steps

    temp_1=[];temp_2=[]
    while solver.successful() and solver.t<=N_stop(b,f):
        solver.integrate(N_stop(b,f),step=True)
        temp_1.append(solver.t)
        temp_2.append(solver.y)

    Phi = np.array([item[0] for item in temp_2])
    Phi_prime = np.array([item[1] for item in temp_2])

    plt.subplot(211)
    plt.plot(np.array(temp_1)-b*f*6,Phi,label=b*f)
    plt.subplot(212)
    plt.plot(np.array(temp_1)-b*f*6,-alpha/f*Phi_prime,label=b*f)

plt.legend()
plt.show()