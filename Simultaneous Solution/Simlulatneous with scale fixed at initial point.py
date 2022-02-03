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

phi_initial = sqrt(282)
phi_final = p / sqrt(2)
phi_prime_initial = -0.1

N_initial = 0
N_final = 70

epsilon_final = 1
phi_prime_final = sqrt(2*epsilon_final)

a_initial = 1
H_initial = 1
a_final = a_initial*exp(N_initial)

k_min = alpha/f*a_initial*abs(phi_prime_initial)*H_initial;print(k_min)
k_max = alpha/f*a_final*abs(phi_prime_final)*H_initial;print(k_max)

for k in np.linspace(k_min,1000*k_max, 3):

    for h in [-1]:
        def ODE(N, u):
            eps = 1 / 2 * u[1] ** 2
            H = H_initial * sqrt(3/(3-eps) * (u[0] / phi_initial) ** 2)
            a = a_initial * exp(N)

            return np.array(
                [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p),
                 u[3], -(1 - eps) * u[3] - ((k / (a * H)) ** 2 - (h * alpha * k / (f * a * H)) * u[1]) * u[2]])


        r = ode(ODE).set_integrator('zvode', method='bdf', order=5, rtol=1e-6, atol=1e-6, nsteps=1e6)

        # Initial Condition
        t_initial = -1
        A_initial = (2*k)**(1/2)*exp(-1j * k * t_initial)
        A_prime_initial = -1j * k * A_initial / (a_initial*H_initial)

        init = np.array([phi_initial, phi_prime_initial, A_initial, A_prime_initial])
        print('A_initial:%str'%A_initial, 'A_prime_initial:%str'%A_prime_initial)
        r.set_initial_value(init, N_initial)

        u = []
        t = []
        while r.successful() and r.t <= N_final:
            r.integrate(N_final, step=True)
            u.append(r.y)
            t.append(r.t)

        phi = np.array([item[0] for item in u])
        phi_prime = np.array([item[1] for item in u])
        A = absolute(np.array([item[2] for item in u]))
        A_prime = np.array([item[3] for item in u])
        N = np.array(t)

        print('Shape of N:%d' % np.shape(N))
        plt.subplot(211)
        plt.semilogy(N, A, label = '%d%str'%(h,k))
        plt.legend()

        epsilon = 1/2*phi_prime**2
        H = H_initial * sqrt(3 / (3 - epsilon) * (phi / phi_initial) ** 2)
        a = a_initial * exp(N)
        plt.subplot(212)
        plt.semilogy(N, (k/(a*H))/(alpha/f*(sqrt(2*epsilon))))
plt.show()
