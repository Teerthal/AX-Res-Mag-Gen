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

phi_initial = sqrt(282)
phi_final = p / sqrt(2)
phi_prime_initial = -0.1

N_initial = -70
N_final = 0

alpha = 100*f

epsilon_final = 1
phi_prime_final = sqrt(2*epsilon_final)
a_f = 1
H_f = 1e-5
a_initial = a_f*exp(N_initial)

epsilon_initial = 0.00428117288666
H_initial = H_f*2/(3- epsilon_initial)*(phi_initial/phi_final)**2
k_ins_min = alpha/f*a_initial*abs(phi_prime_initial)*H_initial
k_ins_max = alpha/f*a_f*abs(phi_prime_final)*H_f

k = 100*k_ins_min;print('k:%str' %k)
for k in [100*k_ins_min,200*k_ins_min, 500*k_ins_min]:

    for h in [-1]:
        def ODE(N, u):
            eps = 1 / 2 * u[1] ** 2
            H = H_f * sqrt(2 / (3 - eps) * (u[0] / phi_final) ** 2)
            a = a_f * exp(N)

            return np.array(
                [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p),
                 u[3], -(1 - eps) * u[3] - ((k / (a * H)) ** 2 - (h * alpha * k / (f * a * H)) * u[1]) * u[2]])


        r = ode(ODE).set_integrator('zvode', method='bdf', order=5, rtol=1e-6, atol=1e-6, nsteps=1e6)

        # Initial Condition
        t_initial = -1
        A_initial = (2*k)**(1/2)*exp(-1j * k * t_initial)
        A_prime_initial = -1j * k * A_initial / H_initial

        init = np.array([phi_initial, phi_prime_initial, A_initial, A_prime_initial])
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
        plt.plot(N, A, label = '%d%str'%(h,k))
        plt.legend()

        epsilon = 1/2*phi_prime**2
        H = H_f * sqrt(2 / (3 - epsilon) * (phi / phi_final) ** 2)
        a = a_f * exp(N)
        plt.subplot(212)
        plt.semilogy(N, (k/(a*H))/(alpha/f*(sqrt(2*epsilon))))
plt.show()

plt.subplot(311)
plt.plot(N, phi)
plt.subplot(312)
plt.plot(N, k/(a*H))

xi = alpha/f*phi_prime
plt.subplot(313)
plt.plot(N, xi)

plt.show()

