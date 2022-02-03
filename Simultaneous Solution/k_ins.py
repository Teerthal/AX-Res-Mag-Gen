import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

p = 2
f = 0.1
b = 0
for b in np.linspace(0,1./f,3):

    phi_initial = sqrt(282)
    phi_final = p / sqrt(2)
    phi_prime_initial = -0.1

    N_initial = 0
    N_final = 20

    alpha = 7 * f


    def system(N, u):
        eps = u[1] ** 2
        return [[u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p)]]


    r = ode(system).set_integrator('vode', method='bdf', order=5, atol=1e-12, rtol=1e-12,
                                   with_jacobian=False)

    init = np.array([phi_initial, phi_prime_initial])
    r.set_initial_value(init, N_initial)

    u = []
    t = []
    while r.successful() and r.t <= N_final:
        r.integrate(N_final, step=True)
        u.append(r.y)
        t.append(r.t)

    phi = np.array([item[0] for item in u])
    phi_prime = np.array([item[1] for item in u])
    N = np.array(t)

    print(np.shape(t))

    epsilon = 1 / 2 * phi_prime ** 2
    epsilon_inital = 1 / 2 * phi_prime[100] ** 2;
    print(epsilon_inital)
    epsilon_final = 1
    phi_prime_final = sqrt(2 * epsilon_final)
    a_initial = 1
    H_initial = 1
    H_f = H_initial * sqrt(3 / 2 * (phi_final / phi_initial) ** 2)

    a = a_initial * exp(N)
    a_f = a_initial * exp(N_final)

    H = H_f * sqrt(2 / (3 - epsilon) * (phi / phi_final) ** 2)

    k_ins = alpha / f * a * abs(phi_prime) * H

    k_ins_min = alpha / f * a_initial * abs(phi_prime_initial) * H_initial / a_f
    k_ins_max = alpha / f * a_f * abs(phi_prime_final) * H_f / a_f
    k = np.linspace(k_ins_min, k_ins_max, len(phi))
    print(k_ins_min, k_ins_max)
    print(log(k_ins_max), log(k_ins_min))
    k_steps = (log(k_ins_max) - log(k_ins_min)) * 100
    print(k_steps)

    a_final = a_initial * exp(N_final)
    k = np.linspace(1, 1000 * k_ins_max, len(N))

    xi = alpha / f * phi_prime
    Asym = 1 / sqrt(2 * k) * exp(pi * abs(xi)) / sqrt(2 * pi * abs(xi))

    plt.plot(N, xi, label = 'b:%.2f'%b)
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel(r'$\xi$')
    plt.title('f:%.3f'%f)
plt.legend()
plt.show()

plt.subplot(211)
plt.plot(N,phi)
plt.subplot(212)
plt.plot(N,epsilon)
plt.show()

plt.semilogy(N,1/(a*H))
plt.show()

plt.subplot(211)
plt.plot(N,(k/(a*H))**2)
plt.subplot(212)
plt.semilogy(N,(k_ins/(a*H))**2)
plt.show()