import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
from multiprocessing import Pool
from scipy.integrate import trapz
import functools
from multiprocessing import cpu_count

start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;
sin = np.sin;
cos = np.cos;
exp = np.exp;
log = np.log;
abs = np.abs;
absolute = np.absolute;
sqrt = np.sqrt

delta = 5.
p = 2

F = 0.0025
alpha_0 = 20.

N_final = 70.
N_initial = 0.

Offset = 6;

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1


def a(N):
    return exp(N-N_final)

def phi(N):
    return sqrt(2 * p * (N_final - N) + phi_final ** 2)


def phi_prime(N):
    return -p / phi(N)

def f(N):
    f_power = F*(phi(N))**2
    f_constant = F
    return f_constant

b = 0./f(N_final)
print('Amplitude:',b*f(N_final))
def N_stop(b):
    return N_final + b * f(N_final) * Offset

def Scalar_ODE(N, u):
    eps = 1 / 2 * u[1] ** 2
    return np.array(
        [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f(N))) * (eps - 3) / (u[0] ** p)])


solver = ode(Scalar_ODE).set_integrator('vode')
Scalar_init = np.array([phi(N_initial), phi_prime(N_initial)])
solver.set_initial_value(Scalar_init, N_initial)

# Scalar ODE solving steps

steps = 2e3
dt = (N_stop(b) - N_initial) / steps

temp_1 = []
temp_2 = []
while solver.successful() and solver.t <= N_stop(b):
    solver.integrate(solver.t + dt)
    temp_1.append(solver.t)
    temp_2.append(solver.y)

def Phi_field_plot():
    phi_solved = np.array([item[0] for item in temp_2])
    phi_prime_solved = np.array([item[1] for item in temp_2])
    plt.subplot(211)
    plt.plot(temp_1, phi_solved)
    plt.subplot(212)
    plt.plot(temp_1, phi_prime_solved)
    return plt.show()

Phi_field_plot()

def Phi(N):
    phi_solved = np.array([item[0] for item in temp_2])
    index_N = int(abs(N / N_final * steps - 1))

    return phi_solved[index_N]

def Phi_Prime(N):
    phi_prime_solved = np.array([item[1] for item in temp_2])
    index_N = int(abs(N / N_final * steps - 1))

    return phi_prime_solved[index_N]

def epsilon(N):
    return (Phi_Prime(N)) ** 2 / 2.

def H(N):
    return abs(H_initial * sqrt((3. - epsilon(N)) / (3. - epsilon(N_initial)) * (Phi(N) / phi_initial) ** 2))

def N_k(N_start):
    return delta + N_start

def k(N_start):
    k = a(N_final) * H(N_final) * exp(N_k(N_start) - N_final)
    return k

Gauge_steps = int(1e4)

def Coupling_Plot():
    plt.semilogy(temp_1, alpha_0*f(60)/np.array([*map(f, temp_1)]));plt.show()
    return plt.show()

Coupling_Plot()

def ODE(N, u, arg):
    N_k = arg[0]
    h = arg[1]
    alpha = arg[2]
    Phi_Prime = arg[3](N)

    return np.array(
        [u[1],
         -u[1] - (
             exp(2 * (N_k - N)) - exp(N_k - N) * h * alpha / f(N) * Phi_Prime) * u[0]])


def core(N_start, h, alpha):

    Parameters = np.array([N_k(N_start), h, alpha, Phi_Prime])

    r = ode(ODE).set_integrator('zvode').set_f_params(Parameters)

    # Initial Condition
    t_initial = -1
    A_initial = exp(-1j * k(N_start) * t_initial)
    A_prime_initial = -1j * A_initial * exp(N_k(N_start) - N_start)

    init = np.array([A_initial, A_prime_initial]);

    r.set_initial_value(init, N_start)

    u = []
    t = []

    dt = (N_final - 2. - N_start) / Gauge_steps
    while r.successful() and r.t <= N_final - 2:
        r.integrate(r.t + dt)
        u.append(r.y)
        t.append(r.t)

    A = [item[0] for item in u]
    A_prime = [item[1] for item in u]

    if len(t) > Gauge_steps:
        A.pop()
        A_prime.pop()
        t.pop()

    A = absolute(np.array(A))
    A_prime = absolute(np.array(A_prime))
    N = np.array(t)

    return np.array([N, A, A_prime])

def execute(N_start):

    list_1 = []
    for alpha in [alpha_0 * f(60)]:
        list_2 = []
        for h in [-1]:
            list_2.append(core(N_start, h, alpha))
        list_1.append(list_2)

    return np.array([list_1])

N_start = np.linspace(55.,63.,cpu_count())

pool = Pool(cpu_count())

if __name__ == '__main__':
    stack = np.array(pool.map(execute, N_start))
    pool.close()

print('Shape of stack:', np.shape(stack))

end = time.time()
print(end - start)

N = stack[0, 0, 0, 0, 0]
N_index = np.arange(0, int(Gauge_steps), 1)
alpha_index = 0

k_map = np.array([*map(k, N_start)])
a_map = np.array([*map(a, N)])
H_map = np.array([*map(H, N)])


def Gauge_curves():
    k_map = np.array([*map(k, N_start)])
    for i in np.arange(0, len(N_start), 1):
        k_i = k_map[i]
        plt.subplot(211)
        plt.ylabel(r'$|\mathcal{A}|$')
        plt.semilogy(N, stack[i, 0, 0, 0, 1], label='k:%s' % k_i)
        plt.legend()
        plt.subplot(212)
        plt.ylabel(r'$|\frac{d\mathcal{A}}{dx}|$')

        A_prime = a_map*H_map/(k_map[i])*stack[i, 0, 0, 0, 2]
        plt.semilogy(N, A_prime, label='k:%s' % k_i)
        plt.legend()
    return plt.show()


def Asymptotic_plot():
    plt.loglog(k_map/a(N_final), stack[:,0,0,0,1,-1], label = r'$\rho_B$')
    plt.loglog(k_map/a(N_final), stack[:, 0, 0, 0, 2, -1], label = r'$\rho_E$' )
    plt.legend()
    return plt.show()

Asymptotic_plot()

def A(N_index):
    return stack[:, 0, 0, 0, 1, N_index]

def A_prime(N_index):
    return stack[:,0,0,0,2,N_index]

def rho_B(N_index):

    Integrand = 1 / (4 * pi ** 2) * (A(N_index) ** 2 -1) * (k_map)** 4 / k_map
    return trapz(Integrand, x=k_map)

pool = Pool(cpu_count())

if __name__ == '__main__':
    rho_B_map = np.array(pool.map(rho_B, N_index))
    pool.close()

print('Shape of rho_B_map:', np.shape(rho_B_map))

end = time.time()
print('Elapsed Time:%s' % (end - start))

def rho_E(N_index):

    Integrand = 1 / (4 * pi ** 2) * ((A_prime(N_index)/k_map) ** 2 - k_map**2) * (k_map)** 4 / k_map
    return trapz(Integrand, x=k_map)

pool = Pool(cpu_count())

if __name__ == '__main__':
    rho_E_map = np.array(pool.map(rho_E, N_index))
    pool.close()

print('Shape of rho_E_map:', np.shape(rho_E_map))

end = time.time()
print('Elapsed Time:%s' % (end - start))


def rho_plot():
    plt.semilogy(N, H_map ** 2, label=r'$\phi$')
    plt.semilogy(N, rho_B_map , label=r'$\rho_B$')
    plt.semilogy(N, rho_E_map*(a_map*H_map)**2, label = r'$\rho_E$')

    plt.legend()
    return plt.show()

rho_plot()

