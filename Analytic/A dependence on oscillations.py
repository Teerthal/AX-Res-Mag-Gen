import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
from multiprocessing import Pool
from scipy.integrate import trapz
import functools
from multiprocessing import cpu_count

# from numdifftools import Derivative
flatten = np.ndarray.flatten
# from scipy.interpolate import UnivariateSpline
# from scipy import interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess

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

p = 2

F = 0.0025
alpha_0 = 10.

N_final = 70.
N_initial = 0.

Offset = 5;

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

b_i = 0.0
b_f = 0.028
b_list_size = 2

steps = int(9.4e4)
Gauge_steps = int(9.5e4)

N_start = 55.

cpu = 4

N_start_intervals = int(1)
delta_intervals = N_start_intervals

alpha_i = 10.
alpha_f = 20.
alpha_list_size = 1

phi_cmb = 15

# PARAMETERS FOR LOOPS
####################################

# f_list = np.array([0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.0001])
f_list = np.array([0.00023])
####################################

index_b = np.arange(0, b_list_size, 1)
index_f = np.arange(0, len(f_list), 1)
index_alpha = np.arange(0, alpha_list_size, 1)

N_phi = np.linspace(N_initial, N_final, steps)
N = np.linspace(N_start, N_final - 2., Gauge_steps)
N_index = np.arange(0, int(Gauge_steps), 1)


def b_list(item_f):
    b = np.linspace(b_i, b_f, b_list_size) / item_f
    return b


def alpha_list(item_f):
    alpha = np.linspace(alpha_i, alpha_f, alpha_list_size) * item_f

    return alpha
def a(N):
    return exp(N - N_final)


def phi(N):
    return sqrt(2 * p * (N_final - N) + phi_final ** 2)


def phi_prime(N, delta_phi):

    power_roll_solution = -p / phi(N)
    constant = delta_phi

    return constant

def H(N, delta_phi):
    Epsilon = phi_prime(N, delta_phi) ** 2 / 2
    return abs(H_initial * sqrt((3. - Epsilon) / (3. - Epsilon) * (phi(N) / phi_initial) ** 2))

def N_k(delta):
    return delta + N_start

a_map = np.array([*map(a, N)])

def k(delta, delta_phi):
    phi_prime_solved = np.array([*map(functools.partial(phi_prime, delta_phi=delta_phi), N)])
    xi = alpha_i*phi_prime_solved
    N_k_delta = N_k(delta)
    index = int(N_k_delta/N_final*steps)
    xi_N_k = abs(xi[index])
    k = a(N_k_delta)*H(N_k_delta, delta_phi)*xi_N_k

    return k


def ODE(N, u, arg):
    N_k = arg[0]
    h = arg[1]
    alpha = arg[2]
    index_b = arg[3]
    index_f = arg[4]
    delta_phi = arg[7]
    Phi_Prime = arg[5](N, delta_phi)
    k_delta = arg[6]


    return np.array(
        [u[1],
         -u[1] - (k_delta ** 2 / (a(N)*H(N, delta_phi))**2
                  - k_delta / (a(N)*H(N, delta_phi))
                  * h * alpha / f_list[index_f] * Phi_Prime) * u[0]])

def core(delta, h, alpha, index_b, index_f, delta_phi):

    Parameters = np.array([N_k(delta), h, alpha, index_b, index_f, phi_prime, k(delta, delta_phi), delta_phi])

    r = ode(ODE).set_integrator('zvode').set_f_params(Parameters)

    # Initial Condition
    t_initial = -1
    A_initial = exp(-1j * k(delta, delta_phi) * t_initial)
    A_prime_initial = -1j * A_initial * k(delta, delta_phi) / (a(N_start)*H(N_start, delta_phi))

    init = np.array([A_initial, A_prime_initial]);
    r.set_initial_value(init, N_start)

    u = []
    t = []

    dt = (N_final - 2. - N_start) / Gauge_steps
    while r.successful() and r.t<=N_final - 2.:
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

    #Normalising

    N_k_del = N_k(delta)
    index = int(N_k_del/(N_final-2.)*Gauge_steps)
    N_Gauge = np.linspace(N_start, N_final - 2., Gauge_steps)
    a_map_Gauge = np.array([*map(a, N)])
    H_map_Gauge = np.array([*map(functools.partial(H, delta_phi=delta_phi), N)])

    norm_A = []
    norm_A[:index] = A[:index]**2-1
    norm_A[index:] = A[index:]**2

    H_map = np.array([*map(functools.partial(H, delta_phi=delta_phi), N)])

    norm_del_A = []
    norm_del_A[:index] = (A_prime*a_map*H_map_Gauge/k(delta, delta_phi))[:index]**2-1
    norm_del_A[index:] = (A_prime*a_map*H_map_Gauge/k(delta, delta_phi))[index:]**2

    norm_A = np.array(norm_A)
    norm_del_A = np.array(norm_del_A)

    norm_A[norm_A < 0.1] = 0.
    norm_del_A[norm_del_A < 0.1] = 0.

    return np.array([N, A, A_prime, norm_A, norm_del_A])


def execute(delta, delta_phi):
    stack_f = []
    for j in index_f:
        item_f = f_list[j]
        alpha = alpha_list(item_f)
        b = b_list(item_f)

        stack_b = []
        for i in index_b:
            item_b = b[i]

            stack_alpha = []
            for z in index_alpha:
                item_alpha = alpha[z]

                stack_h = []
                for h in [-1, 1]:

                    stack_delta = []
                    for delk in delta:

                        stack_delta_phi =[]
                        for del_phi in delta_phi:

                            stack = np.array(core(delta=delk, h=h, alpha=item_alpha, index_b=i,
                                                                   index_f=j, delta_phi=del_phi))

                            pool = Pool(cpu)


                            #if __name__ == '__main__':
                                #stack = np.array(pool.map(functools.partial(core, h=h, alpha=item_alpha,
                                                                            #index_b=i, index_f=j, delta=8.), delta_phi))
                                #pool.close()

                            stack_delta_phi.append(stack)
                        stack_delta.append(stack_delta_phi)
                    stack_h.append(stack_delta)

    return stack_h


# Parameters for computation start point and wavemodes
####################################

delta = np.array([8.,9.])
delta_phi = np.linspace(0.1,0.5,1)

####################################
stack = np.array(execute(delta, delta_phi))
print(np.shape(stack))

end = time.time()
print(end - start)


def plot_gauge_curves():

    h = [-1.,1.]
    for index_h in [0,1]:
        h_i = h[index_h]

        plt.subplot(211)
        plt.semilogy(stack[index_h, :, 0, 0], stack[index_h, :, 0, 1])
        plt.subplot(212)
        plt.semilogy(stack[index_h, :, 0, 0], stack[index_h, :, 0, 2])
    plt.show()

    return

plot_gauge_curves()