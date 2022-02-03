import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
from multiprocessing import Pool
from scipy.integrate import trapz
import functools
from multiprocessing import cpu_count

flatten = np.ndarray.flatten

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

Offset = 6

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

b_list_size = 1

steps = int(1.4e4)
Gauge_steps = int(1.5e4)

N_start = 55.

N_start_intervals = int(4)
delta_intervals = N_start_intervals

alpha_list_size = 1

phi_cmb = 15

cpu = 24

# PARAMETERS FOR LOOPS
####################################

f_list = np.array([0.01])

####################################

print(f_list)

Master_path = '/home/teerthal/Repository/Gauge_Evolution/Dynamic/Data'

def save(data, name, directory):
    path = '%s/%s/%s.npy' % (Master_path, directory, name)

    file = open(path, 'wb')
    np.save(file, data)
    file.close()
    return

def load(name, directory):
    path = '%s/%s/%s.npy' % (Master_path, directory, name)
    file = open(path, 'rb')
    stack = np.load(file)
    file.close()
    return stack

def a(N):
    return exp(N - N_final)


def phi(N):
    return sqrt(2 * p * (N_final - N) + phi_final ** 2)


def phi_prime(N):
    return -p / phi(N)


def f(N):
    f_power = F * (phi(N)) ** 2
    f_constant = F
    return f_constant


def b_list(item_f):
    b = np.linspace(0.01, 0.5, b_list_size) / item_f
    return b


def alpha_list(item_f):
    alpha = np.linspace(5, 25, alpha_list_size) * item_f

    return alpha


def N_stop(b):
    return N_final + b * f(N_final) * Offset


def Scalar_ODE(N, u, arg):
    b = arg[0]
    f = arg[1]
    eps = 1 / 2 * u[1] ** 2
    return np.array(
        [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p)])


def Scalar_Core(b, f):
    Parameters = np.array([b, f])
    solver = ode(Scalar_ODE).set_integrator('vode').set_f_params(Parameters)
    Scalar_init = np.array([phi(N_initial), phi_prime(N_initial)])
    solver.set_initial_value(Scalar_init, N_initial)

    # Scalar ODE solving steps

    dt = (N_stop(b) - N_initial) / steps

    temp_1 = []
    temp_2 = []
    while solver.successful() and solver.t <= N_stop(b):
        solver.integrate(solver.t + dt)
        temp_1.append(solver.t)
        temp_2.append(solver.y)

    if len(temp_1) > steps:
        temp_1.pop()
        temp_2.pop()

    phi_solved = np.array([item[0] for item in temp_2])
    phi_prime_solved = np.array([item[1] for item in temp_2])

    return [phi_solved, phi_prime_solved]


def Phi_Solver(f):
    stack_f = []
    for item_f in f:

        stack_b = []

        for item_b in b_list(item_f):
            stack_b.append(Scalar_Core(item_b, item_f))
        stack_f.append(stack_b)

    return stack_f

# Globally executed and stored Phi array

Buffer = np.array([Phi_Solver(f_list)])
print('Shape of Phi Buffer:', np.shape(Buffer))

save(Buffer, 'Phi', 'Phi')

index_b = np.arange(0, b_list_size, 1)
index_f = np.arange(0, len(f_list), 1)
index_alpha = np.arange(0, alpha_list_size, 1)


def Phi(N, index_b, index_f):
    phi_solved = Buffer[0, index_f, index_b, 0]
    index_N = int(abs(N / N_final * steps - 1))

    return phi_solved[index_N]


def Phi_Prime(N, index_b, index_f):
    phi_prime_solved = Buffer[0, index_f, index_b, 1]
    index_N = int(abs(N / N_final * steps - 1))

    return phi_prime_solved[index_N]


def Scalar_plot():
    N = np.linspace(N_initial, N_final, steps)
    for j in index_f:
        for i in index_b:
            phi_solved = Buffer[0, j, i, 0]
            phi_prime_solved = Buffer[0, j, i, 1]

            plt.subplot(211)
            plt.plot(N, phi_solved)
            plt.subplot(212)
            plt.plot(N, phi_prime_solved)

    plt.show()
    return

Scalar_plot()

def epsilon(N, index_b, index_f):
    return (Phi_Prime(N, index_b, index_f)) ** 2 / 2.


def H(N):
    Epsilon = phi_prime(N) ** 2 / 2
    return abs(H_initial * sqrt((3. - Epsilon) / (3. - Epsilon) * (phi(N) / phi_initial) ** 2))


N = np.linspace(N_initial, N_final, steps)


def N_k(delta):
    return delta + N_start


def k(delta):
    k = a(N_final) * H(N_final) * exp(N_k(delta) - N_final)
    return k


def ODE(N, u, arg):
    N_k = arg[0]
    h = arg[1]
    alpha = arg[2]
    index_b = arg[3]
    index_f = arg[4]
    Phi_Prime = arg[5](N, index_b, index_f)

    return np.array(
        [u[1],
         -u[1] - (
             exp(2 * (N_k - N)) - exp(N_k - N) * h * alpha / f_list[index_f] * Phi_Prime) * u[0]])


def core(delta, h, alpha, index_b, index_f):
    Parameters = np.array([N_k(delta), h, alpha, index_b, index_f, Phi_Prime])

    r = ode(ODE).set_integrator('zvode').set_f_params(Parameters)

    # Initial Condition
    t_initial = -1
    A_initial = exp(-1j * k(delta) * t_initial)
    A_prime_initial = -1j * A_initial * exp(N_k(delta) - N_start)

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


def execute(delta):
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
                for h in [-1]:

                    pool = Pool(cpu)

                    if __name__ == '__main__':

                        stack = np.array(pool.map(functools.partial(core, h=h, alpha = item_alpha,
                                                           index_b=i, index_f=j), delta))
                        pool.close()

                    stack_h.append(stack)

                print(np.shape(stack_h))
                save(stack_h, 'alpha:%.4f_b:%.4f_f:%.5f'%(item_alpha, item_b, item_f), 'Raw')

    return


# Parameters for computation start point and wavemodes
####################################

delta = np.linspace(5., 10, delta_intervals)

####################################

execute(delta)

end = time.time()
print(end - start)

N = np.linspace(N_start, N_final, Gauge_steps)
N_index = np.arange(0, int(Gauge_steps), 1)

k_map = np.array([*map(k, delta)])
a_map = np.array([*map(a, N)])
H_map = np.array([*map(H, N)])

def Gauge_curves():
    for i in np.arange(0, delta_intervals, 1):

        k_i = k_map[i]

        for l in index_f:
            item_f = f_list[l]
            b = b_list(item_f)
            alpha = alpha_list(item_f)

            for j in index_b:
                item_b = b[j]

                for z in index_alpha:
                    item_alpha = alpha[z]

                    stack = load('alpha:%.4f_b:%.4f_f:%.5f'%(item_alpha, item_b, item_f), 'Raw')
                    print(np.shape(stack))
                    plt.subplot(211)
                    plt.ylabel(r'$|\mathcal{A}|$')
                    plt.semilogy(N, stack[0,i, 1], label='k:%s ,b:%s, f:%s' % (k_i, item_b, item_f))
                    plt.legend()
                    plt.subplot(212)
                    plt.ylabel(r'$|\frac{d\mathcal{A}}{dx}|$')

                    A_prime = a_map * H_map / k_i * stack[0,i, 2]
                    plt.semilogy(N, A_prime, label='k:%s' % k_i)
                    plt.legend()

    return plt.show()
Gauge_curves()
def Asymptotes():
    stack_f = []
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        stack_j = []
        for i in index_b:
            b_i = b[i]

            stack_z = []
            for z in index_alpha:
                alpha_z = alpha[z]

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' %(alpha_z, b_i, f_j), 'Raw')

                asymptotes = np.array([delta, stack[0, :, 1, -1], stack[0, :, 2, -1]])

                save(asymptotes, 'alpha:%.4f_b:%.4f_f:%.5f'%(alpha_z, b_i, f_j), 'Asymptotes')

    return

Asymptotes()

def Asymptotic_plot():
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        for i in index_b:
            b_i = b[i]

            for z in index_alpha:
                alpha_z = alpha[z]

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Asymptotes')

                plt.loglog(stack[0] / a(N_final), stack[1], label=r'$\rho_B$')
                plt.loglog(stack[0] / a(N_final), stack[2], label=r'$\rho_E$')
                plt.legend()

    return plt.show()

def A(N_index, b, f, alpha):
    stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha, b, f), 'Raw')

    return stack[0, :, 1, N_index]

def A_prime(N_index, b, f, alpha):

    stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha, b, f), 'Raw')

    return stack[0, :, 2, N_index]

def rho_B(N_index, b, f, alpha):
    Integrand = 1 / (4 * pi ** 2) * (A(N_index, b, f, alpha) ** 2 - 1) * (k_map) ** 4 / k_map
    return trapz(Integrand, x=k_map)

def rho_E(N_index, b, f, alpha):
    Integrand = 1 / (4 * pi ** 2) * ((A_prime(N_index, b, f, alpha) / k_map) ** 2 - k_map ** 2) * (
                                                                                                                    k_map) ** 4 / k_map
    return trapz(Integrand, x=k_map)

def Backreaction_Functional(N_index, b, f, alpha):
    Integrand = k_map ** 2 / (16 * pi ** 2) * (A(N_index, b, f, alpha) ** 2 - 1)
    return trapz(Integrand, x=k_map)

def Spooler_rho_B():
    stack_j = []
    for j in index_f:
        item_f = f_list[j]
        alpha = alpha_list(item_f)
        b = b_list(item_f)

        stack_i = []
        for i in index_b:
            item_b = b[i]

            stack_z = []
            for z in index_alpha:
                item_alpha = alpha[z]

                pool = Pool(cpu)

                if __name__ == '__main__':
                    rho_B_map = pool.map(functools.partial(rho_B, b=item_b,
                                                           f=item_f, alpha=item_alpha), N_index)
                    pool.close()
                plt.plot(N, rho_B_map); plt.show()
                save(rho_B_map, 'alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Rho_B')

    return

Spooler_rho_B()

end = time.time()
print('Elapsed Time:%s' % (end - start))

def Spooler_rho_E():
    stack_j = []
    for j in index_f:
        item_f = f_list[j]
        alpha = alpha_list(item_f)
        b = b_list(item_f)

        stack_i = []
        for i in index_b:
            item_b = b[i]

            stack_z = []
            for z in index_alpha:
                item_alpha = alpha[z]

                pool = Pool(cpu)

                if __name__ == '__main__':
                    rho_E_map = np.array(
                        pool.map(functools.partial(rho_E, b=item_b,
                                                           f=item_f, alpha=item_alpha), N_index))
                    pool.close()

                save(rho_E_map, 'alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Rho_E')

    return

Spooler_rho_E()

end = time.time()
print('Elapsed Time:%s' % (end - start))

def Spooler_Bacreaction():
    stack_j = []
    for j in index_f:

        item_f = f_list[j]
        b = b_list(item_f)
        alpha = alpha_list(item_f)

        stack_i = []
        for i in index_b:

            item_b = b[i]

            stack_z = []
            for z in index_alpha:

                item_alpha = alpha[z]

                pool = Pool(cpu)

                if __name__ == '__main__':
                    Backreaction_map = pool.map(
                        functools.partial(Backreaction_Functional, b=item_b,
                                                           f=item_f, alpha=item_alpha), N_index)
                    pool.close()

                dN = (N_final - 2. - N_start) / Gauge_steps
                # dy = np.array([np.diff(Backreaction_map)])[0]
                # dy.flatten()
                # H = np.array(H_map)[0]

                # print('shape of H_map',np.shape(H))
                # print('shape of dy to H_map', np.shape(dy))

                # Backreaction = alpha_z/f_j*H*dy/dN
                # print('shape of backreaction array to stacker', np.shape(Backreaction))

                y = Backreaction_map

                dy = np.array([t - s for s, t in zip(y, y[1:])])

                Backreaction = item_alpha / item_f * dy / dN

                save(Backreaction, 'alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Backreaction')

    return

Spooler_Bacreaction()

end = time.time()
print('Elapsed Time:%s' % (end - start))

def rho_plot():
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        for i in index_b:
            b_i = b[i]

            for z in index_alpha:
                alpha_z = alpha[z]

                rho_B_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Rho_B')
                rho_E_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Rho_E')

                plt.semilogy(N, H_map ** 2, label=r'$\phi$')
                plt.semilogy(N, rho_B_Spool, label=r'$\rho_B$')
                plt.semilogy(N, rho_E_Spool * (a_map * H_map) ** 2, label=r'$\rho_E$')
                plt.legend()

    return plt.show()

rho_plot()

def Backreaction_plot():
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        for i in index_b:
            b_i = b[i]

            phi_solved = Buffer[0, j, i, 0]
            phi_prime_solved = Buffer[0, j, i, 1]

            dVdphi = phi_solved / phi_cmb

            phi_time = np.linspace(0., N_final, steps)
            plt.plot(phi_time, dVdphi, label=r'$V_{,\phi}$')

            for z in index_alpha:
                alpha_z = alpha[z]

                Backreaction_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Backreaction')

                plt.semilogy(N[:-1], Backreaction_Spool, label=r'$\frac{\alpha}{f}< E.B >$')

                plt.legend()
    plt.xlim(55., N_final)
    return plt.show()

Backreaction_plot()