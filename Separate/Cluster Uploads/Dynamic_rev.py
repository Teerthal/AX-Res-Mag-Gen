import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
from multiprocessing import Pool
from scipy.integrate import trapz
import functools
from multiprocessing import cpu_count
from scipy.interpolate import interp1d

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

b_i = 0.01
b_f = 0.07
b_list_size = 2

steps = int(1.5e5)
Gauge_steps = int(1.7e5)

N_start = 55.

cpu = 4

N_start_intervals = int(3)
delta_intervals = N_start_intervals

alpha_i = 15.
alpha_f = 20.
alpha_list_size = 1

phi_cmb = 15

# PARAMETERS FOR LOOPS
####################################

# f_list = np.array([0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.0001])
f_list = np.array([0.0023])
####################################

Data_set = '6'

Master_path = '/home/teerthal/Repository/Gauge_Evolution/test'


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
    b = np.linspace(b_i, b_f, b_list_size) / item_f
    return b


def alpha_list(item_f):
    alpha = np.linspace(alpha_i, alpha_f, alpha_list_size) * item_f

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
    while solver.successful() and (sqrt(solver.y[1]**2/2))<=1.1:#solver.t <= N_stop(b):
        solver.integrate(solver.t + dt)
        temp_1.append(solver.t)
        temp_2.append(solver.y)

    if len(temp_1) > steps:
        temp_1.pop()
        temp_2.pop()

    N_phi_solved = np.array(temp_1)
    phi_solved = np.array([item[0] for item in temp_2])
    phi_prime_solved = np.array([item[1] for item in temp_2])

    return [phi_solved, phi_prime_solved, N_phi_solved]


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

slow_roll_stack = Scalar_Core(0., 0.01)

def slow_roll_interpolator(array, quantity_index):

    ref_size = len(array[0]);print(ref_size)
    slow_roll_size = len(slow_roll_stack[0])

    slow_roll_interp = interp1d(np.arange(slow_roll_size), slow_roll_stack[quantity_index])
    adjusted_slow_roll = slow_roll_interp(np.linspace(0,slow_roll_size-1, ref_size))

    return adjusted_slow_roll/array

index_b = np.arange(0, b_list_size, 1)
index_f = np.arange(0, len(f_list), 1)
index_alpha = np.arange(0, alpha_list_size, 1)


def Phi(index_N, index_b, index_f):
    phi_solved = Buffer[0, index_f, index_b, 0]
    #N = Buffer[0, index_f, index_b, 2]
    #index_N = int(abs(N / max(N) * steps - 1))

    return phi_solved[index_N]


def Phi_Prime(N, index_b, index_f):

    phi_prime_solved = Buffer[0, index_f, index_b, 1]
    N_ns = np.array(Buffer[0, index_f, index_b, 2])
    index_N = int(N / N_ns[-1] * len(N_ns))-1
    return phi_prime_solved[index_N]

def delta_phi():
    phi_slow = slow_roll_stack[0]
    phi_prime_slow = slow_roll_stack[1]
    N_slow = slow_roll_stack[2]
    eps_slow = abs(phi_prime_slow)

    stack_j = []
    for j in index_f:
        stack_i = []
        for i in index_b:

            N_ns = Buffer[0, index_f, index_b, 2]
            phi_ns = Buffer[0, index_f, index_b, 0]
            phi_prime_ns = Buffer[0, index_f, index_b, 1]
            eps_ns = abs(phi_prime_ns)

            adj_N = slow_roll_interpolator(N_ns,2)
            adj_phi = slow_roll_interpolator(phi_ns,0)
            adj_phi_prime = slow_roll_interpolator(phi_prime_ns, 1)
            adj_eps = abs(adj_phi_prime)

            del_phi = abs(phi_ns-adj_phi)
            del_phi_prime = abs(phi_prime_ns-adj_phi_prime)
            del_eps = del_phi_prime

            stack_i.append(np.array([del_phi, del_phi_prime]))
        stack_j.append(stack_i)
    return np.array(stack_j)

#delta_phi_Buffer = delta_phi()

def epsilon(N, index_b, index_f):
    return (Phi_Prime(N, index_b, index_f)) ** 2 / 2.

def H(N):
    Epsilon = phi_prime(N) ** 2 / 2
    return abs(H_initial * sqrt((3. - Epsilon) / (3. - Epsilon) * (phi(N) / phi_initial) ** 2))

def N_k(delta):
    return delta + N_start


def k(delta):
    phi_prime_solved = Buffer[0, 0, 0, 1]
    xi = 15.*phi_prime_solved
    N_k_delta = N_k(delta)
    index = int(N_k_delta/N_final*steps)
    xi_N_k = abs(xi[index])
    k = a(N_k_delta)*H(N_k_delta)*xi_N_k

    return k


def Scalar_plot():
    for j in index_f:
        for i in index_b:

            N_phi_solved = Buffer[0, j, i, 2]
            phi_solved = Buffer[0, j, i, 0]
            phi_prime_solved = Buffer[0, j, i, 1]

            # Growth of the physical wavenumber v growth of xi
            #a_map = np.array([*map(a, N)])
            #H_map = np.array([*map(H, N)])
            #plt.plot(N, 15.*abs(phi_prime_solved))
            #plt.plot(N, k(5.)/(a_map*H_map))
            #plt.show()

            plt.subplot(211)
            plt.semilogy(N_phi_solved, phi_solved)
            plt.semilogy(slow_roll_stack[2], slow_roll_stack[0])
            plt.ylabel(r'$\xi$')
            plt.subplot(212)
            plt.semilogy(N_phi_solved, phi_prime_solved ** 2 / 2)
            plt.semilogy(slow_roll_stack[2], slow_roll_stack[1])

    plt.show()
    return

Scalar_plot()


def Delta_plot():

    phi_slow = slow_roll_stack[0]
    phi_prime_slow = slow_roll_stack[1]
    N_slow = slow_roll_stack[2]
    eps_slow = abs(phi_prime_slow)

    for j in index_f:
        for i in index_b:

            N_ns = Buffer[0, j, i, 2]

            del_phi = delta_phi_Buffer[j,i,0,0]
            del_phi_prime = delta_phi_Buffer[j,i,1,0]
            del_eps = del_phi_prime
            print(np.shape(del_phi))

            plt.subplot(211)
            plt.semilogy(N_ns, del_phi)
            plt.ylabel(r'$\delta\phi/\phi$')
            plt.subplot(212)
            plt.semilogy(N_ns, del_eps)
            plt.ylabel(r'$\delta\epsilon/\epsilon$')

    plt.show()
    return

#Delta_plot()

N = np.linspace(N_start, N_final - 2., Gauge_steps)
N_index = np.arange(0, int(Gauge_steps), 1)

a_map = np.array([*map(a, N)])
H_map = np.array([*map(H, N)])

#plt.plot(N, np.array([*map(functools.partial(Phi_Prime, index_b=0, index_f=0), N)]));plt.show()


def Interpolator_Gauge_solver(N,index_b, index_f):
    x = Buffer[0, index_f, index_b, 2]
    y = Buffer[0, index_f, index_b, 0]
    z = Buffer[0, index_f, index_b, 1]

    y_inter = interp1d(x, y)
    z_inter = interp1d(z, x)

    y_new = y_inter(N)
    z_new = z_inter(N)

    return z_new


def ODE(N, u, arg):
    N_k = arg[0]
    h = arg[1]
    alpha = arg[2]
    index_b = arg[3]
    index_f = arg[4]
    Phi_Prime = arg[5](N, index_b, index_f)
    k_delta = arg[6]

    return np.array(
        [u[1],
         -u[1] - (k_delta ** 2 / (a(N)*H(N))**2
                  - k_delta / (a(N)*H(N))
                  * h * alpha / f_list[index_f] * Phi_Prime) * u[0]])


def core(delta, h, alpha, index_b, index_f):
    Parameters = np.array([N_k(delta), h, alpha, index_b, index_f, Phi_Prime, k(delta)])

    r = ode(ODE).set_integrator('zvode').set_f_params(Parameters)

    # Initial Condition
    t_initial = -1
    A_initial = exp(-1j * k(delta) * t_initial)
    A_prime_initial = -1j * A_initial * k(delta) / (a(N_start)*H(N_start))

    init = np.array([A_initial, A_prime_initial]);

    r.set_initial_value(init, N_start)

    u = []
    t = []

    dt = (N_final - 2. - N_start) / Gauge_steps
    while r.successful() and r.t <= N_final - 2.:
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
    a_map_Gauge = np.array([map(a, N)])
    H_map_Gauge = np.array([map(H, N)])

    norm_A = []
    norm_A[:index] = A[:index]**2-1
    norm_A[index:] = A[index:]**2

    norm_del_A = []
    norm_del_A[:index] = (A_prime*a_map*H_map/k(delta))[:index]**2-1
    norm_del_A[index:] = (A_prime*a_map*H_map/k(delta))[index:]**2

    norm_A = np.array(norm_A)
    norm_del_A = np.array(norm_del_A)

    norm_A[norm_A < 0.1] = 0.
    norm_del_A[norm_del_A < 0.1] = 0.

    return np.array([N, A, A_prime, norm_A, norm_del_A])


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
                        stack = np.array(pool.map(functools.partial(core, h=h, alpha=item_alpha,
                                                                    index_b=i, index_f=j), delta))
                        pool.close()

                    stack_h.append(stack)
                    #print(np.shape(stack))
                save(stack_h, 'alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Raw')

    return


# Parameters for computation start point and wavemodes
####################################

delta = np.linspace(5., 12., delta_intervals)

####################################
execute(delta)

end = time.time()
print(end - start)

k_map = np.array([*map(k, delta)])

def Gauge_curves():
    for i in np.arange(0, delta_intervals, 1):
        del_i = delta[i]
        k_i = k_map[i]

        for l in index_f:
            item_f = f_list[l]
            b = b_list(item_f)
            alpha = alpha_list(item_f)

            for j in index_b:
                item_b = b[j]

                for z in index_alpha:
                    item_alpha = alpha[z]

                    stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Raw')
                    plt.subplot(211)
                    plt.ylabel(r'$|\mathcal{A}|$')
                    plt.semilogy(N, sqrt(stack[0,i, 3]), label='k:%s ,bf:%s' % (k_i, item_b*item_f))
                    plt.semilogy(N, stack[0, i, 1], label='k:%s ,bf:%s' % (k_i, item_b * item_f), linestyle=':')
                    plt.legend()
                    plt.subplot(212)
                    plt.ylabel(r'$|\frac{d\mathcal{A}}{dx}|$')

                    A_prime_norm = sqrt(stack[0,i, 4])
                    A_prime = stack[0,i, 2]*a_map*H_map/k(del_i)
                    plt.semilogy(N, A_prime_norm, label='k:%s' % k_i)
                    plt.semilogy(N, A_prime, label='k:%s' % k_i, linestyle=':')
                    plt.legend()
    plt.show()
    return


#Gauge_curves()


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

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Raw')
                print(np.shape(stack))
                asymptotes = np.array([k_map, stack[0, :, 1, -1], stack[0, :, 2, -1]])
                print('asymp check', asymptotes)
                save(asymptotes, 'alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Asymptotes')

    return


Asymptotes()
end = time.time();print(start-end)

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

                plt.loglog(stack[0], stack[1], label=r'$\mathcal{A}_-$')
                #plt.loglog(stack[0], stack[2], label=r'$\frac{d\mathcal{A}_-}{dN}$')
                plt.legend()

    return plt.show()


Asymptotic_plot()


def A(N_index, b, f, alpha):
    stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha, b, f), 'Raw')

    return stack[0, :, 1, N_index]


def A_prime(N_index, b, f, alpha):
    stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha, b, f), 'Raw')

    return stack[0, :, 2, N_index]


def rho_B(N_index, b, f, alpha):
    Integrand = 1 / (4 * pi ** 2) * (A(N_index, b, f, alpha) ** 2 - k_map ** 2) * (k_map) ** 4 / k_map
    return trapz(Integrand, x=k_map)


def rho_E(N_index, b, f, alpha):
    Integrand = 1 / (4 * pi ** 2) * ((A_prime(N_index, b, f, alpha) / k_map) ** 2 - k_map ** 2) * (k_map) ** 4 / k_map
    return trapz(Integrand, x=k_map)


def Backreaction_Functional(N_index, b, f, alpha):
    Integrand = k_map ** 2 / (16 * pi ** 2) * (A(N_index, b, f, alpha) ** 2)
    return trapz(Integrand, x=k_map)


end = time.time()
print('Elapsed Time:%s' % (end - start))


def Shredder_B(stack, N_index):
    A_z = stack[0, :, 3, N_index]
    Integrand = 1 / (4 * pi ** 2) * A_z * k_map ** 4
    integrated = trapz(Integrand, x=log(k_map))

    return integrated


def Shredder_E(stack, N_index):
    A_prime_z = stack[0, :, 4, N_index]
    Integrand = 1 / (4 * pi ** 2) * A_prime_z * (k_map) ** 4 / k_map

    return trapz(Integrand, x=k_map)


def Shredder_Back(stack, N_index):
    A_z = stack[0, :, 3, N_index]
    Integrand = k_map ** 2 / (8 * pi ** 2) * A_z

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

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Raw')

                partial_Shred = functools.partial(Shredder_B, stack)

                pool = Pool(cpu)

                if __name__ == '__main__':
                    rho_B_map = np.array(pool.map(partial_Shred, N_index))
                pool.close()

                # pool = Pool(cpu)

                # if __name__ == '__main__':
                # rho_B_map = np.array(
                # pool.map(functools.partial(rho_B, b=item_b,
                # f=item_f, alpha=item_alpha), N_index))
                # pool.close()
                print(np.shape(rho_B_map))
                save(rho_B_map/(a_map*H_map)**4, 'alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Rho_B')

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

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Raw')

                partial_Shred = functools.partial(Shredder_E, stack)

                pool = Pool(cpu)

                if __name__ == '__main__':
                    rho_E_map = np.array(pool.map(partial_Shred, N_index))
                pool.close()

                # pool = Pool(cpu)

                # if __name__ == '__main__':
                # rho_E_map = np.array(
                # pool.map(functools.partial(rho_E, b=item_b,
                # f=item_f, alpha=item_alpha), N_index))
                # pool.close()
                save(rho_E_map/(a_map*H_map)**4, 'alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Rho_E')

    return


Spooler_rho_E()

end = time.time()
print('Elapsed Time:%s' % (end - start))


def Down_res(array, spacing):
    orig_size = len(array)
    orig_index = np.arange(0, orig_size, 1)
    downgraded_index = np.arange(0, orig_size, spacing)
    downgraded_array = array[downgraded_index]
    print(np.shape(downgraded_array))
    return downgraded_array


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

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Raw')

                partial_Shred = functools.partial(Shredder_Back, stack)

                pool = Pool(cpu)

                if __name__ == '__main__':
                    Backreaction_map = np.array(pool.map(partial_Shred, N_index))
                pool.close()


                #dN = float((N_final - 2. - N_start) / Gauge_steps)
                #y = Down_res(Backreaction_map, 10)

                #N_Backreaction = np.linspace(N_start, N_final - 2., len(y))
                #H_map_Backreaction = np.array([*map(H, N_Backreaction)])
                #a_map_Backreaction = np.array([*map(a, N_Backreaction)])

                #dy = np.array(np.diff(y))
                #dx = np.diff(N_Backreaction)

                #dy_filtered = lowess(abs(dy / dx), N_Backreaction[:-1], frac=0.01)

                #Backreaction_filtered = item_alpha / item_f * dy_filtered[:, 1] * H_map_Backreaction[:-1] / a_map_Backreaction[:-1]**3

                #Backreaction_stack = np.array([dy_filtered[:, 0], Backreaction_filtered])

                save(Backreaction_map, 'alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Backreaction')

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
                plt.semilogy(N, rho_E_Spool , label=r'$\rho_E$')
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
            plt.semilogy(phi_time, dVdphi, label=r'$V_{,\phi}$')

            for z in index_alpha:
                alpha_z = alpha[z]

                Backreaction_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Backreaction')

                plt.semilogy(N, Backreaction_Spool, label=r'$\frac{\alpha}{f}< E.B >$')

                plt.legend()
    plt.xlim(55., N_final - 2.)
    return plt.show()

Backreaction_plot()

def Parameters_Log():
    print('Frequency Parameters')
    print(f_list)
    print('------------------------------------------------')
    print('Parameters')
    print('------------------------------------------------')
    print('N initial :', N_initial)
    print('N final :', N_final)
    print('Offest in Phi solver:', Offset)
    print('Steps for Phi Solver:', steps)
    print('------------------------------------------------')
    print('Steps for Gauge Solver:', Gauge_steps)
    print('Gauge Solver')
    print('N initial:', N_start)
    print('Stopping time:', N_final - 2.)
    print('Number of Vector modes:', N_start_intervals)
    print('------------------------------------------------')
    print('alpha/f')
    print('alpha_i:', alpha_i, 'alpha_f:', alpha_f)
    print('size of alpha intervals:', alpha_list_size)
    print('bf')
    print('b_i', b_i, 'b_f', b_f)
    print('Size of b intervals', b_list_size)

    return


Parameters_Log()
