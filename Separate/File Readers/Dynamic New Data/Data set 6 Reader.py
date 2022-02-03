import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
from multiprocessing import Pool
from scipy.integrate import trapz
import functools
from multiprocessing import cpu_count
#from numdifftools import Derivative
flatten = np.ndarray.flatten
#from scipy.interpolate import UnivariateSpline
#from scipy import interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess
from mpl_toolkits.mplot3d import Axes3D

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

Offset = 2;

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

b_i = 0.005
b_f = 0.1
b_list_size = 10

steps = int(2.5e5)
Gauge_steps = int(2.45e5)

N_start = 55.

cpu = 24

N_start_intervals = int(200)
delta_intervals = N_start_intervals

alpha_i = 15.
alpha_f = 20.
alpha_list_size = 1

phi_cmb = 15

# PARAMETERS FOR LOOPS
####################################

f_list = np.array([0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.0001])
#f_list = np.array([0.0001])
####################################

Data_set = '6'

Master_path = '/home/teerthal/Repository/Gauge_Evolution/Dynamic/New_Data/%s' %Data_set

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
    b = np.geomspace(b_i, b_f, b_list_size) / item_f
    return b


def alpha_list(item_f):
    alpha = np.linspace(alpha_i, alpha_f, alpha_list_size) * item_f

    return alpha


def N_stop(b):
    return N_final + b * f(N_final) * Offset

index_b = np.arange(0, b_list_size, 1)
index_f = np.arange(0, len(f_list), 1)
index_alpha = np.arange(0, alpha_list_size, 1)


def Phi(N, index_b, index_f):
    Buffer = load('Phi', 'Phi')
    phi_solved = Buffer[0, index_f, index_b, 0]
    index_N = int(abs(N / N_final * steps - 1))

    return phi_solved[index_N]


def Phi_Prime(N, index_b, index_f):
    Buffer = load('Phi', 'Phi')
    phi_prime_solved = Buffer[0, index_f, index_b, 1]
    index_N = int(abs(N / N_final * steps - 1))

    return phi_prime_solved[index_N]

def epsilon(N, index_b, index_f):
    return (Phi_Prime(N, index_b, index_f)) ** 2 / 2.

def H(N):
    Epsilon = phi_prime(N) ** 2 / 2
    return abs(H_initial * sqrt((3. - Epsilon) / (3. - Epsilon) * (phi(N) / phi_initial) ** 2))


def Scalar_plot():
    N = np.linspace(N_initial, N_final, steps)

    Buffer = load('Phi', 'Phi')

    index_f = [3]

    for j in index_f:
        f_j = f_list[j]
        #index_b = [0]
        b = b_list(f_j)

        for i in index_b:
            b_i = b[i]
            phi_solved = Buffer[0, j, i, 0]
            phi_prime_solved = Buffer[0, j, i, 1]
            plt.subplot(211)
            plt.plot(N, phi_solved, label='%.4f'%(b_i*f_j))
            plt.ylabel(r'$\phi$')
            plt.legend()
            plt.subplot(212)
            plt.plot(N, phi_prime_solved**2/2)
            plt.xlim(65,70)
            plt.ylabel(r'$\epsilon$')
            plt.plot(N,[1]*steps, linestyle=':', color='k')
            plt.xlabel('N')

    plt.show()
    return

#Scalar_plot()


def N_k(delta):
    return delta + N_start

def k(delta):
    k = a(N_final) * H(N_final) * exp(N_k(delta) - N_start)
    return k

delta = np.linspace(2., 10., delta_intervals)

N = np.linspace(N_start, N_final-2., Gauge_steps)
N_index = np.arange(0, int(Gauge_steps), 1)

k_map = np.array([*map(k, delta)])
a_map = np.array([*map(a, N)])
H_map = np.array([*map(H, N)])

def rho_plot():

    index_f = [3]
    #index_b = []
    index_alpha = [0]

    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        plt.plot([],[], color='w', label='alpha/f:%.1f'%(alpha[0]/f_j))
        plt.plot([],[], color='w', label='bf')

        for i in index_b:
            b_i = b[i]

            for z in index_alpha:
                alpha_z = alpha[z]

                rho_B_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Rho_B')[:,0]
                rho_E_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Rho_E')[:,0]

                plt.semilogy(N, rho_B_Spool, label=r'$\rho_B$')
                #plt.semilogy(N, rho_E_Spool* (a_map * H_map) ** 2, label=r'$\rho_E$')
                #plt.semilogy(N, rho_B_Spool+rho_E_Spool* (a_map * H_map) ** 2, label='%.4f'%(b_i*f_j))
                plt.legend()
                plt.xlabel('N')
                plt.ylabel(r'$\rho_{em}$')
    return plt.show()

#rho_plot()

def xi_plot():

    N = np.linspace(N_initial, N_final, steps)
    Buffer = load('Phi', 'Phi')

    index_f = [3]
    index_b = [0]
    index_alpha = [0]

    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)


        for i in index_b:
            b_i = b[i]

            phi_prime_solved = Buffer[0, j, i, 1]

            for z in index_alpha:
                alpha_z = alpha[z]

                xi = alpha_z/f_j*phi_prime_solved
                #plt.semilogy(delta + N_start, (k_map/(a(N_final)*H(N_final)))**2)
                #xi_analytical = a(N)*H(N)*exp(65-N)
                #plt.semilogy(N,abs(xi))
                #delta_full = np.linspace(0, N_final-N_start, 1000)
                #plt.semilogy(N_start+delta_full, np.array([*map(k, delta_full)]))
                #plt.plot(N, xi_analytical)

                delta_sample = 10
                func_1 = exp(N_k(delta_sample)-N);print(exp(N_k(delta_sample)-65))
                func_2 = abs(func_1*xi)
                plt.semilogy(N,func_1, label='1')
                plt.semilogy(N, func_2/func_1)
                #plt.xlim(N_start, N_final)
                plt.legend()
    return plt.show()

xi_plot()



def Backreaction_plot():

    index_f = [1]
    #index_b = [2]
    index_alpha = [0]

    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        plt.plot([],[], color='w', label='alpha/f:%.1f'%(alpha[0]/f_j))
        plt.plot([],[],color='w', label='bf')

        Phi_Buffer = load('Phi', 'Phi')

        phi_solved = Phi_Buffer[0, j, b_list_size-1, 0]
        phi_prime_solved = Phi_Buffer[0, j, b_list_size-1, 1]

        dVdphi = phi_solved / phi_cmb

        phi_time = np.linspace(0., N_final, steps)
        plt.semilogy(phi_time, dVdphi, label=r'$V_{,\phi}$', color='k', linestyle=':')

        for i in index_b:
            b_i = b[i]

            for z in index_alpha:
                alpha_z = alpha[z]

                Backreaction_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Backreaction')

                plt.semilogy(Backreaction_Spool[0], Backreaction_Spool[1], label='%.4f'%(b_i*f_j))

                plt.legend()
    plt.xlim(N_start, N_final-2.)
    plt.ylabel(r'$\frac{\alpha}{f}< E.B >$')
    plt.xlabel(r'$N$')
    return plt.show()

#Backreaction_plot()

def Asymptotic_plot():

    index_f = [5]
    #index_b = [1]
    index_alpha = [0]

    for j in index_f:
        f_j = f_list[j]

        plt.plot([], [], color='w', label='f:%.4f' %f_j)
        plt.plot([], [], color='w', label='bf')

        b = b_list(f_j)
        alpha = alpha_list(f_j)

        for i in index_b:
            b_i = b[i]

            for z in index_alpha:
                alpha_z = alpha[z]

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Asymptotes')

                plt.semilogy(N_start+delta, stack[1], label=b_i*f_j)
                #plt.loglog(k_map / a(N_final), stack[2], label=b_i*f_j)
                #plt.semilogy(N, k_map**4*stack[1]**2)
                print((k(10))**3*max(stack[1]**2))
                plt.xlabel(r'$\k/a_f$')
                plt.ylabel(r'$\sqrt{2k}\mathcal{A}_-(-k\eta\ll1)$')
    return plt.show()

Asymptotic_plot()

def Magnetic_Field(alpha, f, b):
    rho_B_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha, b, f), 'Rho_B')
    B_max = sqrt(max(rho_B_Spool))

    Asymp_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha, b, f), 'Asymptotes')
    Asymp_max = max(Asymp_Spool[1])
    k_max_indice = np.argmax(Asymp_max)
    k_max = k_map[int(k_max_indice)]/a(N_final);print(k_max)
    M_PL = 2.4e18
    z_rec = 1100
    T_r = 1e14
    H_f = H(N_final)
    Kelvin = 8.6e-14
    T_0 = 3 * Kelvin
    Gauss = 6.8e-20
    Mpc = 1e-39*6.4
    B_max_final = (1+z_rec)**(-2/3)*T_0**(7/3)*B_max**(2/3) * (T_r/M_PL)**(1/3)*(H_f)**(1/3)*k_max**(-1/3)*M_PL**2*Gauss
    Lamda_max_final = B_max**(2/3)*(T_0)**(-5/3)*(1+z_rec)**(-2/3)*(T_r/M_PL)**(1/3)*(H_f)**(1/3)*Mpc/M_PL*k_max**(-2/3)
    print(B_max_final, Lamda_max_final)
    return np.array([B_max_final, Lamda_max_final])

def Max_Mag_asymptotic(alpha, f, b):

    Asymp_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha, b, f), 'Asymptotes')
    Asymp_max = max(Asymp_Spool[1])
    k_max_indice = np.argmax(Asymp_max);print(k_max_indice)
    k_max = k_map[int(k_max_indice)] / a(N_final)
    M_PL = 2.4e18
    z_rec = 1100
    T_r = 1e14
    H_f = H(N_final)
    Kelvin = 8.6e-14
    T_0 = 3 * Kelvin
    Gauss = 6.8e-20
    Mpc = 1e-39 * 6.4

    B_max = abs(Asymp_max)*k_max**2

    B_max_final = (1 + z_rec) ** (-2 / 3) * T_0 ** (7 / 3) * B_max ** (2 / 3) * (T_r / M_PL) ** (1 / 3) * (H_f) ** (
    1 / 3) * M_PL ** 2 * Gauss
    Lamda_max_final = B_max**(2/3)*T_0**(-5/3)*(1+z_rec)**(-2/3)*(H_f)**(1/3)*Mpc/M_PL
    print(B_max_final, Lamda_max_final)
    return np.array([B_max_final, Lamda_max_final])


def Mag_Collect():

    stack = []
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        for i in index_b:
            b_i = b[i]

            for z in index_alpha:
                alpha_z = alpha[z]

                B = Magnetic_Field(alpha_z, f_j, b_i)

                stack.append(abs(np.array( [b_i*f_j, alpha_z/f_j, B[0], B[1]])))
    print(np.shape(stack))
    return np.array(stack)

def Mag_Scatter():
    stack = Mag_Collect()

    plt.subplot(211)
    plt.scatter(stack[:,0], stack[:,2])
    plt.yscale('Log')
    plt.ylim(1e-20,1e-13)
    plt.ylabel(r'$B_{max}$')
    plt.xlabel(r'$bf$')
    plt.subplot(212)
    plt.scatter(stack[:,1], stack[:,2])
    plt.yscale('Log')
    plt.ylim(1e-20, 1e-13)
    plt.xlabel(r'$\alpha/f$')
    plt.show()
    return

#Mag_Scatter()

def Mag_3D_Scatter():

    fig = plt.figure()
    ax = Axes3D(fig)

    stack = Mag_Collect()

    ax.scatter(stack[:,0], stack[:,1], stack[:,2])

    ax.set_zscale('log')

    ax.set_zlim3d(1e-19,1e-14)
    plt.show()

    return

#Mag_3D_Scatter()