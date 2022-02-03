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

steps = int(1.5e5)
Gauge_steps = int(1.45e5)

N_start = 55.

cpu = 24

N_start_intervals = int(100)
delta_intervals = N_start_intervals

alpha_i = 15.
alpha_f = 20.
alpha_list_size = 1

phi_cmb = 15

# PARAMETERS FOR LOOPS
####################################

f_list = np.array([0.0005, 0.00025, 0.0001])
#f_list = np.array([0.0001])
####################################

Data_set = '5'

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

    index_f = [1]

    for j in index_f:
        f_j = f_list[j]
        #index_b = [0]
        b = b_list(f_j)
        for i in index_b:
            b_i = b[i]
            phi_solved = Buffer[0, j, i, 0]
            phi_prime_solved = Buffer[0, j, i, 1]

            plt.subplot(211)
            plt.plot(N, phi_solved)
            plt.subplot(212)
            plt.plot(N, phi_prime_solved)

    plt.show()
    return

#Scalar_plot()



def N_k(delta):
    return delta + N_start

def k(delta):
    k = a(N_final) * H(N_final) * exp(N_k(delta) - N_final)
    return k

delta = np.linspace(2., 10., delta_intervals)

N = np.linspace(N_start, N_final, Gauge_steps)
N_index = np.arange(0, int(Gauge_steps), 1)

k_map = np.array([*map(k, delta)])
a_map = np.array([*map(a, N)])
H_map = np.array([*map(H, N)])


def rho_plot():

    index_f = [1]
    #index_b = []
    index_alpha = [0]

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

                #plt.semilogy(N, H_map ** 2, label=r'$\phi$')
                plt.semilogy(N, rho_B_Spool, label=r'$\rho_B$')
                #plt.semilogy(N, rho_E_Spool, label=r'$\rho_E$')
                #plt.semilogy(N, rho_B_Spool+rho_E_Spool* (a_map * H_map) ** 2)
                plt.legend()

    return plt.show()

#rho_plot()

def Backreaction_plot():

    index_f = [1]
    #index_b = [2]
    index_alpha = [0]

    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        for i in index_b:
            b_i = b[i]

            #Phi_Buffer = load('Phi', 'Phi')

            #phi_solved = Phi_Buffer[0, j, i, 0]
            #phi_prime_solved = Phi_Buffer[0, j, i, 1]

            #dVdphi = phi_solved / phi_cmb

            #phi_time = np.linspace(0., N_final, steps)
            #plt.semilogy(phi_time, dVdphi, label=r'$V_{,\phi}$')

            for z in index_alpha:
                alpha_z = alpha[z]

                Backreaction_Spool = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Backreaction')

                plt.semilogy(Backreaction_Spool[0], Backreaction_Spool[1], label=r'$\frac{\alpha}{f}< E.B >$')

                plt.legend()
    #plt.xlim(55., N_final-2.)
    return plt.show()

#Backreaction_plot()

def Asymptotic_plot():

    index_f = [2]
    #index_b = [1]
    index_alpha = [0]

    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        for i in index_b:
            b_i = b[i]

            for z in index_alpha:
                alpha_z = alpha[z]

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Asymptotes')

                plt.loglog(k_map / a(N_final), stack[1], label=r'$\rho_B$')
                #plt.loglog(k_map / a(N_final), stack[2], label=r'$\rho_E$')
                plt.legend()

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

Mag_Scatter()

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