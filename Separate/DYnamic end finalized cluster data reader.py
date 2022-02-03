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

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'
    , 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

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
b_f = 0.2
b_list_size = 5

steps = int(5.e5)
Gauge_steps = int(1.8e5)

N_start = 55.

steps = Gauge_steps*N_final/(N_final-N_start)

cpu = 24

N_start_intervals = int(100)
delta_intervals = N_start_intervals

alpha_i = 15.
alpha_f = 20.
alpha_list_size = 1

phi_cmb = 15

# PARAMETERS FOR LOOPS
####################################

# f_list = np.array([0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.0001])
f_list = np.array([0.0001, 0.00005, 0.0001])
####################################

index_b = np.arange(0, b_list_size, 1)
index_f = np.arange(0, len(f_list), 1)
index_alpha = np.arange(0, alpha_list_size, 1)


######################

#Setting epsilon limits for killing slow roll and non slow roll cases
eps_limit_ns = 1.5
eps_limit_slow = 1.
eps_overhead = 0.25
##################

# Parameters for computation start point and wavemodes
####################################

delta = np.linspace(.1, 10., delta_intervals)

####################################

#End time for gauge solver
N_gauge_off = 1.5

xi_samp_inter = 0.1     #For sampling xi around itme of interest to take 'non-oscillating' del phi for computing
                            #.....corresponding k

Data_set = '6'

Master_path = '/home/teerthal/Repository/Gauge_Evolution/test_cluster/test'


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

def eps_limite_ns(item_b, item_f):
    limit = (sqrt(2.) + (item_b*item_f))**2./2. #Hueristic
    if item_b == 0.:
        limit = 1.
    else:
        limit = 1. + eps_overhead  # Tuned

    return limit

def H(N):
    Epsilon = phi_prime(N) ** 2 / 2
    return abs(H_initial * sqrt((3. - Epsilon) / (3. - Epsilon) * (phi(N) / phi_initial) ** 2))

def k(delta, index_f, index_b):

    k = H(N_final) * exp(-delta)
    return k


def k_sr(delta):
    k = H(N_final) * exp(-delta)

    return k

def Asymptotes():
    stack_f = []
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])

        stack_j = []
        for i in index_b:
            b_i = b[i]

            k_map = np.array([*map(functools.partial(k, index_f=j, index_b=i),delta)])#;print(k_map)

            stack_z = []
            for z in index_alpha:
                alpha_z = alpha[z]
                alpha_sr_z = alpha_sr[z]

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Raw')
                asymptotes = np.array([k_map, stack[0, :, 1, -1], stack[0, :, 2, -1], stack[0, :, 3, -1]])

                save(asymptotes, 'alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Asymptotes')

                gauge_sr = load('slow_roll_alpha:%.4f'%alpha_sr_z, 'Raw')
                k_map_sr = np.array([*map(k_sr, delta)])#;print(k_map_sr)
                asymptotes_sr = np.array([k_map_sr, gauge_sr[0, :, 1, -1], gauge_sr[0, :, 2, -1], gauge_sr[0, :, 3, -1]])
                save(asymptotes_sr, 'slow_roll_alpha:%.4f'%alpha_sr_z, 'Asymptotes')

    return


#Asymptotes()
#nd = time.time();print(start-end)

def Asymptotic_plot():

    index_f = [0]

    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])

        for z in index_alpha:
            alpha_z = alpha[z]
            alpha_sr_z = alpha_sr[z]

            stack_sr = load('slow_roll_alpha:%.4f' % alpha_sr_z, 'Asymptotes')
            plt.loglog(stack_sr[0], stack_sr[1], label='Slow roll')

            for i in index_b:
                b_i = b[i]

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (alpha_z, b_i, f_j), 'Asymptotes')

                plt.loglog(stack[0], stack[1], label=r'bf:%.4f' % (b_i * f_j))
                # plt.loglog(stack[0], stack[2], label=r'$\frac{d\mathcal{A}_-}{dN}$')
                plt.legend()

    plt.xlabel(r'$k/a_{end}H_{end}$')
    plt.ylabel(r'$\sqrt{2k}\mathcal{A}_-(|k\eta_{end}|)$')
    return plt.show()


Asymptotic_plot()

