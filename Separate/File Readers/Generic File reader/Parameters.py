###############################
#Parameters for data computed fbasedd Dyn_eff, Dyn end formalized code(eff)
###############################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
from multiprocessing import Pool
from scipy.integrate import trapz
import functools
from multiprocessing import cpu_count
from scipy.interpolate import interp1d
import os
from shutil import copyfile
import matplotlib.cm as cm
from decimal import Decimal
from matplotlib import ticker

# from numdifftools import Derivative
flatten = np.ndarray.flatten
# from scipy.interpolate import UnivariateSpline
# from scipy import interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'
    , 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

start = time.time()

def timer(note):
    t = time.time()
    print(note, ':', t-start)
    return


plt.rcParams['axes.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15


pi = np.pi;
sin = np.sin;
cos = np.cos;
exp = np.exp;
log = np.log;
abs = np.abs;
absolute = np.absolute;
sqrt = np.sqrt

code = 0

if code == 0:
    p = 2

    F = 0.0025
    alpha_0 = 10.

    N_final = 70.
    N_initial = 50.

    Offset = 5;

    phi_initial = sqrt(282)
    phi_final = p / sqrt(2)

    H_initial = 1.

    b_i = 0.1
    b_f = 0.2
    b_list_size = 3

    steps = int(1.85e5)
    Gauge_steps = int(1.9e5)

    # Gauge_steps = int(2e4)#Test

    N_start = 55.

    steps = Gauge_steps * N_final / (N_final - N_start)
    N_index = np.arange(0, int(Gauge_steps), 1)

    cpu = 24

    N_start_intervals = int(cpu * 8)
    # N_start_intervals = 2 #Test
    delta_intervals = N_start_intervals

    alpha_i = 1.
    alpha_f = 15.
    alpha_list_size = 20
    # alpha_list_size = 2 #Test

    phi_cmb = 15

    # PARAMETERS FOR LOOPS
    ####################################

    # f_list = np.array([0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.0001])
    # f_list = np.array([5e-2, 1e-2, 5e-3, 1e-3])
    f_list = np.array([5e-2, 1e-3])  # Test
    ####################################

    index_b = np.arange(0, b_list_size, 1)
    index_f = np.arange(0, len(f_list), 1)
    index_alpha = np.arange(0, alpha_list_size, 1)

    ######################

    # Setting epsilon limits for killing slow roll and non slow roll cases
    eps_limit_ns = 1.5
    eps_limit_slow = 1.
    eps_overhead = 1.
    sparce_res = 1000  # The number with which the scalar solution is split into for computing the mean
    # \epsilon
    ##################

    # Parameters for computation start point and wavemodes
    ####################################

    delta = np.linspace(5., 12., delta_intervals)

    ####################################

    # End time for gauge solver
    N_gauge_off = 0.

    xi_samp_inter = 0.1  # For sampling xi around itme of interest to take 'non-oscillating' del phi for computing
    # .....corresponding k
    Data_set = '%.2f_%.2f_%.7f(final f)_%.0f_%.0f_%.0f' % (
    alpha_i, alpha_f, f_list[-1], Gauge_steps, len(f_list), delta_intervals)

    Master_path = '/media/teerthal/Repo/Monodromy/Cluster/%s'%Data_set



if code == 1:
    p = 2

    F = 0.0025
    alpha_0 = 10.

    N_final = 70.
    N_initial = 50.

    Offset = 5;

    phi_initial = sqrt(282)
    phi_final = p / sqrt(2)

    H_initial = 0.5

    b_i = 0.1
    b_f = 0.2
    b_list_size = 3

    steps = int(5.e5)
    Gauge_steps = int(2e5)
    # Gauge_steps = int(2e4)#Test

    N_start = 60.

    steps = Gauge_steps * N_final / (N_final - N_start)
    N_index = np.arange(0, int(Gauge_steps), 1)

    cpu = 24

    N_start_intervals = int(200)
    # N_start_intervals = 2 #Test
    delta_intervals = N_start_intervals

    alpha_i = 1.
    alpha_f = 15.
    alpha_list_size = 100
    # alpha_list_size = 2 #Test

    phi_cmb = 15

    # PARAMETERS FOR LOOPS
    ####################################

    # f_list = np.array([0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.0001])
    f_list = np.array([1e-4])
    # f_list = np.array([1e-3])   #Test
    ####################################

    index_b = np.arange(0, b_list_size, 1)
    index_f = np.arange(0, len(f_list), 1)
    index_alpha = np.arange(0, alpha_list_size, 1)

    ######################

    # Setting epsilon limits for killing slow roll and non slow roll cases
    eps_limit_ns = 1.5
    eps_limit_slow = 1.
    eps_overhead = 1.
    sparce_res = 1000  # The number with which the scalar solution is split into for computing the mean
    # \epsilon
    ##################

    # Parameters for computation start point and wavemodes
    ####################################

    delta = np.linspace(1., 10., delta_intervals)

    ####################################

    # End time for gauge solver
    N_gauge_off = 0.

    xi_samp_inter = 0.1  # For sampling xi around itme of interest to take 'non-oscillating' del phi for computing
    # .....corresponding k
    Data_set = '%.2f_%.2f_%.7f(final f)_%.0f_%.0f_%.0f' % (
        alpha_i, alpha_f, f_list[-1], Gauge_steps, len(f_list), delta_intervals)

    Master_path = '/media/teerthal/Repo/Monodromy/Cluster/%s' % Data_set



