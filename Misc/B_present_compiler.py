import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
from multiprocessing import Pool
from scipy.integrate import trapz
import functools
from multiprocessing import cpu_count
from scipy.interpolate import interp1d
from numba import jit, prange
from decimal import Decimal
import math as mt
import scipy.interpolate
import _pickle as cPickle
import matplotlib.animation as Animation
import pylab as pylab
import scipy.fftpack as fft
import os as os
from shutil import copyfile
import matplotlib.cm as cm
from matplotlib import gridspec
import scipy.integrate as integrate

# from numdifftools import Derivative
flatten = np.ndarray.flatten
# from scipy.interpolate import UnivariateSpline
# from scipy import interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'
    , 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


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
acos = mt.acos
asin = mt.sin
sinh = np.sinh
cosh = np.cosh


def timer(note, start):
    t = time.time()
    print(note, ':', t-start)
    return

def make_dir(Master_path):
    if os.path.exists(Master_path):
        print('Data Directory exists')
    else:
        os.mkdir(Master_path)
        print('Data directory does not exist')


def save(data, name, directory, Master_path):
    path = '%s/%s/%s.npy' % (Master_path, directory, name)

    file = open(path, 'wb')
    np.save(file, data)
    file.close()
    return


def load(name, directory, Master_path):
    path = '%s/%s/%s.npy' % (Master_path, directory, name)
    file = open(path, 'rb')
    stack = np.load(file)
    file.close()
    return stack


def load_or_exec(filename, directory, function, Master_path, execute_arguments):
    command = input('Rewrite %s in %s (T/F):'%(filename, directory))

    sub_path = '%s/%s' % (Master_path, directory)
    file_path = '%s/%s/%s.npy' % (Master_path, directory, filename)

    if os.path.exists(sub_path):
        print('Sub Directory exists')
    else:
        os.mkdir(sub_path)
        print('Sub directory does not exist')
        print('Builing new Subdirectory %s'%directory)

    if os.path.exists(file_path) and command == 'F':

        print('File exists')
        file = load(filename, directory, Master_path)

    else:

        print('File does not exist or computing fresh set')
        file = function(*execute_arguments)
        save(file, filename, directory, Master_path)


    return np.array(file)


def V_PN(phi,p):

    V = (1-1/(1+((phi)**2))**p)
    V_norm = 0.5*(1 - 1 / (1 + ((phi) ** 2) / p) ** p)

    V_mon = phi**p + 0.1*cos(phi)

    return V_norm


def load_or_exec_multi_file(Master_path, directory, function, execute_arguments):

    path_data = '%s/%s' % (Master_path, directory)

    Rewrite_command = input('Overwrite existing data in %s?(T or F):'%directory)

    if os.path.exists(path_data) == True and Rewrite_command == 'F':

        print('Using Existing Data')

    else:
        if os.path.exists(path_data):
            print('Sub Directory exists')
            print('Overwriting data in %s'%(directory))
            function(*execute_arguments)
        else:
            os.mkdir(path_data)
            print('Sub directory does not exist')
            print('Builing new Subdirectory %s' % directory)
            print('Computing new data')
            function(*execute_arguments)

    return

H_inf = 1e-8
T_R = (90 / (100 * pi ** 2)) ** (1 / 2) * (H_inf) ** (1 / 2)
T_0 = 23.5 * 1e-14
Mpl = 2.435 * 1e18  ##Mpl in GeV
GeV2 = 6.8 * 1e20  ##Gev^2 in Gauss
a_R = T_0 / (T_R * Mpl)


def mapper(var, var_arr , target_arr):

    diff_arr = abs(var_arr-var)
    min_idx = np.argmin(diff_arr)

    out = target_arr[min_idx]

    return out

def Lambda_0(Lambda_k_f, B_k_f):
    Lambda_0 = 3.3e5 * a_R * (Lambda_k_f) ** (1 / 3) * (B_k_f) ** (2 / 3);
    return Lambda_0

def B_0(Lambda_0):
    B_0 = 1e-8 * Lambda_0
    return B_0

def rho_plot(p, alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix, Phi_Stack, asymp_stack):
    h = [-1, 1]

    targ_f = [0.01, 0.005]
    f_idxs = [np.argmin(abs(targ - f)) for targ in targ_f]

    p_idxs = [0]

    targ_alpha = [0.5, 0.6, 0.7, 0.8]
    alpha_idxs = [np.argmin(abs(targ - alpha)) for targ in targ_alpha]

    for cnt_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        for cnt_p, idx_p, in enumerate(p_idxs):
            item_p = p[idx_p]
            sol = Phi_Stack[idx_f][idx_p]
            t = sol[0];
            phi_a = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in t]) \
                    / mapper(t[0],a_stack[idx_f][idx_p][0],a_stack[idx_f][idx_p][1])
            phi = sol[1]
            dphi = sol[2]

            for cnt_alpha, idx_alpha in enumerate(alpha_idxs):
                item_alpha = alpha[idx_alpha]

                name_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, -1, item_p, item_f)
                name_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, 1, item_p, item_f)
                name_0_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., -1, item_p, item_f)
                name_0_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., 1, item_p, item_f)


                stack_B_1, stack_B_2 = np.array(load(name_1, 'Rho_B', Master_path)), np.array(
                    load(name_2, 'Rho_B', Master_path))
                stack_B_0_1, stack_B_0_2 = np.array(load(name_0_1, 'Rho_B', Master_path)), np.array(
                    load(name_0_2, 'Rho_B', Master_path))

                x_B = stack_B_1[:, 0]

                x_B_0_1, rho_B_0_1 = stack_B_0_1[:, 0], stack_B_0_1[:, 1]
                x_B_0_2, rho_B_0_2 = stack_B_0_2[:, 0], stack_B_0_2[:, 1]
                rho_B = stack_B_1[:, 1] / rho_B_0_1 + stack_B_2[:, 1] / rho_B_0_2  # *m**2/H_map**2/(a_i**4)

                stack_E_1 = np.array(load(name_1, 'Rho_E', Master_path));
                stack_E_0_1 = np.array(load(name_0_1, 'Rho_E', Master_path))
                stack_E_2 = np.array(load(name_2, 'Rho_E', Master_path));
                stack_E_0_2 = np.array(load(name_0_2, 'Rho_E', Master_path))
                x_E_0_1, rho_E_0_1 = stack_E_0_1[:, 0], stack_E_0_1[:, 1]
                x_E_0_2, rho_E_0_2 = stack_E_0_2[:, 0], stack_E_0_2[:, 1]
                rho_E = stack_E_1[:, 1] / rho_E_0_1 + stack_E_2[:, 1] / rho_E_0_2  # *m**2/H_map**2/(a_i**2)


                m = H_inf / item_f * (3 * (0.5 * absolute(dphi[0]) ** 2 + V_PN(phi[0], item_p)) ** (-1)) ** (1 / 2)

                H_map = sqrt(item_f ** 2 * (0.5 * dphi ** 2 + np.array([V_PN(z, item_p) for z in phi]))) / sqrt(
                    (0.5 * dphi[0] ** 2 + V_PN(phi[0], item_p)));
                H_map = np.array([mapper(x, t, H_map) for x in x_B])
                t_i = trigger_matrix[idx_f][idx_p]
                a_i = mapper(t_i, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                a_map = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in x_B]) / a_i;
                a_map = 1. / a_i  # print(np.shape(rho_B),np.shape(rho_E),np.shape(H_map))#a_map = 1/a_i
                cross_idx = np.argwhere(
                    np.diff(np.sign(np.absolute(rho_E + rho_B) - 3 * H_map ** 2 / m ** 2))).flatten()

                t_i = trigger_matrix[idx_f][idx_p]
                t_f = kill_matrix[idx_f][idx_p];
                a_f = mapper(t_f, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])

                A_0 = [absolute(asymp_stack[idx_f, idx_h, idx_p, 0, :, 1]) for idx_h in [0, 1]]
                A = [absolute(asymp_stack[idx_f, idx_h, idx_p, idx_alpha, :, 1]) - absolute(A_0[idx_h]) for idx_h in
                     [0, 1]]

                max_idx = np.argmax(A[0])
                max_k, max_A = [k[idx_f][idx_p][max_idx],
                                sqrt(absolute(A[0][max_idx]) ** 2 + absolute(A[1][max_idx]) ** 2)]

                max_A0 = sqrt(absolute(A_0[0][max_idx]) ** 2 + absolute(A_0[1][max_idx]) ** 2)

                rho_B_1 = (absolute(stack_B_1[:, 1]) - absolute(rho_B_0_1)) * m ** 4 * Mpl ** 4;
                rho_E_1 = (absolute(stack_E_1[:, 1]) - absolute(rho_E_0_1)) * m ** 4 * Mpl ** 4  # *m**2/H_map**2
                rho_B_2 = (absolute(stack_B_2[:, 1]) - absolute(rho_B_0_2)) * m ** 4 * Mpl ** 4;
                rho_E_2 = (absolute(stack_E_2[:, 1]) - absolute(rho_E_0_2)) * m ** 4 * Mpl ** 4  # *m**2/H_map**2
                ratio = \
                (((rho_B_1 + rho_B_2) / a_map ** 4 + (rho_E_1 + rho_E_2) / a_map ** 2) * m ** 2 / (3 * H_map ** 2))[-1]
                print(r'$\alpha:$', item_alpha, 'f:', item_f, 'Ratio:%.5f' % ratio)

                Lambda_f = 2 / (m * Mpl);
                Lambda_f = (a_f / (max_k * m * Mpl))

                if ratio < 2:
                    print(r'$B_{gen}:%.4e$' % (sqrt(rho_B_1 + rho_B_2) * m ** 2 * Mpl ** 2 * a_map ** -2)[-1])
                    print(r'$\Lambda_{gen}:%.4e \quad \Lambda_{gen, peak}:%.4e$' % (
                    2 / (m * Mpl), (a_f / (max_k * m * Mpl))))
                    B_k_f = (m ** 2 * (max_k / (a_f / a_i)) ** 2 * (max_A / max_A0)) * Mpl ** 2;
                    print(r'$B_{gen, peak}:%.4e$' % (B_k_f));

                    Lambda_0 = Lambda_0(Lambda_f, B_k_f);
                    B_0 = B_0(Lambda_0)
                    print(r'$\Lambda_{m,0}:%.4e$' % (Lambda_0))

                    print(r'$B_{0, peak}:%.4e$' % (B_0))
                    # a_R = 1e-29 ##Subhramanian entropy conservation
                    cross_idx_1 = np.argwhere(np.diff(np.sign(np.absolute(
                        rho_E_1 / a_map ** 2 + rho_B_1 / a_map ** 4) - 3 * H_map ** 2 * m ** 2 * Mpl ** 4))).flatten()
                    cross_idx_2 = np.argwhere(np.diff(np.sign(np.absolute(
                        rho_E_2 / a_map ** 2 + rho_B_2 / a_map ** 4) - 3 * H_map ** 2 * m ** 2 * Mpl ** 4))).flatten()

                    B_k_f_1 = sqrt((rho_B_1 / a_map ** 4)[cross_idx_1]) * (m * Mpl) ** 2




    return