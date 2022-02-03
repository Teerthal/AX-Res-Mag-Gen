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
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib import ticker


# from numdifftools import Derivative
flatten = np.ndarray.flatten
# from scipy.interpolate import UnivariateSpline
# from scipy import interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'
    , 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


plt.rcParams['axes.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20


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

N = 10000
t_i = 0
t_f = N*pi + t_i

a_i = 1.

phi_i = 3.
dphi_i = 0.

steps = 1e5
gauge_steps = 1e3

H_i = 2

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


def mapper(var, var_arr , target_arr):

    diff_arr = abs(var_arr-var)
    min_idx = np.argmin(diff_arr)

    out = target_arr[min_idx]

    return out


def V(phi,p):

    V = (1-1/(1+((phi)**2))**p)
    V_norm = 0.5*(1 - 1 / (1 + ((phi) ** 2) / p) ** p)

    V_mon = phi**p + 0.1*cos(phi)

    return V_norm

def dV(phi,p):

    dV = p*(1+((phi)**2))**(-p-1)*(2*phi)
    dV_norm = 0.5*p * (1 + ((phi) ** 2) / p) ** (-p - 1) * (2 * phi / p)
    dV_mon = p*phi**(p-1) - cos(phi)

    return dV_norm

def H(args, p, f):
    [dphi, phi] = args
    H = 1/3*f*sqrt(0.5*absolute(dphi)**2+V(phi,p))/(sqrt(V(phi_i,p)))

    return H

def a(t,p):

    a = a_i*t**p

    return a_i

def Scalar_ODE(t,u, arg):

    p = arg[0]
    f = arg[1]

    phi = u[0]
    dphi = u[1]

    H_i = H([dphi,phi], p, f)

    dV_dphi = dV(phi,p)
    sys = [dphi, -3*H_i*dphi - dV_dphi]

    return sys

def Scalar_solver(arg, *, initial= [phi_i,dphi_i]):

    [p, f] = arg

    ode_paras = np.array([p, f])
    solver = ode(Scalar_ODE).set_integrator('lsoda').set_f_params(ode_paras)

    #initial = np.array([phi_i, dphi_i])

    solver.set_initial_value(initial, t_i)

    t = []; u_0 = []; u_1 = []

    dt = (t_f-t_i)/steps

    #while solver.successful() and solver.y[1]**2./2.<=1:
    #while solver.successful() and phi_i>= solver.y[0] >= sqrt(2):
    while solver.successful() and t_i<=solver.t <=t_f:

        solver.integrate(solver.t + dt)

        u_0.append(solver.y[0])
        u_1.append(solver.y[1])
        t.append(solver.t)

    u_0 = np.array(u_0)
    u_1 = np.array(u_1)
    t = np.array(t)

    return np.array([t,u_0,u_1])


def ODE_sys(t,u,arg):
    p = arg[0]
    h = arg[1]
    alpha   = arg[2]
    k = arg[3]
    f = arg[4]
    a_stack = arg[5]

    phi = u[0]
    dphi = u[1]
    A = u[2]
    dA = u[3]

    H_i = H([dphi, phi], p, f)
    a_t = mapper(t, a_stack[0], a_stack[1])

    dV_dphi = dV(phi, p)

    sys = [
        dphi, -3 * H_i * dphi - dV_dphi,
        dA, -H_i*dA-((k/a_t)**2-(h*alpha*k/a_t*dphi))*A

    ]

    return sys

def ODE_solver(k, arg):

    [p,h,alpha, f, a_stack, t_i, t_f, phi_i, dphi_i] = arg
    ode_para = np.array([p,h,alpha,k, f, a_stack])

    solver = ode(ODE_sys).set_integrator('zvode').set_f_params(ode_para)

    a_t_i = mapper(t_i, a_stack[0], a_stack[1])

    A_i = 1.
    dA_i = 1j*k/a_t_i

    initial = [phi_i,dphi_i,A_i,dA_i]
    solver.set_initial_value(initial, t_i)

    t = []; u_0 = []; u_1 = []; u_2 = []; u_3 = []

    dt = (t_f-t_i)/steps

    #while solver.successful() and solver.y[1]**2./2.<=1:
    #while solver.successful() and phi_i>= solver.y[0] >= sqrt(2):
    while solver.successful() and t_i<=solver.t <=t_f:

        solver.integrate(t_f, step=True)

        u_0.append(solver.y[0])
        u_1.append(solver.y[1])
        u_2.append(solver.y[2])
        u_3.append(solver.y[3])
        t.append(solver.t)

    u_0 = np.array(u_0)
    u_1 = np.array(u_1)
    u_2 = np.array(u_2)
    u_3 = np.array(u_3)
    t = np.array(t)

    return [t,u_0,u_1,u_2,u_3]


def ODE_spooler(p,alpha,k,a_stack, Phi_stack, kill_matrix,trigger_mattrix, Master_path, f):

    start = time.time()

    h = [-1,1]

    stack_f = []
    for idx_f, item_f in enumerate(f):

        stack_p = []
        for idx_p, item_p in enumerate(p):
            t_i = trigger_mattrix[idx_f][idx_p]
            t_f = kill_matrix[idx_f][idx_p]

            phi_i = mapper(t_i, Phi_stack[idx_f][idx_p, 0], Phi_stack[idx_f][idx_p, 1])
            dphi_i = mapper(t_i, Phi_stack[idx_f][idx_p, 0], Phi_stack[idx_f][idx_p, 2])

            p_i = p[idx_p]
            stack_h = []
            for idx_h, item_h in enumerate(h):
                stack_alpha = []
                for idx_alpha, item_alpha in enumerate(alpha):

                    arg = [item_p,item_h,item_alpha, item_f, a_stack[idx_f][idx_p], t_i, t_f, phi_i, dphi_i]

                    pool = Pool(cpu_count())
                    par = functools.partial(ODE_solver,
                                            arg=arg)

                    sol_k = np.array(pool.map(par, k[idx_f][idx_p]))
                    pool.close()

                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f'%(item_alpha,item_h,item_p, item_f)
                    save(sol_k,name, 'Raw', Master_path=Master_path)

                    timer('Finished stack alpha:%.5f_h:%s_p:%.3f:_f_%.5f'%(item_alpha,item_h,item_p, item_f), start)

    return

def Phi_Spooler(p, f):

    stack_f = []
    for item_f in f:
        stack_p = []
        for p_i in p:
            sol = Scalar_solver([p_i, item_f])
            stack_p.append(sol)
        stack_f.append(stack_p)

    return np.array(stack_f)

def a_spooler(p, Phi_stack, f):

    stack_f = []
    for idx_f, item_f in enumerate(f):

        a_stack = []
        for idx_p in np.arange(0,len(p),1):
            p_i = p[idx_p]
            substack = Phi_stack[idx_f][idx_p]

            t = substack[0]
            phi = substack[1]
            dphi = substack[2]

            par_H = functools.partial(H, p=p_i, f=item_f)
            args = np.array(list(zip(phi,dphi)))
            H_arr = np.array([*map(par_H, args)])

            integ = integrate.cumtrapz(H_arr, t)
            a = a_i*exp(integ)
            t = np.delete(t, 0)

            a_stack.append([t,a])
        stack_f.append(a_stack)

    return np.array(stack_f)

def initiation_matrix(args):
    [p, Phi_stack, use_limit_of, lim_val, f] = args
    print(np.shape(Phi_stack))
    stack_f = []
    for idx_f, item_f in enumerate(f):
        trigger_t = []
        for idx_p, item_p in enumerate(p):
            stack = Phi_stack[idx_f][idx_p]
            [t, phi, dphi] = [stack[i] for i in [0, 1, 2]]
            if use_limit_of == 'Phi':
                diff_arr = abs(np.array(phi)-lim_val)
                idx = np.argmin(diff_arr)
                t_low = t[idx]
            else:
                diff_arr = abs(np.array(dphi) - lim_val)
                idx = np.argmin(diff_arr)
                t_low = t[idx]
            trigger_t.append(t_low)
        stack_f.append(trigger_t)

    return stack_f

def kill_matrix(args):
    [p, Phi_stack, splice, use_limit_of, lim_val, f, cross_lim] = args
    splice = int(splice)

    stack_f = []
    for idx_f, item_f in enumerate(f):

        kill_t = []
        for idx_p, item_p in enumerate(p):
            stack = Phi_stack[idx_f][idx_p]
            [t,phi, dphi] = [stack[i] for i in [0,1,2]]
            t_splice = np.array_split(t, splice)
            phi_splice = np.array_split(phi, splice)
            dphi_splice = np.array_split(dphi, splice)

            run_idx = np.arange(0, len(t_splice), 1)
            avg_t = [np.mean(t_splice[i]) for i in run_idx]

            cross_idx = np.argwhere(np.diff(np.sign((dphi) - [0] * len(dphi)))).flatten();#print(len(cross_idx))
            
            if len(cross_idx) != 0:


                if use_limit_of == 'Phi':

                    max_val_idxs = [np.argmax(phi_splice[i]) for i in run_idx]
                    max_arr = [phi_splice[i][z] for [i,z] in zip(run_idx, max_val_idxs)]
                    diff_arr = abs(np.array(max_arr)-lim_val)
                    idx = np.argmin(diff_arr)
                    #t = avg_t[idx]
                else:
                    max_val_idxs = [np.argmax(dphi_splice[i]) for i in run_idx]
                    max_arr = [phi_splice[i][z] for [i, z] in zip(run_idx, max_val_idxs)]
                    diff_arr = abs(np.array(max_arr) - lim_val)
                    idx = np.argmin(diff_arr)
                    #t = avg_t[idx]

                t = t[cross_idx[cross_lim]]
            else:
                t = t[-1]
            kill_t.append(t)
        stack_f.append(kill_t)

    return stack_f


def Gauge_ODE(t, u, arg):
    idx_p = arg[0]
    h = arg[1]
    alpha = arg[2]
    k = arg[3]
    p = arg[4]
    Phi_stack = arg[6]
    f = arg[7]
    idx_f = arg[8]
    a_stack = arg[5][idx_f][idx_p]

    A = u[0]
    dA = u[1]

    [dphi, phi] = [
        mapper(t, Phi_stack[idx_f][idx_p, 0], Phi_stack[idx_f][idx_p, 2]),
        mapper(t, Phi_stack[idx_f][idx_p, 0], Phi_stack[idx_f][idx_p, 1])
    ]
    a_t = mapper(t, a_stack[0], a_stack[1])
    H_i = H([dphi, phi], p, f)

    sys = [dA, -H_i * dA - ((k / a_t) ** 2 - (h * alpha * k / a_t * dphi))*A]

    return sys

def Gauge_solver(item_k,*,  arg, a_stack, Phi_stack):
    [idx_p,h,alpha,p, t_f, t_i, f, idx_f] = arg

    ode_para = [idx_p, h, alpha, item_k,p, a_stack, Phi_stack, f, idx_f]

    solver = ode(Gauge_ODE).set_integrator('zvode').set_f_params(ode_para)

    a_t_i = mapper(t_i, a_stack[idx_p][0], a_stack[idx_f][idx_p][1])
    A_i = 1.
    dA_i = 1j * item_k / a_t_i

    initial = [A_i, dA_i]
    solver.set_initial_value(initial, t_i)

    t = [];
    u_0 = [];
    u_1 = [];
    dt = (t_f - t_i) / gauge_steps

    # while solver.successful() and solver.y[1]**2./2.<=1:
    # while solver.successful() and phi_i>= solver.y[0] >= sqrt(2):
    while solver.successful() and t_i <= solver.t <= t_f:
        #solver.integrate(solver.t + dt)
        solver.integrate(t_f, step=True)
        u_0.append(solver.y[0])
        u_1.append(solver.y[1])

        t.append(solver.t)
    u_0 = np.array(u_0)
    u_1 = np.array(u_1)
    t = np.array(t)
    return [t, u_0, u_1]




def Gauge_spooler(p,alpha,k,a_stack, Phi_stack, kill_matrix,trigger_mattrix, Master_path, f):

    start = time.time()

    h = [-1,1]

    stack_f = []
    for idx_f, item_f in enumerate(f):

        stack_p = []
        for idx_p in np.arange(0,len(p), 1):
            t_i = trigger_mattrix[idx_f][idx_p]
            t_f = kill_matrix[idx_f][idx_p]
            p_i = p[idx_p]
            stack_h = []
            for h_i in h:
                stack_alpha = []
                for alpha_i in alpha:

                    arg = [idx_p, h_i, alpha_i, p_i, t_f, t_i, item_f, idx_f]

                    pool = Pool(cpu_count())
                    par = functools.partial(Gauge_solver,
                                            arg=arg, a_stack=a_stack, Phi_stack=Phi_stack)

                    sol_k = np.array(pool.map(par, k[idx_f][idx_p]))
                    pool.close()

                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f'%(alpha_i,h_i,p_i, item_f)
                    save(sol_k,name, 'Raw', Master_path=Master_path)

                    timer('Finished stack alpha:%.5f_h:%s_p:%.3f:_f_%.5f'%(alpha_i,h_i,p_i, item_f), start)

    return

def Asymptotes(p,alpha,k,a_stack, Phi_stack, f, Master_path):

    h = [-1,1]

    stack_f = []
    for idx_f, item_f in enumerate(f):
        stack_h = []
        for idx_h, item_h in enumerate(h):
            stack_p = []
            for idx_p, item_p in enumerate(p):
                stack_alpha = []
                for idx_alpha, item_alpha in enumerate(alpha):
                    stack_k = []
                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f'%(item_alpha,item_h,item_p, item_f)
                    stack = load(name, 'Raw', Master_path)

                    for idx_k, item_k in enumerate(k[idx_f][idx_p]):
                        sub_stack = stack[idx_k]

                        A = sub_stack[3][-1]
                        dA = sub_stack[4][ -1]
                        asyms = [item_k, A, dA]
                        stack_k.append(asyms)
                    stack_alpha.append(stack_k)
                stack_p.append(stack_alpha)
            stack_h.append(stack_p)
        stack_f.append(stack_h)

    return np.array(stack_f)

def Shredder_B(args ,Gauge_stack):
    [x, k, a_stack,idx_p, idx_f] = args

    A_z = []
    for idx_k in range(len(k)):

        A_k = mapper(x, Gauge_stack[idx_k][0], Gauge_stack[idx_k][3])
        A_z.append(absolute(A_k))

    A_z = np.array(A_z)

    a_x = mapper(x, a_stack[idx_f][idx_p, 0], a_stack[idx_f][idx_p, 1])
    Integrand = 1 / (4 * pi ** 2) * A_z**2 * k ** 4 / a_x**4
    integrated = trapz(Integrand, x=log(k))

    return integrated


def Rho_B_spooler(p,alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix):

    h = [-1,1]

    stack_f = []
    for idx_f, item_f in enumerate(f):
        stack_p = []
        for idx_p, item_p in enumerate(p):
            stack_alpha = []
            for idx_alpha, item_alpha in enumerate(alpha):
                stack_h = []
                for idx_h, item_h in enumerate(h):

                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, item_h, item_p, item_f)
                    gauge_stack = load(name, 'Raw', Master_path)
                    sub_stack = gauge_stack
                    stack_x = []

                    t_i = trigger_matrix[idx_f][idx_p]
                    t_f = kill_matrix[idx_f][idx_p]

                    for x in np.linspace(t_i, t_f, x_len):
                        Shredder_args = [x,k[idx_f][idx_p], a_stack, idx_p, idx_f]
                        Rho_B = Shredder_B(Shredder_args, sub_stack)

                        stack_x.append([x, Rho_B])

                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, item_h, item_p, item_f)
                    save(stack_x, name, 'Rho_B', Master_path=Master_path)
    return


def Shredder_E(args ,Gauge_stack):
    [x, k, a_stack,idx_p, idx_f] = args

    del_A_z = []
    for idx_k in range(len(k)):

        del_A_k = mapper(x, Gauge_stack[idx_k][0], Gauge_stack[idx_k][4])
        del_A_z.append(absolute(del_A_k))

    del_A_z = np.array(del_A_z)

    a_x = mapper(x, a_stack[idx_f][idx_p, 0], a_stack[idx_f][idx_p, 1])

    Integrand = 1 / (4 * pi ** 2) * k / a_x ** 2 * (del_A_z ** 2)
    integrated = trapz(Integrand, x=k)

    return integrated



def Shredder_Back(args, Gauge_stack):
    [x, k, a_stack,idx_p, idx_f] = args

    A_z = [];del_A_z = []
    for idx_k in range(len(k)):

        del_A_k = mapper(x, Gauge_stack[idx_k][0], Gauge_stack[idx_k][4])
        
        del_A_z.append(absolute(del_A_k))

        
        A_k = mapper(x, Gauge_stack[idx_k][0], Gauge_stack[idx_k][3])
        A_z.append(absolute(A_k))
    del_A_z = np.array(del_A_z)
    A_z = np.array(A_z)

    Integrand = k ** 3 / (8 * pi ** 2) * 2 * absolute(A_z)*absolute(del_A_k)

    return trapz(Integrand, x=log(k))


def BR_spooler(p,alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix):

    h = [-1,1]
    start = time.time()
    stack_f = []
    for idx_f, item_f in enumerate(f):

        stack_p = []
        for idx_p, item_p in enumerate(p):
            stack_alpha = []
            for idx_alpha, item_alpha in enumerate(alpha):
                stack_h = []
                for idx_h, item_h in enumerate(h):

                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, item_h, item_p, item_f)
                    gauge_stack = load(name, 'Raw', Master_path)
                    sub_stack = gauge_stack

                    t_i = trigger_matrix[idx_f][idx_p]
                    t_f = kill_matrix[idx_f][idx_p]

                    stack_x = []
                    for x in np.geomspace(t_i, t_f, x_len):
                        Shredder_args = [x,k[idx_f][idx_p], a_stack, idx_p, idx_f]
                        Rho_E = Shredder_Back(Shredder_args, sub_stack)

                        stack_x.append([x, Rho_E])
                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, item_h, item_p, item_f)
                    save(stack_x, name, 'Backreaction', Master_path=Master_path)
                    timer('Finished stack alpha:%.5f_h:%s_p:%.3f:_f_%.5f'%(item_alpha,item_h,item_p, item_f), start)
    return


def Rho_E_spooler(p,alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix):

    h = [-1,1]

    stack_f = []
    for idx_f, item_f in enumerate(f):

        stack_p = []
        for idx_p, item_p in enumerate(p):
            stack_alpha = []
            for idx_alpha, item_alpha in enumerate(alpha):
                stack_h = []
                for idx_h, item_h in enumerate(h):

                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, item_h, item_p, item_f)
                    gauge_stack = load(name, 'Raw', Master_path)
                    sub_stack = gauge_stack

                    t_i = trigger_matrix[idx_f][idx_p]
                    t_f = kill_matrix[idx_f][idx_p]

                    stack_x = []
                    for x in np.linspace(t_i, t_f, x_len):
                        Shredder_args = [x,k[idx_f][idx_p], a_stack, idx_p, idx_f]
                        Rho_E = Shredder_E(Shredder_args, sub_stack)

                        stack_x.append([x, Rho_E])
                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, item_h, item_p, item_f)
                    save(stack_x, name, 'Rho_E', Master_path=Master_path)

    return

def Shredder_helicity(args, Master_path):
    [x, k, a_stack,idx_p, item_p, item_alpha, item_f] = args

    h = [-1,1]
    name_list = ['alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, item_h, item_p, item_f) for item_h in h]
    gauge_stack = [load(name, 'Raw', Master_path) for name in name_list]

    for idx_k in range(len(k)):
        A_k = [mapper(x, gauge_stack[i][idx_k][0], gauge_stack[i][idx_k][3]) for i in [0,1]]

    Integrand = 1/ (8*pi**2) * k**2 * (A_k[0]**2 - A_k[1]**2)
    integrated = trapz(Integrand, x = k)

    return integrated


def Helicity_spooler(p,alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix):
    h = [-1,1]

    stack_f = []
    for idx_f, item_f in enumerate(f):

        stack_p = []
        for idx_p, item_p in enumerate(p):
            stack_alpha = []
            for idx_alpha, item_alpha in enumerate(alpha):
                t_i = trigger_matrix[idx_f][idx_p]
                t_f = kill_matrix[idx_f][idx_p]
                stack_x = []
                for x in np.geomspace(t_i, t_f, x_len):
                    Shredder_args = [x, k[idx_f][idx_p], a_stack, idx_p, item_p, item_alpha, item_f]
                    Hel = Shredder_helicity(Shredder_args, Master_path)

                    stack_x.append([x, Hel])
                print(np.shape(stack_x))
                name = 'alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                save(stack_x, name, 'Helicity', Master_path=Master_path)

    return

def intg_par(i, A_h, intg_k):
    Integrand = trapz(intg_k ** 2 * (A_h[0,:,i] ** 2 - A_h[1,:,i] ** 2), x=intg_k)

    return Integrand

def intg_par_2(i, A_h, intg_k):
    Integrand = trapz(intg_k ** 2 * (A_h[0,:,i] ** 2 + A_h[1,:,i] ** 2), x=intg_k)
    return Integrand

def hel_spooler_2(p, alpha, k_stack, a_stack, Phi_stack, Master_path, f, x_len, trigger_matrix, kill_matrix):

    h = [-1,1]
    h_style = ['-', '--']
    #[plt.plot([],[],label = 'h:%s'%(h[i]), linestyle=h_style[i]) for i in [0,1]]
    #ax = [ 0, 0]
    counter = 0
    for idx_f, item_f in enumerate(f):
        for idx_p, item_p in enumerate(p):
            t_i = trigger_matrix[idx_f][idx_p]
            t_f = kill_matrix[idx_f][idx_p]
            x = np.linspace(t_i, t_f, x_len)
            k = k_stack[idx_f][idx_p]
            intg_k = np.round(np.linspace(0, len(k)-1 , 500)).astype(int)
            for cnt_alpha, idx_alpha in enumerate(np.arange(0,len(alpha)-1,1)):
                item_alpha = alpha[idx_alpha]
                A_h = []
                for idx_h, item_h in enumerate(h):
                    style_h = h_style[idx_h]

                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f'% (item_alpha, item_h, item_p, item_f)
                    stack = load(name, 'Raw', Master_path)
                    
                    A_x = []
                    for subidx_k, idx_k in enumerate(intg_k):
                        item_k = k[idx_k];#print(idx_k)

                        sub_stack = stack[idx_k]

                        t = sub_stack[0]
                        #phi = sub_stack[1]
                        #dphi = sub_stack[2]
                        A = absolute(sub_stack[3])
                        dA = sub_stack[4]

                        sol = Phi_stack[idx_f][idx_p]
                        t_phi = sol[0]
                        phi = sub_stack[1]
                        dphi = sub_stack[2]

                        A_x.append([mapper(x_i, t, A) for x_i in x])

                        t_a = a_stack[idx_f][idx_p, 0]
                        a = a_stack[idx_f][idx_p, 1]

                    A_h.append(A_x)
                A_h = np.array(A_h)
                print(np.shape(A_h))

                pool = Pool(cpu_count())
                par_1 = functools.partial(intg_par, A_h = A_h, intg_k=intg_k)

                Integrand = np.array(pool.map(par_1, range(int(x_len))))
                pool.close()
                
                pool = Pool(cpu_count())
                par_2 = functools.partial(intg_par_2, A_h = A_h, intg_k=intg_k)

                Integrand_2 = np.array(pool.map(par_2, range(int(x_len))))
                pool.close()

                
                norm_int = np.array(Integrand)/np.array(Integrand_2)
                stacked = np.array([x, norm_int, Integrand, Integrand_2])
                name = 'resolved alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                save(stacked, name, 'Helicity', Master_path=Master_path)
                counter = counter +  1; print(counter)
                #plt.plot(x,norm_int, label='f:%.4f alpha:%.2f'%(item_f, item_alpha))
    #plt.legend();plt.show()
    return


def hel_spooler_sel(p, alpha, k_stack, a_stack, Phi_stack, Master_path, f, x_len, trigger_matrix, kill_matrix, targ_f,
                  targ_alpha):
    alpha_idxs = [np.argmin(abs(targ - alpha)) for targ in targ_alpha]
    f_idxs = [np.argmin(abs(targ - f)) for targ in targ_f]

    h = [-1, 1]
    h_style = ['-', '--']
    # [plt.plot([],[],label = 'h:%s'%(h[i]), linestyle=h_style[i]) for i in [0,1]]
    # ax = [ 0, 0]
    counter = 0
    for cnt_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        for idx_p, item_p in enumerate(p):
            t_i = trigger_matrix[idx_f][idx_p]
            t_f = kill_matrix[idx_f][idx_p]
            x = np.linspace(t_i, t_f, x_len)
            k = k_stack[idx_f][idx_p]
            intg_k = np.round(np.linspace(0, len(k) - 1, 500)).astype(int)
            for cnt_alpha, idx_alpha in enumerate(alpha_idxs):
                item_alpha = alpha[idx_alpha]
                A_h = []
                for idx_h, item_h in enumerate(h):
                    style_h = h_style[idx_h]

                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, item_h, item_p, item_f)
                    stack = load(name, 'Raw', Master_path)

                    A_x = []
                    for subidx_k, idx_k in enumerate(intg_k):
                        item_k = k[idx_k];  # print(idx_k)

                        sub_stack = stack[idx_k]

                        t = sub_stack[0]
                        # phi = sub_stack[1]
                        # dphi = sub_stack[2]
                        A = absolute(sub_stack[3])
                        dA = sub_stack[4]

                        sol = Phi_stack[idx_f][idx_p]
                        t_phi = sol[0]
                        phi = sub_stack[1]
                        dphi = sub_stack[2]

                        A_x.append([mapper(x_i, t, A) for x_i in x])

                        t_a = a_stack[idx_f][idx_p, 0]
                        a = a_stack[idx_f][idx_p, 1]

                    A_h.append(A_x)
                A_h = np.array(A_h)
                print(np.shape(A_h))

                pool = Pool(cpu_count())
                par_1 = functools.partial(intg_par, A_h=A_h, intg_k=intg_k)

                Integrand = np.array(pool.map(par_1, range(int(x_len))))
                pool.close()

                pool = Pool(cpu_count())
                par_2 = functools.partial(intg_par_2, A_h=A_h, intg_k=intg_k)

                Integrand_2 = np.array(pool.map(par_2, range(int(x_len))))
                pool.close()

                norm_int = np.array(Integrand) / np.array(Integrand_2)
                stacked = np.array([x, norm_int, Integrand, Integrand_2])
                name = 'resolved alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                save(stacked, name, 'Helicity', Master_path=Master_path)
                counter = counter + 1;
                print(counter)
                # plt.plot(x,norm_int, label='f:%.4f alpha:%.2f'%(item_f, item_alpha))
    # plt.legend();plt.show()
    return
