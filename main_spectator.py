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
import matplotlib.animation as animation
import h5py
import hickle as hkl

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

N = 1000
t_i = 0.
t_f = N*pi + t_i

t_f = 700
t_f = 500

a_i = 1.

phi_i = 3
#phi_i = 4
dphi_i = 0
steps = 1e4
gauge_steps = 1e3
f = 1
tilde_t_i = 1/2
eps_H = 0.01
#eps_H = 0.1

#H_m = 0.05


Bck = 'exp'#input('Use background(cons_a/exp/rad):')

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
    #hkl.dump(data, path, mode='w')
    
    #h5f = h5py.File(path, 'w')
    #h5f.create_dataset('dataset_1', data=data)

    return


def load(name, directory, Master_path):
    path = '%s/%s/%s.npy' % (Master_path, directory, name)
    file = open(path, 'rb')
    stack = np.load(file)
    file.close()
    #stack = hkl.load(path)

    #h5f = h5py.File(path,'r')
    #stack = h5f['dataset_1'][:]
    #h5f.close()

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

    V = (1-1/(1+((phi)**2)/p)**p)

    return V

def dV(phi,p):

    dV = p*(1+(phi)**2)**(-p-1)*(2*phi)

    return dV

def a1(t,p, H_m):

    a = a_i*t**p
    if Bck == 'cons_a':
        a = 1

    if Bck == 'exp':
        a = exp(H_m*t)

    if Bck == 'rad':
        a = a_i * ((t+tilde_t_i)/tilde_t_i)**(1/2)
    return a

def a(t,t_i, t_f):
    return exp(t-t_i)

def H(p,t, H_m):
    #[dphi, phi] = args
    #H = f**2/3*(dphi**2/2+V(phi,p))
    if Bck == 'cons_a':
        H = H_m
    if Bck == 'exp':
        H = exp(t) ** (-eps_H)
    #H = p/t
    if Bck == 'rad':
        H = 1/(2*(t+tilde_t_i))
    return H


def Scalar_ODE(t,u, arg):

    p = arg[0]
    H_m = arg[1]
    phi = u[0]
    dphi = u[1]

    H_i = H(p,t, H_m)
    #H_i = exp(t)**(-eps_H)
    dV_dphi = dV(phi,p)
    sys = [dphi, -3 * dphi - dV_dphi / (H_i)**2]

    return sys

def Scalar_solver(arg, *, initial= [phi_i,dphi_i]):

    p,H_m = arg

    ode_paras = np.array([p, H_m])
    solver = ode(Scalar_ODE).set_integrator('vode', max_step=1e-3).set_f_params(ode_paras)

    #initial = np.array([phi_i, dphi_i])

    solver.set_initial_value(initial, t_i)

    t = []; u_0 = []; u_1 = []

    dt = (t_f-t_i)/steps

    #while solver.successful() and solver.y[1]**2./2.<=1:
    #while solver.successful() and phi_i>= solver.y[0] >= sqrt(2):
    while solver.successful() and t_i<=solver.t <=t_f:

        #solver.integrate(solver.t + dt)
        solver.integrate(t_f, step=True)
        u_0.append(solver.y[0])
        u_1.append(solver.y[1])
        t.append(solver.t)

    u_0 = np.array(u_0)
    u_1 = np.array(u_1)
    t = np.array(t)
    print(len(t))
    return [t,u_0,u_1]


def Phi_Spooler(p,H_m):

    stack_p = []
    for p_i in p:
        stack_H = []
        for H_i in H_m:
            sol = Scalar_solver([p_i,H_i])
            stack_H.append(sol)
        stack_p.append(stack_H)
    return stack_p

def a_spooler(p, H_m, Phi_stack):

    stack_p = []
    for idx_p, item_p in enumerate(p):
        stack_H = []
        for idx_H, item_H in enumerate(H_m):
            substack = Phi_stack[idx_p][idx_H]
            t = substack[0]
            stack_H.append([t, exp(t)])
        stack_p.append(stack_H)
    return stack_p

def initiation_matrix(args):
    [p, H_m, Phi_stack, use_limit_of, lim_val, lnA_thresh] = args
    trigger_t_p = []
    for idx_p, item_p in enumerate(p):
        trigger_t_H = []
        for idx_H, item_H in enumerate(H_m):
            stack = Phi_stack[idx_p][idx_H]
            [t, phi, dphi] = [stack[i] for i in [0, 1, 2]]
            if use_limit_of == 'Phi':
                diff_arr = abs(np.array(phi) - lim_val)
                idx = np.argmin(diff_arr)
                t_low = t[idx]
            else:
                diff_arr = abs(np.array(dphi) - lim_val)
                idx = np.argmin(diff_arr)
                t_low = t[idx]

            #max_dphi_idx = np.argmax(abs(dphi))
            #del_t = t[max_dphi_idx] - lnA_thresh
            trigger_t_H.append(t_low)
        trigger_t_p.append(trigger_t_H)
    return trigger_t_p


def initiation_matrix_red(args):
    [p, H_m, Phi_stack, use_limit_of, lim_val, lnA_thresh] = args
    trigger_t_p = []
    for idx_p, item_p in enumerate(p):
        trigger_t_H = []
        for idx_H, item_H in enumerate(H_m):
            stack = Phi_stack[idx_p][idx_H]
            [t, phi, dphi] = [stack[i] for i in [0, 1, 2]]
            if use_limit_of == 'Phi':
                diff_arr = abs(np.array(phi) - lim_val)
                idx = np.argmin(diff_arr)
                t_low = t[idx]
            else:
                diff_arr = abs(np.array(dphi) - lim_val)
                idx = np.argmin(diff_arr)
                match_idxs = np.argwhere(diff_arr < 1e-6)
                print('# of matches:', len(match_idxs))
                t_low = t[idx]
                t_low = t[match_idxs[0]]

            max_dphi_idx = np.argmax(abs(dphi))
            del_t = t[max_dphi_idx] - lnA_thresh
            trigger_t_H.append(del_t)
        trigger_t_p.append(trigger_t_H)
    return trigger_t_p

def kill_matrix(args):
    [p, H_m, Phi_stack, splice, use_limit_of, lim_val, cross_lim, trig_arr, lnA_thresh] = args
    splice = int(splice)
    kill_t = []
    for idx_p, item_p in enumerate(p):
        kill_t_H = []
        for idx_H, item_H in enumerate(H_m):
            t_i = trig_arr[idx_p][idx_H]

            stack = Phi_stack[idx_p][idx_H]
            [t, phi, dphi] = [stack[i] for i in [0, 1, 2]]
            len_o = len(t); splice = int(len_o/splice)

            mx_idx = np.argmax(abs(dphi))
            [t, phi, dphi] = [t[mx_idx:], phi[mx_idx:], dphi[mx_idx:]]
            #print(np.shape(t))
            t_splice = np.array_split(t, splice)
            phi_splice = np.array_split(phi, splice)
            dphi_splice = np.array_split(dphi, splice)
            #print(np.shape(phi_splice))
            run_idx = np.arange(0, len(t_splice), 1)
            avg_t = [np.mean(t_splice[i]) for i in run_idx]

            if use_limit_of == 'Phi':

                max_val_idxs = [np.argmax(phi_splice[i]) for i in run_idx]
                max_arr = [phi_splice[i][z] for [i, z] in zip(run_idx, max_val_idxs)]
                diff_arr = abs(np.array(max_arr) - lim_val)
                idx = np.argmin(diff_arr)
                t = avg_t[idx]

            else:
                max_val_idxs = [np.argmax(dphi_splice[i]) for i in run_idx]
                max_arr = [phi_splice[i][z] for [i, z] in zip(run_idx, max_val_idxs)]
                diff_arr = abs(np.array(max_arr) - lim_val)
                idx = np.argmin(diff_arr)
                t = avg_t[idx]

            kill_t_H.append(t)
        kill_t.append(kill_t_H)
    return kill_t



def kill_matrix_red(args):
    [p, H_m, Phi_stack, splice, use_limit_of, lim_val, cross_lim, trig_arr, lnA_thresh] = args
    splice = int(splice)
    kill_t = []
    for idx_p, item_p in enumerate(p):
        kill_t_H = []
        for idx_H, item_H in enumerate(H_m):
            t_i = trig_arr[idx_p][idx_H]

            stack = Phi_stack[idx_p][idx_H]
            [t, phi, dphi] = [stack[i] for i in [0, 1, 2]]

            cross_idx = np.argwhere(np.diff(np.sign((dphi) - [0] * len(dphi)))).flatten()
            print(item_p, 'cross count', len(cross_idx))
            if len(cross_idx) >= cross_lim:
                t_cross = t[cross_idx[cross_lim]]
                del_t = t_cross
            else:
                t_cross = t[cross_idx[-1]]
                #t_cross = t[cross_idx[-1]]

            kill_t_H.append(t_cross)
        kill_t.append(kill_t_H)
    return kill_t


def Gauge_ODE(t, u, arg):
    idx_p = arg[0]
    h = arg[1]
    alpha = arg[2]
    k = arg[3]
    p = arg[4]
    H_m = arg[5]
    #a_stack = arg[5][idx_p]
    #a_t = a(t,p, H_m)

    phi = u[0]
    dphi = u[1]
    A = u[2]
    dA = u[3]

    H_i = H(p,t, H_m)
    #H_i = t**(-0.01)
    #H_i = exp(t)**(-0.01)
    a = exp(t)

    dV_dphi = dV(phi, p)

    sys = [
        dphi, -3 * dphi - dV_dphi / (H_i) ** 2,
        dA, -dA - ((k / (H_i*a)) ** 2 - (h * alpha * k / (H_i*a) * dphi))*A
    ]

    return sys

def Gauge_solver(item_k,*,  arg):
    [idx_p,h,alpha,p, t_f, t_i, item_H, phi_arr] = arg

    ode_para = [idx_p, h, alpha, item_k,p, item_H]

    solver = ode(Gauge_ODE).\
        set_integrator('zvode',nsteps=1e5).set_f_params(ode_para)
    
    #if p < 4:
        #delta = 0.9
    #else:
        #delta = .9

    #H_i = H(p,t_i, item_H)
    #dphi_i = mapper(t_i, phi_arr[0], phi_arr[2])
    
    #if alpha==0:
        #t_i = log(item_k*0.9/H_i)#;print(t_i)     
    #else:
        #t_i = log(item_k*delta/(alpha*abs(max(phi_arr[2]))*H_i))#;print(t_i)             

    a_i = exp(t_i)


    delta = 0.1
    #t_i = log(item_k*delta)
    #dphi_i = mapper(t_i, phi_arr[0], phi_arr[2])


    delta = 0.1
    t_i = log(item_k * delta)
    dphi_i = mapper(t_i, phi_arr[0], phi_arr[2])

    #if alpha * abs(dphi_i) > delta:
        #t_i = t_i - log(abs(dphi_i) * alpha)


    a_i = exp(t_i)


    #print(t_f-t_i)

    #a_t_i = mapper(t_i, a_stack[idx_p][0], a_stack[idx_p][1])
    #a_t_i = a(t_i, p, item_H)

    phi_i = mapper(t_i, phi_arr[0], phi_arr[1])
    dphi_i = mapper(t_i, phi_arr[0], phi_arr[2])

    #A_i = exp(1j*t_i*item_k)
    #dA_i = 1j * item_k / a_i**3*A_i
    A_i = exp(1j*-0.001*item_k)
    dA_i = -1j * item_k / a_i * A_i
    dA_i = 1j*A_i

    initial = [phi_i,dphi_i,A_i,dA_i]
    solver.set_initial_value(initial, t_i)

    t = [];
    u_0 = [];
    u_1 = [];
    u_2 = [];
    u_3 = []
    dt = (t_f - t_i) / gauge_steps

    # while solver.successful() and solver.y[1]**2./2.<=1:
    # while solver.successful() and phi_i>= solver.y[0] >= sqrt(2):
    while solver.successful() and t_i <= solver.t <= t_f:
        #solver.integrate(solver.t + dt)
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
    arr = [t, u_0, u_1, u_2, u_3]
    #if len(t)!=0 and max(absolute(u_2)) < 1:
        #emp = np.array([])
        #arr = [emp, emp, emp, emp]
    #else:
        #arr = [t, u_0, u_1, u_2, u_3]
    return arr

def Gauge_spooler(p,alpha,k, H_m, kill_matrix,trigger_mattrix, Phi_stack, Master_path):

    start = time.time()

    h = [-1,1]
    stack_p = []
    for idx_p in np.arange(0,len(p), 1):
        for idx_H, item_H in enumerate(H_m):
            t_i = trigger_mattrix[idx_p][idx_H]
            t_f = kill_matrix[idx_p][idx_H]
            p_i = p[idx_p]
            stack_h = []
            for h_i in h:
                stack_alpha = []
                for alpha_i in alpha:
                    arg = [idx_p, h_i, alpha_i, p_i, t_f, t_i, item_H, Phi_stack[idx_p][idx_H]]
                    pool = Pool(cpu_count())
                    par = functools.partial(Gauge_solver,
                                            arg=arg)
                    sol_k = np.array(pool.map(par, k[idx_p][idx_H]))
                    pool.close()
                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (alpha_i, h_i, p_i, item_H)
                    save(sol_k, name, 'Raw', Master_path=Master_path)
                    
        timer('Finished stack p:%.3f:' % (p_i), start)
    return

def Asymptotes(p, H_m, alpha,k,a_stack, Phi_stack, Master_path):

    h = [-1,1]

    stack_h = []
    for idx_h, item_h in enumerate(h):
        stack_p = []
        for idx_p, item_p in enumerate(p):
            stack_H = []
            for idx_H, item_H in enumerate(H_m):
                stack_alpha = []
                for idx_alpha, item_alpha in enumerate(alpha):
                    stack_k = []
                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H)
                    stack = load(name, 'Raw', Master_path)

                    for idx_k, item_k in enumerate(k[idx_p][idx_H]):
                        sub_stack = stack[idx_k]
                        # print(np.shape(sub_stack))

                        if len(sub_stack[0] != 0):
                            A = sub_stack[3][-1]
                            dA = sub_stack[4][-1]
                            asyms = [item_k, A, dA]
                            stack_k.append(asyms)
                        
                        else:
                            A = 1
                            dA = 1
                            asyms = [item_k, A, dA]
                            stack_k.append(asyms)


                    stack_alpha.append(stack_k)
                stack_H.append(stack_alpha)
            stack_p.append(stack_H)
        stack_h.append(stack_p)

    return np.array(stack_h)

def peaks_rms_N2(p, H_m, alpha, k, Master_path):

    h = [-1,1]
    start = time.time()
    stack_f=[]
    for idx_p, item_p in enumerate(p):
        stack_H = []
        for idx_H, item_H in enumerate(H_m):
            stack_alpha = []
            for idx_alpha, item_alpha in enumerate(alpha):
                stack_h = []
                for idx_h, item_h in enumerate(h):
                    h_i = h[idx_h]

                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H)
                    stack = load(name, 'Raw', Master_path)

                    [base_lnA, base_dphi] = [stack[0][i] for i in [0,1]]
                    base_cross_idx = np.argwhere(np.diff(np.sign(np.real(base_dphi) - [0] * len(base_dphi)))).flatten()
                    base_cross_idx = [base_cross_idx[i] for i in np.arange(1, len(base_cross_idx) - 1, 2)]
                    base_tcross = base_lnA[base_cross_idx]
                    base_dphi_rm = [sqrt(np.mean(abs(base_dphi[base_cross_idx[i]:base_cross_idx[i + 1]]) ** 2)) for i in
                            np.arange(0, len(base_tcross) - 1)]


                    stack_k = []
                    for idx_k, item_k in enumerate(k[idx_p][idx_H]):
                        sub_stack = stack[idx_k]

                        t = sub_stack[0]
                        phi = sub_stack[1]
                        dphi = sub_stack[2]
                        A = sub_stack[3]
                        dA = sub_stack[4]

                        cross_idx = np.argwhere(np.diff(np.sign(np.real(dphi) - [0] * len(dphi)))).flatten()
                        cross_idx = [cross_idx[i] for i in np.arange(1, len(cross_idx) - 1, 2)]
                        t_cross = t[cross_idx]
                        A_cross = A[cross_idx]

                        A_rm = [sqrt(np.mean(absolute(A[cross_idx[i]:cross_idx[i + 1]]) ** 2)) for i in
                                np.arange(0, len(t_cross) - 1)]
                        dphi_rm = np.array([sqrt(np.mean(abs(dphi[cross_idx[i]:cross_idx[i + 1]]) ** 2)) for i in
                                np.arange(0, len(t_cross) - 1)])
                        # print(len(np.diff(t_cross)))
                        t_mean = [np.mean(t[cross_idx[i]:cross_idx[i + 1]]) for i in np.arange(0, len(t_cross) - 1)]
                        mu_list = [log(A_rm[i + 1] / A_rm[i]) / (np.diff(t_cross)[i + 1]) for i in
                                   np.arange(0, len(A_rm) - 1)]
                        # print(A_rm[0], A_rm[-1], log(A_rm[-1]/A_rm[0]))
                        # print(mean_mu)
                        mean_mu = np.mean(mu_list)
                        #print(len(A_rm), len(mu_list))
                        if len(mu_list) != 0:
                            print(max(mu_list))

                        mu_arr = []

                        for idx_i, i in enumerate(base_dphi_rm):

                            if len(mu_list) != 0 and t_cross[0] > base_tcross[idx_i] :
                                map_idx = np.argmin(abs(dphi_rm - i))
                                #print(map_idx, len(dphi_rm))
                                mu_arr.append(mu_list[map_idx-1])
                            else:
                                mu_arr.append(0.)

                        #print(len(base_dphi_rm) - 1, len(mu_list),len(mu_arr))
                        stack_k.append([t_mean, mu_list, A_rm, mean_mu, t_cross, mu_arr])
                    stack_h.append(stack_k)
                stack_alpha.append(stack_h)
            stack_H.append(stack_alpha)
        stack_f.append(stack_H)
        print(np.shape(stack_f))
        timer('Finishing stack p%.1f'%item_p, start)
    return stack_f


def peaks_rms_N2_1(p, H_m, alpha, k, targ_p, targ_p_idxs, Master_path):

    h = [-1,1]
    start = time.time()
    stack_f=[]
    for idx_p, item_idx_p in enumerate(targ_p_idxs):
        p_i = p[item_idx_p]
        item_p = p_i
        print(item_p)
        stack_H = []
        for idx_H, item_H in enumerate(H_m):
            stack_alpha = []
            for idx_alpha, item_alpha in enumerate(alpha):
                stack_h = []
                for idx_h, item_h in enumerate(h):
                    h_i = h[idx_h]

                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H)
                    stack = load(name, 'Raw', Master_path)

                    [base_lnA, base_dphi] = [stack[0][i] for i in [0,1]]
                    base_cross_idx = np.argwhere(np.diff(np.sign(np.real(base_dphi) - [0] * len(base_dphi)))).flatten()
                    base_cross_idx = [base_cross_idx[i] for i in np.arange(1, len(base_cross_idx) - 1, 2)]
                    base_tcross = base_lnA[base_cross_idx]
                    base_dphi_rm = [sqrt(np.mean(abs(base_dphi[base_cross_idx[i]:base_cross_idx[i + 1]]) ** 2)) for i in
                            np.arange(0, len(base_tcross) - 1)]


                    stack_k = []
                    for idx_k, item_k in enumerate(k[item_idx_p][idx_H]):
                        sub_stack = stack[idx_k]

                        t = sub_stack[0]
                        phi = sub_stack[1]
                        dphi = sub_stack[2]
                        A = sub_stack[3]
                        dA = sub_stack[4]

                        cross_idx = np.argwhere(np.diff(np.sign(np.real(dphi) - [0] * len(dphi)))).flatten()
                        cross_idx = [cross_idx[i] for i in np.arange(1, len(cross_idx) - 1, 2)]
                        t_cross = t[cross_idx]
                        A_cross = A[cross_idx]

                        A_rm = [sqrt(np.mean(absolute(A[cross_idx[i]:cross_idx[i + 1]]) ** 2)) for i in
                                np.arange(0, len(t_cross) - 1)]
                        dphi_rm = np.array([sqrt(np.mean(abs(dphi[cross_idx[i]:cross_idx[i + 1]]) ** 2)) for i in
                                np.arange(0, len(t_cross) - 1)])
                        # print(len(np.diff(t_cross)))
                        t_mean = [np.mean(t[cross_idx[i]:cross_idx[i + 1]]) for i in np.arange(0, len(t_cross) - 1)]
                        mu_list = [log(A_rm[i + 1] / A_rm[i]) / (np.diff(t_cross)[i + 1]) for i in
                                   np.arange(0, len(A_rm) - 1)]
                        # print(A_rm[0], A_rm[-1], log(A_rm[-1]/A_rm[0]))
                        # print(mean_mu)
                        mean_mu = np.mean(mu_list)
                        #print(len(A_rm), len(mu_list))
                        #if len(mu_list) != 0:
                            #print(max(mu_list))

                        mu_arr = []

                        for idx_i, i in enumerate(base_dphi_rm[:10]):

                            if len(mu_list) != 0 and t_cross[0] > base_tcross[idx_i] :
                                map_idx = np.argmin(abs(dphi_rm - i))
                                #print(map_idx, len(dphi_rm))
                                mu_arr.append(mu_list[map_idx-1])
                            else:
                                mu_arr.append(0.)

                        #print(len(base_dphi_rm) - 1, len(mu_list),len(mu_arr))
                        stack_k.append([base_tcross[1:11], mu_arr])
                    stack_h.append(stack_k)
                stack_alpha.append(stack_h)
            stack_H.append(stack_alpha)
        
        name = 'p:%.3f' % (p_i)
        save(stack_H, name, 'Peaks2', Master_path=Master_path)

        #stack_f.append(stack_H)
        
        #print(np.shape(stack_f))
        timer('Finishing stack p%.1f'%item_p, start)
    return


def max_rms_N2(p, H_m, alpha, k, targ_p, targ_p_idxs, Master_path):

    h = [-1,1]

    stack_f=[]
    for idx_p, item_p_idx in enumerate(targ_p_idxs):
        item_p = p[item_p_idx]
        name = 'p:%.3f' % (item_p)
        stack = load(name, 'Peaks2',Master_path)

        for idx_H, item_H in enumerate(H_m):
            for idx_alpha, item_alpha in enumerate(alpha):
                for idx_h, item_h in enumerate(h):
                    h_i = h[idx_h]
                    for idx_k, item_k in enumerate(k[item_p_idx][idx_H]):
                        peaks = stack[idx_H][idx_alpha][idx_h][idx_k]
                        #print(len(stack[idx_p][idx_alpha][idx_h]))
                        mu_list = peaks[1]
                        if len(mu_list) == 0:
                            stack_f.append(0.)
                        else:
                            stack_f.append(np.max(mu_list))
    return np.max(stack_f)


def Shredder_B(args ,Gauge_stack):
    [x, k, a_stack,idx_p, idx_H] = args

    A_z = []
    for idx_k in range(len(k)):

        A_k = mapper(x, Gauge_stack[idx_k][0], Gauge_stack[idx_k][1])
        A_z.append(absolute(A_k))

    A_z = np.array(A_z)

    a_x = mapper(x, a_stack[idx_H, idx_p, 0], a_stack[idx_H, idx_p, 1])
    Integrand = 1 / (4 * pi ** 2) * A_z**2 * k ** 4 / a_x**4
    integrated = trapz(Integrand, x=log(k))

    return integrated


def Rho_B_spooler(p, H_m, alpha, k, x_len, a_stack, Master_path):

    h = [-1,1]

    stack_p = []
    for idx_p, item_p in enumerate(p):
        for idx_H, item_H in enumerate(H_m):
            stack_alpha = []
            for idx_alpha, item_alpha in enumerate(alpha):
                stack_h = []
                for idx_h, item_h in enumerate(h):

                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H)
                    gauge_stack = load(name, 'Raw', Master_path)
                    sub_stack = gauge_stack
                    stack_x = []
                    for x in np.linspace(t_i, t_f, x_len):
                        Shredder_args = [x,k[idx_p][idx_H], a_stack, idx_p, idx_H]
                        Rho_B = Shredder_B(Shredder_args, sub_stack)

                        stack_x.append(Rho_B)

                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H)
                    save(stack_x, name, 'Rho_B', Master_path=Master_path)
    return


def Shredder_E(args ,Gauge_stack):
    [x, k, a_stack,idx_p, idx_H] = args

    del_A_z = []
    for idx_k in range(len(k)):

        del_A_k = mapper(x, Gauge_stack[idx_k][0], Gauge_stack[idx_k][2])
        del_A_z.append(absolute(del_A_k))

    del_A_z = np.array(del_A_z)

    a_x = mapper(x, a_stack[idx_H, idx_p, 0], a_stack[idx_H, idx_p, 1])

    Integrand = 1 / (4 * pi ** 2) * k / a_x ** 2 * (del_A_z ** 2)
    integrated = trapz(Integrand, x=k)

    return integrated


def Rho_E_spooler(p, H_m, alpha, k, x_len, a_stack, Master_path):

    h = [-1,1]

    stack_p = []
    for idx_p, item_p in enumerate(p):
        for idx_H, item_H in enumerate(H_m):
            stack_alpha = []
            for idx_alpha, item_alpha in enumerate(alpha):
                stack_h = []
                for idx_h, item_h in enumerate(h):

                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H)
                    gauge_stack = load(name, 'Raw', Master_path)
                    sub_stack = gauge_stack
                    stack_x = []
                    for x in np.linspace(t_i, t_f, x_len):
                        Shredder_args = [x,k[idx_p][idx_H], a_stack, idx_p, idx_H]
                        Rho_E = Shredder_E(Shredder_args, sub_stack)

                        stack_x.append(Rho_E)
                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H)
                    save(stack_x, name, 'Rho_E', Master_path=Master_path)

    return

def Shredder_helicity(args, Master_path):
    [x, k, a_stack,idx_p, item_p, item_alpha, item_H] = args

    h = [-1,1]
    name_list = ['alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H) for item_h in h]
    gauge_stack = [load(name, 'Raw', Master_path) for name in name_list]

    for idx_k in range(len(k)):
        A_k = [mapper(x, gauge_stack[i][idx_k][0], gauge_stack[i][idx_k][1]) for i in [0,1]]

    Integrand = 1/ (8*pi**2) * k**2 * (A_k[0]**2 - A_k[1]**2)
    integrated = trapz(Integrand, x = k)

    Integrand_2 = 1/ (8*pi**2) * k**2 * (A_k[0]**2 + A_k[1]**2)
    integrated_2 = trapz(Integrand, x = k)

    norm_int = integrated/integrated_2
    
    return [integrated, norm_int]


def Helicity_spooler(p, H_m, alpha, k, x_len, a_stack, Master_path):
    h = [-1,1]
    stack_p = []
    for idx_p, item_p in enumerate(p):
        for idx_H, item_H in enumerate(H_m):
            stack_alpha = []
            for idx_alpha, item_alpha in enumerate(alpha):
                stack_x = []
                for x in np.linspace(t_i, t_f, x_len):
                    Shredder_args = [x, k[idx_p][idx_H], a_stack, idx_p, item_p, item_alpha, item_H]
                    Hel = Shredder_helicity(Shredder_args, Master_path)

                    stack_x.append(Hel)
                print(np.shape(stack_x))
                name = 'alpha:%.5f_p:%.3f_H_%.3f' % (item_alpha, item_p, item_H)
                save(stack_x, name, 'Helicity', Master_path=Master_path)

    return
