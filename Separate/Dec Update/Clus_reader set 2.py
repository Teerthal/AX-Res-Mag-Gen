######################################
##Changing the mappers and the start conditions
##for the Gauge solver
#####################################3

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
from decimal import Decimal
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

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
b_list_size = 2


steps = int(5.e5)
Gauge_steps = int(1e5)

N_start = 55.

steps = Gauge_steps*N_final/(N_final-N_start)
N_index = np.arange(0, int(Gauge_steps), 1)


cpu = 24
N_start_intervals = int(cpu*50)
delta_intervals = N_start_intervals

alpha_i = 0.
alpha_f = 20.
alpha_list_size = 5

phi_cmb = 15

# PARAMETERS FOR LOOPS
####################################

# f_list = np.array([0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.0001])
f_list = np.array([1e-2,1e-3])
####################################
index_b = np.arange(0, b_list_size, 1)
index_f = np.arange(0, len(f_list), 1)
index_alpha = np.arange(0, alpha_list_size, 1)


######################

#Setting epsilon limits for killing slow roll and non slow roll cases
eps_limit_ns = 1.5
eps_limit_slow = 1.
eps_overhead = 1.
sparce_res = 1000                #The number with which the scalar solution is split into for computing the mean
                                #\epsilon
##################

# Parameters for computation start point and wavemodes
####################################

delta = np.linspace(-3, 12., delta_intervals)

####################################

#End time for gauge solver
N_gauge_off = 0.

xi_samp_inter = 0.1     #For sampling xi around itme of interest to take 'non-oscillating' del phi for computing
                            #.....corresponding k

Data_set = '%.2f_%.2f_%.7f(final f)_%.0f_%.0f_%.0f'%(alpha_i,alpha_f, f_list[-1], Gauge_steps, len(f_list), delta_intervals)
Master_path = '/media/teerthal/Repo/Monodromy/Cluster/%s'%Data_set
#Master_path = '/home/teerthal/Repository/Gauge_Evolution/Monodromy/%s'%Data_set
#Master_path = '/work/Teerthal/Monodromy/%s'%Data_set

if os.path.exists(Master_path):
    print('Data Directory exists')
else:
    os.mkdir(Master_path)
    print('Data directory does not exist')
    script = os.path.abspath(__file__)
    code_copy = '%s/code.py' % Master_path
    copyfile(script, code_copy)


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

def load_or_exec(filename, directory, function, execute_arguments):
    command = 'F'#input('Rewrite %s in %s (T/F):'%(filename, directory))

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
        file = load(filename, directory)

    else:

        print('File does not exist or computing fresh set')
        file = function(execute_arguments)
        save(file, filename, directory)

    timer('Finish saving/loading')

    return np.array(file)


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
    alpha = np.array([0, 5, 10, 15, 20])*item_f
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

def Scalar_ODE(N, u, arg):
    b = arg[0]
    f = arg[1]
    eps = 1 / 2 * u[1] ** 2
    return np.array(
        [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p)])


def Scalar_Core(arg):

    b,f,eps_limit=arg

    Parameters = np.array([b, f])
    solver = ode(Scalar_ODE).set_integrator('vode').set_f_params(Parameters)
    Scalar_init = np.array([phi(N_initial), phi_prime(N_initial)])
    solver.set_initial_value(Scalar_init, N_initial)

    # Scalar ODE solving steps

    dt = (N_stop(b) - N_initial) / steps

    temp_1 = []
    temp_2 = []
    #print('Phi Prime limit:', sqrt(2*eps_limite_ns(item_b=b,item_f=f)))
    while solver.successful() and solver.y[1]**2./2.<=eps_limite_ns(item_b=b,item_f=f):
                                                        #solver.y[1]**2./2.<=eps_limite_ns(item_b=b,item_f=f):
                                                        # #solver.t <= N_stop(b):
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
        print(item_f)
        stack_b = []

        for item_b in b_list(item_f):
            stack_b.append(Scalar_Core([item_b, item_f, eps_limit_ns]))
        stack_f.append(stack_b)

    return stack_f

Guide_Buffer_name = 'Guide_Buffer'
Guide_Buffer_dir = 'Phi'

# Globally executed and stored Phi array

Guide_Buffer = load_or_exec(Guide_Buffer_name, Guide_Buffer_dir, Phi_Solver, f_list)
print('Shape of Guide Buffer  with test epsilon limit:', np.shape(Guide_Buffer))

sr_name = 'slow_roll_stack'
sr_dir = 'Phi'

#slow_roll_stack = np.array(Scalar_Core(0., 0.01, eps_limit_slow));
slow_roll_stack = load_or_exec(sr_name,sr_dir, Scalar_Core, [0., 0.01, eps_limit_slow])

print('slow roll stack shape',np.shape(slow_roll_stack))

def kill_eps_limit():
    stack_f = []
    for j in index_f:

        item_f = f_list[j]
        b = b_list(item_f)

        stack_b = []
        for i in index_b:
            item_b = b[i]

            del_phi = Guide_Buffer[j, i, 1]
            eps = del_phi**2./2.

            split_arr = []
            space = int(len(eps) / sparce_res)
            for i in np.arange(0, len(eps)-space, space):
                split_arr.append(eps[i:i+space])
            #print('manual split:',np.shape(split_arr))
            #sparced_eps = np.array_split(eps, sparce_res)
            sparced_eps = np.array(split_arr)
            #print('shape of split array', np.shape(sparced_eps))
            avg_eps = np.array([np.mean(i) for i in sparced_eps]);
            #print('average check', min(avg_eps), max(avg_eps))
            std_eps = np.array([np.max(i) for i in sparced_eps])-avg_eps
            #print('shape after avraging', np.shape(avg_eps))
            diff_arr = avg_eps - 1.
            min_idx = np.argmin(abs(diff_arr))

            kill_eps = avg_eps[min_idx] + std_eps[min_idx]
            print('eps condition:', avg_eps[min_idx], std_eps[min_idx], kill_eps)
            #kill_time = Guide_Buffer[0, j, i, 2][-1]

            stack_b.append(kill_eps)
        stack_f.append(stack_b)

    return np.array(stack_f)


kill_matrix = kill_eps_limit(); print('shape of epsilon kill matrix', np.shape(kill_matrix))

def Scalar_Core_Pri(args):
    b, f, eps_limit = args
    Parameters = np.array([b, f])
    solver = ode(Scalar_ODE).set_integrator('vode').set_f_params(Parameters)
    Scalar_init = np.array([phi(N_initial), phi_prime(N_initial)])
    solver.set_initial_value(Scalar_init, N_initial)

    # Scalar ODE solving steps

    dt = (N_stop(b) - N_initial) / steps

    temp_1 = []
    temp_2 = []
    #print('Phi Prime limit:', sqrt(2*eps_limite_ns(item_b=b,item_f=f)))
    while solver.successful() and solver.y[1]**2./2.<=eps_limit:
                                                        #solver.y[1]**2./2.<=eps_limite_ns(item_b=b,item_f=f):
                                                        # #solver.t <= N_stop(b):
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


def Phi_Solver_Pri(f):
    stack_f = []

    for j in index_f:
        item_f = f_list[j]
        b = b_list(item_f)

        stack_b = []

        for i in index_b:
            item_b = b[i]

            eps_limit = kill_matrix[j,i]
            stack_b.append(Scalar_Core_Pri([item_b, item_f, eps_limit]))
        stack_f.append(stack_b)

    return stack_f


# Globally executed and stored Phi array

Phi_Buffer_name = 'Phi'
Phi_Buffer_dir = 'Phi'
Buffer = load_or_exec(Phi_Buffer_name, Phi_Buffer_dir, Phi_Solver_Pri, f_list)
print('Shape of Primary Buffer  with test epsilon limit:', np.shape(Buffer))

def Phi(index_N, index_b, index_f):
    phi_solved = Buffer[index_f, index_b, 0]
    #N = Buffer[0, index_f, index_b, 2]
    #index_N = int(abs(N / max(N) * steps - 1))

    return phi_solved[index_N]

def Phi_Prime(N, index_b, index_f):

    phi_prime_solved = Buffer[index_f, index_b, 1]
    N_ns = np.array(Buffer[index_f, index_b, 2])
    index_N = int(N / N_ns[-1] * len(N_ns))-1
    return phi_prime_solved[index_N]

def epsilon(N, index_b, index_f):
    return (Phi_Prime(N, index_b, index_f)) ** 2 / 2.

def H(N):
    #Epsilon = phi_prime(N) ** 2 / 2
    #H =  abs(H_initial * sqrt((3. - Epsilon) / (3. - Epsilon) * (phi(N) / phi_initial) ** 2))
    H_constant = H_initial
    return H_constant

def N_k(delta, index_f, index_b):
    kill_time = Buffer[index_f, index_b, 2][-1]
    N_kill = kill_time
    #N_k = delta + N_start
    N_k = N_kill - delta
    return N_k

def N_k_sr(delta):
    N_kill = slow_roll_stack[2,-1]
    #N_k = delta + N_start
    N_k = N_kill - delta
    return N_k

def k1(delta, index_f, index_b):
    #phi_prime_solved = slow_roll_stack[1]
    phi_prime_solved= Buffer[index_f, index_b, 1]
    xi = alpha_list(f_list[index_f])*phi_prime_solved/f_list[index_f]
    N_k_delta = N_k(delta, index_f, index_b)
    #N_kill = slow_roll_stack[2,-1]
    kill_time = Buffer[index_f, index_b, 2][-1]
    N_kill = kill_time

    scalar_steps = len(phi_prime_solved)
    index = int(N_k_delta/N_kill*scalar_steps)
    index_1 = int((N_k_delta-xi_samp_inter)/N_kill*scalar_steps)
    index_2 = int((N_k_delta+xi_samp_inter)/N_kill*scalar_steps)

    xi_N_k = abs(xi[index])
    xi_N_k_averaged = np.mean(abs(xi[index_1:index_2]))

    sample_std = np.std(abs(xi[index_1:index_2]))#; print('Xi sampling deviation:', sample_std)

    xi_N_k_smoothed = max(abs(xi[index_1:index_2]))-sample_std

    #k = a(N_k_delta)*abs(H(N_k_delta))*xi_N_k #*exp(7)
    #k = a(N_k_delta) * abs(H(N_k_delta)) * xi_N_k_averaged  # *exp(7)
    k = a(N_k_delta) * abs(H(N_k_delta)) * xi_N_k_smoothed  # *exp(7)

    return k


def k(delta, index_f, index_b):
    #N_k_delta = N_k(delta, index_f, index_b)
    #N_kill = Buffer[0, index_f, index_b, 2][-1]
    #xi = abs(alpha_list(f_list[index_f]) * phi_prime(N_k_delta) / f_list[index_f])
    #k = a(N_final) * H(N_final) * exp(-delta)

    k = H(N_final) * exp(-delta)
    return k


#k_test = functools.partial(k, index_f=0, index_b=0);delta_test =np.linspace(4.,5.,100)
#plt.plot(delta_test, np.array([*map(k_test, delta_test)]));plt.show()


def k_sr(delta):
    #phi_prime_solved = slow_roll_stack[1]
    #phi_prime_solved = Buffer[0, index_f, index_b, 1]
    #xi = alpha_list(f_list[0]) * phi_prime_solved /f_list[0]
    #N_k_delta = N_k_sr(delta)
    #N_kill = slow_roll_stack[2,-1]
    #N_kill = Buffer[0, index_f, index_b, 2][-1]
    #scalar_steps = len(phi_prime_solved)
    #index = int(N_k_delta / N_kill * scalar_steps)
    #xi_N_k = abs(xi[index])
    #k = a(N_k_delta) * abs(H(N_k_delta)) * xi_N_k #* exp(7)

    k = H(N_final) * exp(-delta)

    return k

end = time.time()

def a_corr(N, index_f, index_b):
    N_phi_solved = Buffer[index_f, index_b, 2]
    N_kill = N_phi_solved[-1]
    return exp(N - N_kill)

def a_corr_sr(N):
    N_kill = slow_roll_stack[2][-1]
    return exp(N - N_kill)

def mapper1():
    stack_j = []
    for j in index_f:
        stack_i = []
        for i in index_b:

            N_kill = Buffer[index_f, index_b, 2][-1];print('end scan',N_kill-N_gauge_off)

            N_cons = Buffer[j, i, 2]
            phi_cons = Buffer[j, i, 0]
            phi_prime_cons = Buffer[j, i, 1]

            reduc_idx = int(45/N_kill*len(N_cons))

            N_reduc = N_cons[reduc_idx:]
            phi_reduc = phi_cons[reduc_idx:]
            phi_prime_reduc = phi_prime_cons[reduc_idx:]

            gauge_N = np.linspace(N_start, N_kill - N_gauge_off, Gauge_steps)

            bottom_stack=[]
            for N in gauge_N:

                diff_arr = abs(N_reduc-N)

                #interes_idx = [(x,y) for x,y in enumerate(diff_arr)]
                min_idx = np.argmin(diff_arr)

                bottom_stack.append(np.array([phi_reduc[min_idx], phi_prime_reduc[min_idx], N]))
            stack_i.append(bottom_stack)
        stack_j.append(stack_i)

    return np.array(stack_j)


def mapper(arg):
    stack_j = []
    for j in index_f:
        stack_i = []
        for i in index_b:

            N_cons = Buffer[j, i, 2]
            phi_cons = Buffer[j, i, 0]
            phi_prime_cons = Buffer[j, i, 1]

            #N_kill = Buffer[0, index_f, index_b, 2][-1];print('end scan',N_kill-N_gauge_off)
            N_kill = N_cons[-1];print('end scan',N_kill-N_gauge_off)
            steps = len(N_cons)

            gauge_N = np.linspace(N_start, N_kill-N_gauge_off, Gauge_steps)

            bottom_stack=[]
            for N in gauge_N:


                idx = abs(int((N-N_initial) /(N_kill - N_initial) * steps - 1))

                bottom_stack.append(np.array([phi_cons[idx], phi_prime_cons[idx], N]))
            stack_i.append(bottom_stack)
        stack_j.append(stack_i)

    return np.array(stack_j)

remapped_scalar_name = 'Remapped_scalar'
remapped_scalar_dir = 'Phi'

remapped_scalar = load_or_exec(remapped_scalar_name, remapped_scalar_dir, mapper, [])
print(np.shape(remapped_scalar))



def Scalar_plot():
    #plt.subplot(211)
    plt.plot(slow_roll_stack[2], slow_roll_stack[1]**2/2,label='slow roll')
    plt.plot([N_initial, N_final+2], [1, 1])

    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)[0]
        for i in index_b:
            b_i = b[i]

            N_phi_solved = Buffer[j, i, 2]
            phi_solved = Buffer[j, i, 0]
            phi_prime_solved = Buffer[j, i, 1]

            # Growth of the physical wavenumber v growth of xi
            #a_map = np.array([*map(a, N)])
            #H_map = np.array([*map(H, N)])
            #plt.plot(N, 15.*abs(phi_prime_solved))
            #plt.plot(N, k(5.)/(a_map*H_map))
            #plt.show()

            #N_kill = Buffer[index_f, index_b, 2][-1]

            #plt.subplot(211)
            #plt.semilogy(N_phi_solved, phi_solved)
            # plt.semilogy(slow_roll_stack[2], slow_roll_stack[0])
            plt.plot(N_phi_solved, phi_prime_solved ** 2 / 2, label='bf:%.4f'%(b_i*f_j) )
            plt.plot(remapped_scalar[j,i,:,2], remapped_scalar[j,i,:,1]**2/2, linestyle=':', label='Interp')
            #plt.plot(N_phi_solved, abs(phi_prime_solved)*alpha/f_j)
            k_test = functools.partial(k, index_f=0, index_b=0);delta_test =delta
            #plt.plot(delta_test, np.array([*map(k_test, delta_test)])/a(N_final-delta_test))

            plt.legend()
            plt.ylabel(r'$\epsilon$')
            plt.xlabel('N')
            #plt.subplot(212)
            #plt.plot(N_phi_solved, phi_solved)


            #plt.semilogy(slow_roll_stack[2], slow_roll_stack[1]**2/2)
    plt.show()
    return

Scalar_plot()


def xi_plot():

    index_f = [0]
    index_b = [-1]
    index_alpha = [-1]

    for j in index_f:
        item_f = f_list[j]
        alpha = alpha_list(item_f)
        b = b_list(item_f)

        stack_alpha = []
        for z in index_alpha:
            item_alpha = alpha[z]

            N_slow = slow_roll_stack[2]
            N_kill = N_slow[-1];
            phi_prime_sr = slow_roll_stack[1]
            xi_sr = abs(phi_prime_sr*item_alpha/item_f)

            a_del_sr = np.array([*map(a_corr_sr, (N_kill - delta))])

            k_del_sr = np.array([*map(k_sr, delta)])

            #plt.plot(N_kill - delta, k_del_sr / (a_del_sr * H_initial), marker='x', linestyle='')
            #plt.plot(N_slow, xi_sr, label = 'sr')

            stack_b = []
            for i in index_b:
                item_b = b[i]

                N_cons = Buffer[j, i, 2]
                N_kill = N_cons[-1];
                phi_prime_solved = Buffer[j, i, 1]


                N_remap = remapped_scalar[j, i, :, 2]
                dphi_remap = remapped_scalar[j, i, :, 1]

                gauge_N = N_remap

                #print('end scan', N_kill - N_gauge_off)

                a_map_par = functools.partial(a_corr, index_b=i, index_f=j)
                a_map_del = np.array([*map(a_map_par, (N_kill-delta))])
                a_full = np.array([*map(functools.partial(a_map_par), gauge_N)])
                #H_full = np.array([*map(functools.partial(H), gauge_N)])

                k_map_par = functools.partial(k, index_f = j, index_b=i)
                k_map_del = np.array([*map(k_map_par, delta)])

                #plt.plot(N_kill-delta, k_map_del / (a_map_del * H_initial), marker='x', linestlye = '')

                #plt.plot(N_cons, abs(phi_prime_solved)*item_alpha/item_f, label = 'bf:%.4f' %(item_f*item_b))


                index_delta = [0,-1]
                for l in index_delta:
                    delta_l = delta[l]

                    xi = k(delta_l,j,i)/a_full*abs(dphi_remap)
                    kin = k(delta_l,j,i)**2/a_full**2

                    plt.semilogy(gauge_N,xi)
                    plt.semilogy(gauge_N, kin, label = r'$\delta:%.2f$'%delta_l)

    plt.legend()
    return plt.show()

xi_plot()

warning_collect = []
def phi_prime_mapped(N, i, j):
    N_remapped = remapped_scalar[j, i, :, 2]
    N_kill = remapped_scalar[j, i, -1, 2]#; print('kill time for the gauge solver mapper:', N_kill)
    idx = np.argmin(abs(N-N_remapped))
    mapped = remapped_scalar[j,i,idx,1]
    return mapped


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
         -u[1] - (k_delta ** 2 / (a_corr(N, index_f, index_b)*H(N))**2
                  - k_delta / (a_corr(N, index_f, index_b)*H(N))
                  * h * alpha / f_list[index_f] * Phi_Prime) * u[0]])


def core(delta, h, alpha, index_b, index_f):

    #print('core parameter test:', f_list[index_f]*(b_list(f_list[index_f])[index_b]))

    N_kill =Buffer[index_f, index_b, 2][-1]#;print('kill time', N_kill)
    N_start = N_kill - N_gauge_off - delta - 5.

    Parameters = np.array([N_k(delta, index_f, index_b), h, alpha, index_b, index_f, phi_prime_mapped, k(delta, index_f, index_b)])

    r = ode(ODE).set_integrator('zvode').set_f_params(Parameters)

    # Initial Condition
    t_initial = -1
    A_initial = exp(-1j * k(delta, index_f, index_b) * t_initial)
    A_prime_initial = -1j * A_initial * k(delta, index_f, index_b) / (a_corr(N_start, index_f, index_b)*H(N_start))

    init = np.array([A_initial, A_prime_initial]);

    r.set_initial_value(init, N_start)

    u = []
    t = []

    dt = (N_kill - N_gauge_off - N_start) / Gauge_steps

    N_end = N_kill-N_gauge_off

    while r.successful() and N_start <= r.t <= N_end:
        r.integrate(N_end, step=True)
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

    #N = np.linspace(N_start, N_kill - N_gauge_off, Gauge_steps)

    #Normalising

    N_k_del = N_k(delta, index_f, index_b)
    #index = int(N_k_del/(N_final-2.)*Gauge_steps)
    index = abs(int(abs(N_k_del - N_start) / abs(N_kill - N_gauge_off - N_start) * steps - 1))

    a_par = functools.partial(a_corr, index_f=index_f, index_b=index_b)

    a_map_Gauge = np.array([*map(a_par, N)])

    H_map_Gauge = np.array([*map(H, N)])

    norm_A = []
    norm_A[:index] = A[:index]**2-1
    norm_A[index:] = A[index:]**2

    norm_del_A = []
    norm_del_A[:index] = (A_prime*a_map_Gauge*H_map_Gauge/k(delta, index_f, index_b))[:index]**2-1
    norm_del_A[index:] = (A_prime*a_map_Gauge*H_map_Gauge/k(delta, index_f, index_b))[index:]**2

    norm_A = np.array(norm_A)
    norm_del_A = np.array(norm_del_A)

    norm_A[norm_A < 0.1] = 0.
    norm_del_A[norm_del_A < 0.1] = 0.
    print(delta, len(N))
    return [N, A, A_prime, norm_A, norm_del_A]


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

                #print(r'$\alpha/f for the N.S comp:$',(item_alpha/item_f))

                stack_h = []
                for h in [-1,1]:

                    pool = Pool(cpu)

                    if __name__ == '__main__':
                        stack = np.array(pool.map(functools.partial(core, h=h, alpha=item_alpha,
                                                                    index_b=i, index_f=j), delta))
                        pool.close()

                    stack_h.append(stack)
                    print(np.shape(stack))
                    plt.semilogy(stack[0][0], stack[0][1], label='h%s'%h)
                plt.legend()
                plt.show()
                save(stack_h, 'alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')
                print('Gauge solution non slow roll', np.shape(stack_h))

    return [0]

raw_dir = 'Raw'

sub_path = '%s/%s' % (Master_path, raw_dir)
rewrite_RAW = 'F'#input('Rewrite Raw data directory(T/F):')

if os.path.exists(sub_path):
    print('Sub Directory exists')
else:
    os.mkdir(sub_path)
    print('Sub directory does not exist')
    print('Builing new Subdirectory %s' % raw_dir)

if rewrite_RAW == 'T':
    execute(delta)
    timer('Finish calculating and saving')

else:
    timer('Finish loading')

print('Instances for warning X ', (np.shape(warning_collect)))

####Gauge field solution for the slow roll case
def mapper_sr1():
    N_kill = slow_roll_stack[2, -1]

    gauge_N = np.linspace(N_start, N_kill - N_gauge_off, Gauge_steps)
    N_cons = slow_roll_stack[2]
    phi_cons = slow_roll_stack[0]
    phi_prime_cons = slow_roll_stack[1]

    reduc_idx = int(45 / 70 * len(N_cons))

    N_reduc = N_cons[reduc_idx:]
    phi_reduc = phi_cons[reduc_idx:]
    phi_prime_reduc = phi_prime_cons[reduc_idx:]

    bottom_stack = []
    for N in gauge_N:
        diff_arr = abs(N_reduc - N)

        # interes_idx = [(x,y) for x,y in enumerate(diff_arr)]
        min_idx = np.argmin(diff_arr)
        idx = abs(int((N-N_initial) / (N_kill-N_initial) * steps - 1))
        bottom_stack.append(np.array([phi_reduc[min_idx], phi_prime_reduc[min_idx], N]))

    return np.array(bottom_stack)

def mapper_sr(arg):
    N_cons = slow_roll_stack[2]
    phi_cons = slow_roll_stack[0]
    phi_prime_cons = slow_roll_stack[1]

    # N_kill =Buffer[0, index_f, index_b, 2][-1];print('end scan',N_kill-N_gauge_off)
    N_kill = N_cons[-1]  # ;print('end scan',N_kill-N_gauge_off)
    steps = len(N_cons)

    gauge_N = np.linspace(N_start, N_kill - N_gauge_off, Gauge_steps)

    bottom_stack = []
    for N in gauge_N:
        idx = abs(int((N-N_initial) / (N_kill-N_initial) * steps - 1))

        bottom_stack.append(np.array([phi_cons[idx], phi_prime_cons[idx], N]))

    return np.array(bottom_stack)

remapped_scalar_sr_name = 'Remapped_SR'
remapped_scalar_sr_dir = 'Phi'

remapped_scalar_sr = load_or_exec(remapped_scalar_sr_name, remapped_scalar_sr_dir, mapper_sr, [])
print('Shape of remapped slow roll scalar sol',np.shape(remapped_scalar_sr))


def phi_prime_mapped_sr(N,index_f,index_b):
    N_remapped = remapped_scalar_sr[:, 2]
    idx = np.argmin(abs(N-N_remapped))
    mapped = remapped_scalar_sr[idx, 1]
    return mapped


def ODE_sr(N, u, arg):
    N_k = arg[0]
    h = arg[1]
    alpha = arg[2]
    index_b = arg[3]
    index_f = arg[4]
    Phi_Prime_sr = arg[5](N, index_b, index_f)
    k_delta = arg[6]

    return np.array(
        [u[1],
         -u[1] - (k_delta ** 2 / (a_corr_sr(N)*H(N))**2
                  - k_delta / (a_corr_sr(N)*H(N))
                  * h * alpha / f_list[index_f] * Phi_Prime_sr) * u[0]])


def core_sr(delta, h, alpha, index_b, index_f):

    N_final=slow_roll_stack[2,-1]
    N_start = N_final - N_gauge_off - delta - 5.

    Parameters = np.array([N_k_sr(delta), h, alpha, index_b, index_f, phi_prime_mapped_sr, k_sr(delta)])

    r = ode(ODE_sr).set_integrator('zvode').set_f_params(Parameters)

    # Initial Condition
    t_initial = -1
    A_initial = exp(-1j * k_sr(delta) * t_initial)
    A_prime_initial = -1j * A_initial * k_sr(delta) / (a_corr_sr(N_start)*H(N_start))

    init = np.array([A_initial, A_prime_initial]);

    r.set_initial_value(init, N_start)

    u = []
    t = []

    dt = (N_final - N_gauge_off - N_start) / Gauge_steps
    N_end = N_final - N_gauge_off

    while r.successful() and r.t <= N_end:
        r.integrate(N_end, step=True)
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

    N_k_del = N_k_sr(delta)
    index = abs(int(abs(N_k_del - N_start) / abs(N_final - N_gauge_off - N_start) * steps - 1))

    a_par = functools.partial(a_corr, index_f=index_f, index_b=index_b)
    a_map_Gauge = np.array([*map(a_par, N)])
    H_map_Gauge = np.array([*map(H, N)])

    norm_A = []
    norm_A[:index] = A[:index]**2-1
    norm_A[index:] = A[index:]**2

    norm_del_A = []
    norm_del_A[:index] = (A_prime*a_map_Gauge*H_map_Gauge/k_sr(delta))[:index]**2-1
    norm_del_A[index:] = (A_prime*a_map_Gauge*H_map_Gauge/k_sr(delta))[index:]**2

    norm_A = np.array(norm_A)
    norm_del_A = np.array(norm_del_A)

    norm_A[norm_A < 0.1] = 0.
    norm_del_A[norm_del_A < 0.1] = 0.

    return [N, A, A_prime, norm_A, norm_del_A]


def execute_sr(delta):
    alpha = alpha_list(f_list[0])

    stack_alpha = []
    for z in index_alpha:
        item_alpha = alpha[z]
        print(r'$\alpha/f$ for the slow roll comp:',(item_alpha/f_list[0]))
        stack_h = []
        for h in [-1]:

            pool = Pool(cpu)

            if __name__ == '__main__':
                stack = np.array(pool.map(functools.partial(core_sr, h=h, alpha=item_alpha,
                                                            index_b=0, index_f=0), delta))
                pool.close()

            stack_h.append(stack)
            # print(np.shape(stack))
            # save(stack_h, 'Slow roll gauge', 'Raw')

            print('Gauge solution for sr shape', np.shape(stack_h))
        save(stack_h, 'slow_roll_alpha_f:%.4f'%(item_alpha/f_list[0]), 'Raw')

    return np.array(stack_h)


rewrite_RAW_sr = 'F'#input('Rewrite Raw SR file:')

if rewrite_RAW_sr == 'T':
    execute_sr(delta)
    timer('Finish calculating and saving')

else:
    timer('Finish loading')

def Gauge_curves():
    for l in [index_f[-1]]:
        item_f = f_list[l]
        b = b_list(item_f)
        alpha = alpha_list(item_f)

        for j in [index_b[0]]:
            item_b = b[j]

            k_map = np.array([*map(functools.partial(k, index_f=l, index_b=j),delta)])

            for z in [5,10]:#np.arange(0,len(alpha),5):
                item_alpha = alpha[z]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')
                #gauge_sr = load('slow_roll_alpha_f:%.4f'%(item_alpha/f_list[0]), 'Raw')
                #plt.loglog(k_map, gauge_sr[0,:,1,-1], label='s bf:%.4f' %(item_b*item_f));plt.loglog(k_map,stack[0,:,1,-1]);plt.show()
                for i in [-200]:#np.arange(0, delta_intervals, 500):
                    #plt.subplot(211)
                    #plt.semilogy(gauge_sr[0, i, 0], gauge_sr[0, i, 1], label='sr', linestyle=':')

                    #plt.subplot(212)
                    #plt.semilogy(gauge_sr[0,i,0], gauge_sr[0,i,2], label='sr',linestyle=':')

                    k_i = k_map[i]

                    plt.subplot(211)
                    plt.ylabel(r'$|\mathcal{A}|$')
                    plt.semilogy(stack[0, i, 0], stack[0, i, 1], label='k:%s,bf:%s' % (k_i, item_b * item_f))
                    plt.semilogy(stack[1, i, 0], stack[1, i, 1], label='k:%s ,bf:%s' % (k_i, item_b * item_f), linestyle = '--')
                    plt.legend()
                    plt.subplot(212)
                    #plt.ylabel(r'$|\frac{d\mathcal{A}}{dx}|$')

                    A_prime = (stack[0, i, 2])
                    plt.semilogy(stack[0, i, 0], A_prime, label='k:%s' % k_i)
                    #plt.legend()
    plt.show()

    return

#Gauge_curves()


def Gauge_eps_curves():

    gs = gridspec.GridSpec(3,1)
    ax1 = plt.subplot(gs[:-1, :])
    ax2 = plt.subplot(gs[-1, :], sharex = ax1)
    
    for l in [0]:
        item_f = f_list[l]
        b = b_list(item_f)
        alpha = alpha_list(item_f)
        
        colors_b = cm.rainbow(np.linspace(0,1,len(index_b)))
        for j in index_b:
            item_b = b[j]
            
            k_map = np.array([*map(functools.partial(k, index_f=l, index_b=j),delta)])

            for z in [-1]:
                item_alpha = alpha[z]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')
                gauge_sr = load('slow_roll_alpha_f:%.4f'%(item_alpha/f_list[0]), 'Raw')
                #plt.loglog(k_map, gauge_sr[0,:,1,-1], label='s bf:%.4f' %(item_b*item_f));plt.loglog(k_map,stack[0,:,1,-1]);plt.show()
                for i in np.round(np.linspace(0, len(k_map)-1, 10)).astype(int):
                    print(i)
                    
                    ax1.semilogy(gauge_sr[0, i, 0], gauge_sr[0, i, 2], color = 'grey')#, linewidth = 3., alpha = 0.25, )
                    ax1.semilogy(gauge_sr[1, i, 0], gauge_sr[1, i, 2], color = 'grey', linestyle = '--')#, linewidth = 3., alpha = 0.25, )
                    k_i = k_map[i]
                    ax1.set_ylabel(r'$|\sqrt{2k}\mathcal{A}_\pm|$')
                    ax1.semilogy(stack[0, i, 0], stack[0, i, 2], color = colors_b[j])#, label=r'$k/a_f:%s ,bf:%s$'% (k_i/a(stack[0,i,0][-1]), item_b * item_f))
                    ax1.semilogy(stack[1, i, 0], stack[1, i, 2], linestyle = '--', color = colors_b[j],label=r'$k/a_f:%.2f \quad bf:%.2f$'%(k_i/a(stack[0,i,0][-1]), item_b * item_f))
                    #ax1.set_ylabel(r'$|\frac{d\mathcal{A}}{dx}|$')
                    #plt.legend()

                    phi_N = remapped_scalar[l, j, :, 2]
                    dphi = remapped_scalar[l, j, :, 1]
                    ax2.plot(phi_N, abs(dphi), color = colors_b[j], linewidth = 0.75)#, label = r'$f:%.4f \quad  bf:%.2f$'%(item_f, item_f*item_b))
                    phi_sr_N = remapped_scalar_sr[:,2]
                    dphi_sr = remapped_scalar_sr[:,1]
                    ax2.plot(phi_sr_N, abs(dphi_sr), color = 'grey')#, linewidth = 5, alpha = 0.25)
                    ax2.set_ylabel(r'$|d\phi/dN|$')
        
        plt.show()

    return

#Gauge_eps_curves()


def gen_map(var, var_arr , target_arr):

    diff_arr = abs(var_arr-var)
    min_idx = np.argmin(diff_arr)

    out = target_arr[min_idx]

    return out


def Spectral_evo_curves():
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[:-1, :])
    ax2 = plt.subplot(gs[-1, :], sharex=ax1)

    N_kill_sr = slow_roll_stack[2][-1]

    k_map_sr = np.array([*map(k_sr, delta)]);

    for l in [0]:
        item_f = f_list[l]
        b = b_list(item_f)
        alpha = alpha_list(item_f)

        colors_b = cm.rainbow(np.linspace(0, 1, len(index_b)))
        for j in [index_b[-1]]:
            item_b = b[j]

            N_kill = Buffer[l, j, 2][-1]
            k_map = np.array([*map(functools.partial(k, index_f=l, index_b=j), delta)])

            for z in [-1]:
                item_alpha = alpha[z]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha / item_f, item_b, item_f), 'Raw')
                gauge_sr = load('slow_roll_alpha_f:%.4f' % (item_alpha / f_list[0]), 'Raw')
                # plt.loglog(k_map, gauge_sr[0,:,1,-1], label='s bf:%.4f' %(item_b*item_f));plt.loglog(k_map,stack[0,:,1,-1]);plt.show()
                for i in np.linspace(68, N_kill,5):
                    print(i)

                    A_map_0 = [gen_map(i, stack[0,idx,0], stack[0,idx,1]) for idx in np.arange(0,delta_intervals,1)]
                    A_map_1 = [gen_map(i, stack[1, idx, 0], stack[1, idx, 1]) for idx in np.arange(0, delta_intervals, 1)]

                    delA_map_0 = [gen_map(i, stack[0, idx, 0], stack[0, idx, 4]) for idx in
                               np.arange(0, delta_intervals, 1)]
                    delA_map_1 = [gen_map(i, stack[1, idx, 0], stack[1, idx, 4]) for idx in
                               np.arange(0, delta_intervals, 1)]

                    A_map_sr_0 = [gen_map(i, gauge_sr[0,idx,0], gauge_sr[0,idx,1]) for idx in np.arange(0,delta_intervals,1)]
                    A_map_sr_1 = [gen_map(i, gauge_sr[1, idx, 0], gauge_sr[1, idx, 1]) for idx in
                                  np.arange(0, delta_intervals, 1)]

                    delA_map_sr_0 = [gen_map(i, gauge_sr[0, idx, 0], gauge_sr[0, idx, 4]) for idx in
                                  np.arange(0, delta_intervals, 1)]
                    delA_map_sr_1 = [gen_map(i, gauge_sr[1, idx, 0], gauge_sr[1, idx, 4]) for idx in
                                  np.arange(0, delta_intervals, 1)]

                    ax1.semilogy(k_map_sr, delA_map_sr_0*k_map_sr**2,
                                 color='grey')  # , linewidth = 3., alpha = 0.25, )
                    ax1.semilogy(k_map_sr, delA_map_sr_1*k_map_sr**2, color='grey',
                                 linestyle='--')  # , linewidth = 3., alpha = 0.25, )

                    ax1.set_ylabel(r'$|\sqrt{2k}\mathcal{A}_\pm|$')
                    ax1.semilogy(k_map, delA_map_0*k_map**2, color=colors_b[
                        j])  # , label=r'$k/a_f:%s ,bf:%s$'% (k_i/a(stack[0,i,0][-1]), item_b * item_f))
                    ax1.semilogy(k_map, delA_map_1*k_map**2, linestyle='--', color=colors_b[j],
                                 label=r'$bf:%.2f$' % (item_b * item_f))
                    # ax1.set_ylabel(r'$|\frac{d\mathcal{A}}{dx}|$')
                    # plt.legend()
                    ax1.set_xscale('Log')
                    phi_N = remapped_scalar[l, j, :, 2]
                    dphi = remapped_scalar[l, j, :, 1]
                    ax2.plot(phi_N, abs(dphi), color=colors_b[j],
                             linewidth=0.75)  # , label = r'$f:%.4f \quad  bf:%.2f$'%(item_f, item_f*item_b))
                    phi_sr_N = remapped_scalar_sr[:, 2]
                    dphi_sr = remapped_scalar_sr[:, 1]
                    ax2.plot(phi_sr_N, abs(dphi_sr), color='grey')  # , linewidth = 5, alpha = 0.25)
                    ax2.set_ylabel(r'$|d\phi/dN|$')

        plt.show()

    return


#Spectral_evo_curves()

def Gauge_curves_A():
    for l in index_f:
        item_f = f_list[l]
        b = b_list(item_f)
        alpha = alpha_list(item_f)
        #plt.plot([],[],label='sr', linestyle =':')
        h_list = [-1, 1]
        for j in index_b:
            item_b = b[j]

            k_map = np.array([*map(functools.partial(k, index_f=l, index_b=j),delta)])

            for z in index_alpha:
                item_alpha = alpha[z]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')
                #gauge_sr = load('slow_roll_alpha:%.4f'%item_alpha, 'Raw')

                h_style = ['-', '--']
                [plt.plot([],[],color ='k', linestyle=h_style[i], label = r'$h:%s$'%h_list[i]) for i in range(len(h_style))]
                colors_delta = cm.rainbow(np.linspace(0,1,N_start_intervals))
                for i in np.arange(0, delta_intervals, 1):

                    #plt.semilogy(gauge_sr[0, i, 0], gauge_sr[0, i, 1], linestyle=':')

                    k_i = k_map[i]
                    N_end = stack[0,0,0,-1]
                    plt.plot([],[], color=colors_delta[i], label = r'$k/a_f$:%.2e'%(Decimal(k_i/a(N_end))))

                    for idx_h, item_h in enumerate(h_list):

                        plt.semilogy(stack[idx_h, i, 0], stack[idx_h, i, 1], color=colors_delta[i], linestyle = h_style[idx_h])

        plt.ylabel(r'$|\mathcal{A}|$')
        plt.xlabel('N')
        plt.legend()

        plt.show()

    return

#Gauge_curves_A()

def Asymptotes():
    stack_f = []
    for j in [-1]:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])

        stack_z = []
        for z in index_alpha:
            alpha_z = alpha[z]
            alpha_sr_z = alpha_sr[z]

            gauge_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr_z/f_list[0]), 'Raw')
            k_map_sr = np.array([*map(k_sr, delta)]);
            #print(k_map_sr)

            A_sr = [gauge_sr[0][i][1][-1] for i in np.arange(0,delta_intervals)]
            del_A_sr = [gauge_sr[0][i][2][-1] for i in np.arange(0,delta_intervals)]
            norm_A_sr = [gauge_sr[0][i][3][-1] for i in np.arange(0,delta_intervals)]
            asymptotes_sr = np.array([k_map_sr, A_sr, del_A_sr, norm_A_sr])
            save(asymptotes_sr, 'slow_roll_alpha_f:%.4f' % (alpha_sr_z/f_list[0]), 'Asymptotes')
	
            stack_j = []
            for i in index_b:
                b_i = b[i]

                #k_map = np.array([*map(functools.partial(k, index_f=j, index_b=i), delta)]);
                #print(k_map)

                #stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Raw')

                #A = [stack[0][i][1][-1] for i in np.arange(0, delta_intervals)]
                #del_A = [stack[0][i][2][-1] for i in np.arange(0, delta_intervals)]
                #norm_A = [stack[0][i][3][-1] for i in np.arange(0, delta_intervals)]

                #asymptotes = np.array([k_map, A, del_A, norm_A])
		
                #save(asymptotes, 'alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Asymptotes')

    return

Asym_dir = 'Asymptotes'
sub_path = '%s/%s' % (Master_path, Asym_dir)

rewrite_Asymptote = input('Rewrite Asymptote data directory(T/F):')

if os.path.exists(sub_path):
    print('Sub Directory exists')
else:
    os.mkdir(sub_path)
    print('Sub directory does not exist')
    print('Builing new Subdirectory %s' % Asym_dir)

if rewrite_Asymptote == 'T':
    Asymptotes()
    timer('Finish calculating and saving')

else:
    timer('Finish loading')



def Asymptotic_plot():
    for j in [0]:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])
        #plt.subplot(2,1,1+j)

        targ_alpha_idxs = index_alpha

        colors_alpha = cm.rainbow(np.linspace(0,1,len(index_alpha)))
        for z_idx, z in enumerate(targ_alpha_idxs):
            alpha_z = alpha[z]
            alpha_sr_z = alpha_sr[z]
            print(alpha_z/f_j)
            stack_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr_z/f_list[0]), 'Asymptotes')
            stack_sr_0 = load('slow_roll_alpha_f:%.4f' % (0./f_list[0]), 'Asymptotes')
            A_zero_sr, del_A_zero_sr = [stack_sr_0[1], stack_sr_0[2]]

            plt.loglog(stack_sr[0], stack_sr[1]-A_zero_sr, color = 'grey', linewidth = 5, alpha = 0.35)

            for i in [0]:#index_b:
                b_i = b[i]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Asymptotes')

                stack_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0./f_j, b_i, f_j), 'Asymptotes')
                A_zero, del_A_zero = [stack_0[1], stack_0[2]]
                plt.subplot(211)
                plt.loglog(stack[0], stack[1]-A_zero, linestyle = ['-','--'][i], color = colors_alpha[z_idx])#,label=r'f:%.4f  bf:%.2f' % (b_i * f_j))
                N_kill = Buffer[j, i, 2][-1]
                plt.subplot(212)
                plt.loglog(stack[0], stack[2]-del_A_zero, linestyle = ['-','--'][i], color = colors_alpha[z_idx])
                #plt.legend()
    #plt.axes().set_aspect(2)
    plt.xlabel(r'$k/a_{end}H_{end}$')
    plt.ylabel(r'$\sqrt{2k}\mathcal{A}_\pm(|k,N_{end}|)$')
    return plt.show()


Asymptotic_plot()


def B_0():
    H_inf = 3.6e-5
    r = 0.1
    rho_phi = 1 / (3 * H_inf ** 2)
    dVdphi = 3 * sqrt(r) / H_inf

    Mpl = 2.435 * 1e18  ##Mpl in GeV
    GeV2 = 6.8 * 1e20  ##Gev^2 in Gauss

    T_R = 1e-5  ##in Mpl

    for j, item_f in enumerate(index_f):
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])
        # plt.subplot(2,1,1+j)

        targ_alpha_idxs = index_alpha

        colors_alpha = cm.rainbow(np.linspace(0, 1, len(index_alpha)))
        for z_idx, z in enumerate(targ_alpha_idxs):
            alpha_z = alpha[z]
            alpha_sr_z = alpha_sr[z]
            print(alpha_z / f_j)
            stack_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr_z / f_list[0]), 'Asymptotes')
            stack_sr_0 = load('slow_roll_alpha_f:%.4f' % (0. / f_list[0]), 'Asymptotes')
            A_zero_sr, del_A_zero_sr = [stack_sr_0[1], stack_sr_0[2]]

            plt.loglog(stack_sr[0], stack_sr[1] - A_zero_sr, color='grey', linewidth=5, alpha=0.35)

            for i, item_b in enumerate(b):  # index_b:
                b_i = b[i]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z / f_j, b_i, f_j), 'Asymptotes')

                stack_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0. / f_j, b_i, f_j), 'Asymptotes')
                A_zero, del_A_zero = [stack_0[1], stack_0[2]]

                max_idx = np.argmax(abs(stack[1] - A_zero))
                max_k, max_A = [stack[1][max_idx], stack[0][max_idx]]

                B_k_0 = 2e-22 * T_R**-1 * (1/H_inf * Mpl) ** (-1 / 3) * ((H_inf) ** 2 * max_k ** 2 * max_A * Mpl ** 2) ** (
                            2 / 3) * 23.49 * 1e-32
                Lambda_0 = 5.5e4 * T_R**-1 * (1/H_inf * Mpl) ** (-1 / 3) * ((H_inf) ** 2 * max_k ** 2 * max_A * Mpl ** 2) ** (
                            2 / 3) * 23.49 * 1e-32
                print(r'$\alpha:%.2f\quad bf:%.2f \quad f:%.5f$'% (alpha_z/f_j, b_i*f_j, f_j))
                print(r'$\Lambda_0$', r'$B_{k_0}$')
                print(Lambda_0, B_k_0)

    return


B_0()


def Asymp_alpha_plot():
    for j in [0]:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])
        #plt.subplot(2,1,1+j)
        colors_alpha = cm.rainbow(np.linspace(0,1,len(alpha)))
        for z_idx, z in enumerate(index_alpha):#index_alpha:
            alpha_z = alpha[z]
            alpha_sr_z = alpha_sr[z]
            print(alpha_z/f_j)
            stack_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr_z/f_list[0]), 'Asymptotes')
            stack_sr_0 = load('slow_roll_alpha_f:%.4f' % (0./f_list[0]), 'Asymptotes')
            A_zero_sr, del_A_zero_sr = [stack_sr_0[1], stack_sr_0[2]]

            #plt.loglog(stack_sr[0], stack_sr[1]-A_zero_sr, color = 'grey', linewidth = 5, alpha = 0.35)

            for i in [0,1]:#index_b:
                b_i = b[i]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Asymptotes')

                stack_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0./f_j, b_i, f_j), 'Asymptotes')
                A_zero, del_A_zero = [stack_0[1], stack_0[2]]
                #plt.subplot(211)
                plt.scatter(alpha_z/f_j, max(stack[1]-A_zero), marker = ['+','o'][i], color = 'red')#colors_alpha[z_idx])#,label=r'f:%.4f  bf:%.2f' % (b_i * f_j))
                N_kill = Buffer[j, i, 2][-1]
                #plt.subplot(212)
                plt.scatter(alpha_z/f_j, max(stack[2]-del_A_zero), marker = ['+','o'][i], color = 'blue')# colors_alpha[z_idx])
                plt.yscale('Log')
                #plt.legend()
    #plt.axes().set_aspect(2)
    plt.xlabel(r'$k/a_{end}H_{end}$')
    plt.ylabel(r'$\sqrt{2k}\mathcal{A}_\pm(|k,N_{end}|)$')
    return plt.show()


Asymp_alpha_plot()


intg_steps = 3000

index_f = [0,1]
targ_alpha = np.array([5,15])
#index_alpha = [0,5,10,15,-1]
index_b = [1]


def rho_plot():
    N_kill_sr = slow_roll_stack[2][-1]
    intg_N_sr = np.linspace(N_start, N_kill_sr, intg_steps)
    a_map_sr = np.array([*map(a, intg_N_sr)])

    norm = 1
    rho_phi = 1/(3*(3.6e-5)**2)
    for idx_f, j in enumerate(index_f):
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        targ_alpha_idxs = [np.argmin(abs(targ - alpha)) for targ in targ_alpha*f_j]

        colors = cm.rainbow(np.linspace(0, 1, len(index_b)*len(targ_alpha)))
        color_idx = 0
        plt.subplot(2, 2, 1 + idx_f)
        for idx_b,i in enumerate(index_b):
            b_i = b[i]
            N_kill = Buffer[j, i, 2][-1]
            intg_N = np.linspace(N_start, N_kill, intg_steps)
            a_map = np.array([*map(a, intg_N)])
            
            rho_B_0 =  load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0./f_j, b_i, f_j), 'Rho_B')
            rho_E_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0./f_j, b_i, f_j), 'Rho_E')
            plt.plot([], [], label=r'$bf:%.2f$' % (f_j * b_i))
            for z in targ_alpha_idxs:
                alpha_z = alpha[z]

                rho_B_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Rho_B')
                rho_E_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Rho_E')

                #plt.semilogy(N, H_map ** 2, label=r'$\phi$')

                rho_B = abs(rho_B_Spool[:,norm]-rho_B_0[:,norm])/a_map**4
                rho_E = abs(rho_E_Spool[:,norm]-rho_E_0[:,norm])/a_map**2

                plt.semilogy(intg_N, rho_B/rho_phi,linestyle = '-',
                             label=r'$\alpha/f:%.1f$'%(alpha_z/f_j), color = colors[color_idx])
                plt.semilogy(intg_N, rho_E/rho_phi, linestyle='--',
                             color = colors[color_idx])


                color_idx = color_idx + 1
    plt.subplot(222)
    plt.legend()
    return plt.show()

rho_plot()

def Backreaction_plot():
    N_kill_sr = slow_roll_stack[2][-1]
    intg_N_sr = np.linspace(N_start, N_kill_sr, intg_steps)
    a_map_sr = np.array([*map(a, intg_N_sr)])
    #dVdphi_sr = 2 * np.array([gen_map(i, N, phi_solved) for i in intg_N])
    norm = 1
    H_inf = 3.6e-5
    r = 0.1

    for idx_f,j in enumerate(index_f):
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        targ_alpha_idxs = [np.argmin(abs(targ - alpha)) for targ in targ_alpha * f_j]
        colors = cm.rainbow(np.linspace(0, 1, len(index_b) * len(targ_alpha)))
        color_idx = 0
        plt.subplot(2, 2, 1 + idx_f)

        for idx_b,i in enumerate(index_b):
            b_i = b[i]

            phi_solved = Buffer[j, i, 0]
            phi_prime_solved = Buffer[j, i, 1]
            N = Buffer[j, i, 2]
            N_kill = Buffer[j, i, 2][-1]
            intg_N = np.linspace(N_start, N_kill, intg_steps)
            a_map = np.array([*map(a, intg_N)])
            #dVdphi = 2*np.array([gen_map(i,N, phi_solved) for i in intg_N])[:-1]
            dVdphi = 3 * sqrt(r) / H_inf
            plt.plot([], [], label=r'$bf:%.2f$' % (f_j * b_i),linestyle='')
            #plt.semilogy(N, dVdphi, label=r'$V_{,\phi}$')

            Backreaction_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0/f_j, b_i, f_j), 'Backreaction')

            for z in targ_alpha_idxs:
                alpha_z = alpha[z]

                #gauge_stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Raw')
                #N_g = np.linspace(gauge_stack[0, i, 0, 0], gauge_stack[0, i, 0, -1], len(N_index))
                Backreaction_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Backreaction')
                print(np.shape(Backreaction_Spool))
                Backreaction = Backreaction_Spool

                y = (Backreaction[:,norm]-Backreaction_0[:,norm])
                dy = np.diff(y)
                dx = np.diff(intg_N)
                #dy_filtered = lowess(abs(dy / dx), intg_N[:-1], frac=0.9)
                #Backreaction_filtered = dy_filtered[:, 1] #* H_map_Backreaction[:-1]
                #Backreaction_map = np.array([dy_filtered[:, 0], Backreaction_filtered])

                grad = abs(dy/dx)
                grad = np.gradient(y,intg_N)
                Bck = alpha_z / f_j * H_inf * abs(grad) / dVdphi / a_map**3

                plt.semilogy(intg_N, Bck, label=r'$\alpha/f:%.1f$'%(alpha_z/f_j),
                             color = colors[color_idx], linewidth=0.75)


                color_idx = color_idx+1
    plt.subplot(222)
    plt.legend()
    #plt.xlim(55., N_final - 2.)
    return plt.show()

Backreaction_plot()


def mapper(var, var_arr , target_arr):

    diff_arr = abs(var_arr-var)
    min_idx = np.argmin(diff_arr)

    out = target_arr[min_idx]

    return out



def rho_Back_plot():
    N_kill_sr = slow_roll_stack[2][-1]
    intg_N_sr = np.linspace(N_start, N_kill_sr, intg_steps)
    a_map_sr = np.array([*map(a, intg_N_sr)])

    norm = 0

    H_inf = 3.6e-5
    r = 0.1
    phi_cmb = 16
    rho_phi = 3 / (H_inf ** 2)
    dVdphi = 3 * sqrt(r) / H_inf
    print(rho_phi, dVdphi)

    for idx_f, j in enumerate(index_f):
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])

        targ_alpha_idxs = [np.argmin(abs(targ - alpha)) for targ in targ_alpha * f_j]

        colors = cm.brg(np.linspace(0, 1, (len(index_b)+1) * len(targ_alpha)))
        #colors = ['red', 'blue', 'yellow', 'darkviolet', 'magenta']

        color_idx = 0
        plt.subplot(2, 2, 1 + idx_f)

        for idx_b, i in enumerate(index_b):

            b_i = b[i]
            N_kill = Buffer[j, i, 2][-1]
            intg_N = np.linspace(N_start, N_kill, intg_steps)
            a_map = np.array([*map(a, intg_N)])

            phi_map = abs(np.array([mapper(o, Buffer[j, i, 2], Buffer[j, i, 0]) for o in intg_N]))
            dphi_map = abs(np.array([mapper(o, Buffer[j, i, 2], Buffer[j, i, 1]) for o in intg_N]))

            #rho_phi = (3/H_inf)**2*(phi_map/phi_cmb)**2 / a_map**4
            #dVdphi = (3/H_inf)*(phi_map/phi_cmb)**2*dphi_map / a_map**3

            rho_B_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0. / f_j, b_i, f_j), 'Rho_B')
            rho_E_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0. / f_j, b_i, f_j), 'Rho_E')

            if idx_f+1 == 2:
                plt.plot([], [], label=r'$bf:%.2f$' % (f_j * b_i), linestyle = '')
            Backreaction_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0 / f_j, b_i, f_j), 'Backreaction')
            Backreaction_sr_0  = np.array(load('slow_roll_alpha_f:%.4f'%(0/f_list[0]), 'Backreaction'))
            for z in targ_alpha_idxs:
                alpha_z = alpha[z]

                rho_B_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z / f_j, b_i, f_j), 'Rho_B')
                rho_E_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z / f_j, b_i, f_j), 'Rho_E')

                rho_B_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr[z] / f_list[0]), 'Rho_B')
                rho_E_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr[z] / f_list[0]), 'Rho_E')

                # plt.semilogy(N, H_map ** 2, label=r'$\phi$')

                rho_B = abs(rho_B_Spool[:, norm]) / a_map ** 4 - rho_B_0[:, norm]/ a_map ** 4
                rho_E = abs(rho_E_Spool[:, norm]) / a_map ** 2 - rho_E_0[:, norm]/ a_map ** 2

                rho_B_sr = abs(rho_B_sr[:, norm] ) / a_map ** 4 - rho_B_0[:, norm]/ a_map ** 4
                rho_E_sr = abs(rho_E_sr[:, norm] ) / a_map ** 2 - rho_E_0[:, norm]/ a_map ** 2

                Backreaction_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z / f_j, b_i, f_j), 'Backreaction')
                Backreaction = Backreaction_Spool

                Backreaction_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr[z] / f_list[0]), 'Backreaction')
                Backreaction_sr = np.array(Backreaction_sr)

                y = (Backreaction[:, norm] - Backreaction_0[:, norm])
                y_sr = (Backreaction_sr[:,norm] - Backreaction_sr_0[:,norm])

                Bck = alpha_z / f_j * H_inf * abs(y) / dVdphi / a_map ** 3
                Bck_sr = alpha_z / f_j * H_inf * abs(y_sr) / dVdphi / a_map ** 3

                if idx_f+1 == 2:
                    label = r'$\alpha/f:%.1f$' % (alpha_z / f_j)
                else:
                    label = None
                plt.semilogy(intg_N, (rho_B ) / rho_phi, linestyle='-', color = colors[color_idx], label=label, linewidth=1.2)

                plt.semilogy(intg_N, Bck, color = colors[color_idx], linewidth=1.5, linestyle = '--')

                plt.semilogy(intg_N, rho_E / rho_phi, linestyle=':', color=colors[color_idx])

                color_idx = color_idx + 1

                if z == targ_alpha_idxs[0] and idx_f+1==2:
                    plt.plot([], [], label=r'$bf:0$', linestyle = '')

                if idx_f+1 == 2:
                    plt.plot([],[],label = r'$\alpha/f:%.1f$' % (alpha_z / f_j), color = colors[color_idx])

                plt.semilogy(intg_N, (rho_B_sr ) / rho_phi, linestyle='-', color=colors[color_idx], linewidth=1.2)

                plt.semilogy(intg_N, Bck_sr, color=colors[color_idx], linewidth=1.5, linestyle='--')

                plt.semilogy(intg_N, rho_E_sr / rho_phi, linestyle=':', color=colors[color_idx])

                color_idx = color_idx + 1

    plt.subplot(221)
    plt.legend()
    [plt.plot([],[],color = 'k', linestyle = ['-',':','--'][idx],
              label = [r'$\rho_{B}/\rho_{\phi}$',r'$\rho_{E}/\rho_{\phi}$',
                       r'$\frac{\alpha}{f}\left|\frac{\langle E \cdot B \rangle}{V_{,\phi}} \right|$'][idx])
     for idx in [0,1,2]]
    plt.legend()
    plt.subplot(222)
    plt.legend()
    return plt.show()

rho_Back_plot()

def rho_alpha_plot():
    N_kill_sr = slow_roll_stack[2][-1]
    intg_N_sr = np.linspace(N_start, N_kill_sr, intg_steps)
    a_map_sr = np.array([*map(a, intg_N_sr)])

    norm = 1

    H_inf = 3.6e-5
    r = 0.1
    rho_phi = 1 / (3 * H_inf ** 2)
    dVdphi = 3 * sqrt(r) / H_inf

    for idx_f, j in enumerate(index_f):
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        targ_alpha_idxs = [np.argmin(abs(targ - alpha)) for targ in targ_alpha * f_j]

        colors = cm.rainbow(np.linspace(0, 1, len(index_b)))
        colors = ['red', 'blue', 'yellow', 'darkviolet', 'magenta']
        color_idx = 0
        plt.subplot(2, 2, 1 + idx_f)

        for idx_b, i in enumerate(index_b):
            b_i = b[i]
            N_kill = Buffer[j, i, 2][-1]
            intg_N = np.linspace(N_start, N_kill, intg_steps)
            a_map = np.array([*map(a, intg_N)])

            rho_B_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0. / f_j, b_i, f_j), 'Rho_B')
            rho_E_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0. / f_j, b_i, f_j), 'Rho_E')

            if idx_f + 1 == 2:
                plt.plot([], [], label=r'$bf:%.2f$' % (f_j * b_i), color=colors[color_idx])
            Backreaction_0 = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (0 / f_j, b_i, f_j), 'Backreaction')
            for z in np.arange(0,len(alpha), 1):
                alpha_z = alpha[z]

                rho_B_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z / f_j, b_i, f_j), 'Rho_B')
                rho_E_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z / f_j, b_i, f_j), 'Rho_E')

                # plt.semilogy(N, H_map ** 2, label=r'$\phi$')

                rho_B = abs(rho_B_Spool[:, norm] - rho_B_0[:, norm]) / a_map ** 4
                rho_E = abs(rho_E_Spool[:, norm] - rho_E_0[:, norm]) / a_map ** 2

                Backreaction_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z / f_j, b_i, f_j), 'Backreaction')
                print(np.shape(Backreaction_Spool))
                Backreaction = Backreaction_Spool

                y = (Backreaction[:, norm] - Backreaction_0[:, norm])
                # dy = np.diff(y)
                # dx = np.diff(intg_N)
                # dy_filtered = lowess(abs(dy / dx), intg_N[:-1], frac=0.9)
                # Backreaction_filtered = dy_filtered[:, 1] #* H_map_Backreaction[:-1]
                # Backreaction_map = np.array([dy_filtered[:, 0], Backreaction_filtered])

                # grad = abs(dy / dx)
                # grad = np.gradient(y, intg_N)
                Bck = alpha_z / f_j * H_inf * abs(y) / dVdphi / a_map ** 3

                if idx_f + 1 == 2:
                    label = r'$\alpha/f:%.1f$' % (alpha_z / f_j)

                else:
                    label = None
                plt.scatter(alpha_z/f_j, (rho_B[-1]) / rho_phi, marker='+', color=colors[color_idx])#, label=label)

                #plt.semilogy(intg_N, Bck, color=colors[color_idx], linewidth=0.75, linestyle='--')

                plt.scatter(alpha_z/f_j, rho_E[-1] / rho_phi, marker='.',
                             color=colors[color_idx])
                plt.yscale('Log')
            color_idx = color_idx + 1

    plt.subplot(222)
    plt.legend()
    return plt.show()

rho_alpha_plot()


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

