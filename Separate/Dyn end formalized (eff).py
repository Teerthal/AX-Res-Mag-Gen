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
Gauge_steps = int(5e4)

N_start = 55.

steps = Gauge_steps*N_final/(N_final-N_start)
N_index = np.arange(0, int(Gauge_steps), 1)


cpu = 4

N_start_intervals = int(24*15)
delta_intervals = N_start_intervals

alpha_i = 1.
alpha_f = 20.
alpha_list_size = 3

phi_cmb = 15

# PARAMETERS FOR LOOPS
####################################

# f_list = np.array([0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.0001])
f_list = np.array([1e-2])
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

delta = np.linspace(5., 12., delta_intervals)

####################################

#End time for gauge solver
N_gauge_off = 0.

xi_samp_inter = 0.1     #For sampling xi around itme of interest to take 'non-oscillating' del phi for computing
                            #.....corresponding k

Data_set = '%.2f_%.2f_%.7f(final f)_%.0f_%.0f_%.0f'%(alpha_i,alpha_f, f_list[-1], Gauge_steps, len(f_list), delta_intervals)
Master_path = '/media/teerthal/Repo/Monodromy/%s'%Data_set
Master_path = '/home/teerthal/Repository/Gauge_Evolution/Monodromy/%s'%Data_set


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

            N_kill = Buffer[index_f, index_b, 2][-1]

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

#Scalar_plot()


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
                    plt.semilogy(gauge_N, kin)

    #plt.legend()
    return plt.show()

xi_plot()

warning_collect = []
def phi_prime_mapped(N, i, j):

    #idx = np.where(remapped_scalar[j, i, :, 2]==N)
    #mapped = remapped_scalar[j,i,idx,1]
    #print('check',mapped[0])

    #N_kill = Buffer[0, index_f, index_b, 2][-1]
    #N_kill = N_kill-N_gauge_off

    N_remapped = remapped_scalar[j, i, :, 2]
    N_kill = remapped_scalar[j, i, -1, 2]#; print('kill time for the gauge solver mapper:', N_kill)
    steps = len(N_remapped)

    if N>N_kill:
       idx = int(steps)-1;warning_collect.append([1])
    else:
        idx = abs(int(abs(N-N_start)/abs(N_kill- N_start)*steps-1))
    #idx = np.where(remapped_scalar[j,i,:,2]==N);print(idx)
    #mapped = remapped_scalar[j,i,idx,1]
    #print([N, idx])
    mapped = remapped_scalar[j,i,idx,1]
    #print([N, idx, 'countdown', N_kill-N])


    #N_solved = remapped_scalar[j,i,:,2]
    #diff_matrix = abs(N_solved-N)
    #min_idx = np.argmin(diff_matrix)
    #mapped = remapped_scalar[j,i,min_idx,1]

    return mapped

def test():
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        stack = []
        for i in index_b:
            b_i = b[i]
            memnonic = functools.partial(phi_prime_mapped, i=i, j=j)
            N_kill =remapped_scalar[j, i, -1, 2]; N = np.linspace(N_start, N_kill-N_gauge_off, Gauge_steps)
            #mem_map = map(memnonic, N)
            mem_map = np.array([*map(functools.partial(phi_prime_mapped, i=i,j=j), N)]);print('menomonic shape:', np.shape(mem_map))
            plt.plot(N, mem_map**2/2, label  = (b_i*f_j))

            phi_prime_solved = Buffer[0, j, i, 1]
            N_solved = Buffer[0, j, i, 2]
            plt.plot(N_solved,phi_prime_solved**2/2, label ='solution', linestyle = ':')
    plt.legend()
    plt.show()
    return
#test()

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

    Parameters = np.array([N_k(delta, index_f, index_b), h, alpha, index_b, index_f, phi_prime_mapped, k(delta, index_f, index_b)])

    r = ode(ODE).set_integrator('zvode').set_f_params(Parameters)

    # Initial Condition
    t_initial = -1
    A_initial = exp(-1j * k(delta, index_f, index_b) * t_initial)
    A_prime_initial = -1j * A_initial * k(delta, index_f, index_b) / (a_corr(N_start, index_f, index_b)*H(N_start))

    init = np.array([A_initial, A_prime_initial]);

    r.set_initial_value(init, N_start)

    N_kill =Buffer[index_f, index_b, 2][-1]#;print('kill time', N_kill)

    u = []
    t = []

    dt = (N_kill - N_gauge_off - N_start) / Gauge_steps
    while r.successful() and N_start <= r.t <= N_kill-N_gauge_off:
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
                    plt.semilogy(stack[0,0], stack[0,1], label='h%s'%h)
                plt.legend()
                plt.show()
                save(stack_h, 'alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')
                print('Gauge solution non slow roll', np.shape(stack_h))

    return [0]

raw_dir = 'Raw'

sub_path = '%s/%s' % (Master_path, raw_dir)
rewrite_RAW = input('Rewrite Raw data directory(T/F):')

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

    #idx = np.where(remapped_scalar[j, i, :, 2]==N)
    #mapped = remapped_scalar[j,i,idx,1]
    #print('check',mapped[0])

    #idx = int(N/slow_roll_stack[2,-1]*Gauge_steps)-1
    #mapped = slow_roll_stack[1,idx]

    #N_kill = Buffer[0, index_f, index_b, 2][-1]
    #N_kill = N_kill-N_gauge_off

    #idx_remap = int(N/remapped_scalar_sr[-1,2]*Gauge_steps)-1
    #idx_remap = int(N/N_kill*Gauge_steps)-1
    #mapped = remapped_scalar_sr[idx_remap,1]

    N_kill = remapped_scalar_sr[-1, 2]
    #N_kill = N_kill-N_gauge_off
    steps = len(remapped_scalar_sr[:, 2])

    if N>N_kill:
       idx = int(Gauge_steps)-1#;print('warning')
    else:
        idx = abs(int(abs(N-N_start)/abs(N_kill- N_start)*steps-1))

    mapped = remapped_scalar_sr[idx, 1]

    return mapped

def test_sr():

    N_kill = slow_roll_stack[2][-1]
    N = np.linspace(N_start, N_kill - N_gauge_off, Gauge_steps)
    mem_map = np.array([*map(functools.partial(phi_prime_mapped_sr, index_b=0, index_f=0), N)]);
    print('menomonic shape:', np.shape(mem_map))
    plt.plot(N, mem_map ** 2 / 2, label='sr')

    phi_prime_solved = slow_roll_stack[1]
    N_solved = slow_roll_stack[2]
    plt.plot(N_solved, phi_prime_solved ** 2 / 2, label='solution', linestyle=':')
    plt.legend()
    plt.show()
    return
#test_sr()

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

    N_final=slow_roll_stack[2,-1]

    dt = (N_final - N_gauge_off - N_start) / Gauge_steps
    while r.successful() and r.t <= N_final - N_gauge_off:
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

    return np.array([N, A, A_prime, norm_A, norm_del_A])


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


rewrite_RAW_sr = input('Rewrite Raw SR file:')

if rewrite_RAW_sr == 'T':
    execute_sr(delta)
    timer('Finish calculating and saving')

else:
    timer('Finish loading')

def Gauge_curves():
    for l in index_f:
        item_f = f_list[l]
        b = b_list(item_f)
        alpha = alpha_list(item_f)

        for j in index_b:
            item_b = b[j]

            k_map = np.array([*map(functools.partial(k, index_f=l, index_b=j),delta)])

            for z in index_alpha:
                item_alpha = alpha[z]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')
                gauge_sr = load('slow_roll_alpha_f:%.4f'%(item_alpha/f_list[0]), 'Raw')
                #plt.loglog(k_map, gauge_sr[0,:,1,-1], label='s bf:%.4f' %(item_b*item_f));plt.loglog(k_map,stack[0,:,1,-1]);plt.show()
                for i in np.arange(0, delta_intervals, 1):
                    plt.subplot(211)
                    plt.semilogy(gauge_sr[0, i, 0], gauge_sr[0, i, 1], label='sr', linestyle=':')

                    plt.subplot(212)
                    plt.semilogy(gauge_sr[0,i,0], gauge_sr[0,i,2], label='sr',linestyle=':')

                    k_i = k_map[i]

                    plt.subplot(211)
                    plt.ylabel(r'$|\mathcal{A}|$')
                    plt.semilogy(stack[0, i, 0], stack[0, i, 1], label='k:%s ,bf:%s' % (k_i, item_b * item_f))
                    plt.legend()
                    plt.subplot(212)
                    plt.ylabel(r'$|\frac{d\mathcal{A}}{dx}|$')

                    A_prime = (stack[0, i, 2])
                    plt.semilogy(stack[0, i, 0], A_prime, label='k:%s' % k_i)
                    plt.legend()
        plt.show()

    return


#Gauge_curves()

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
    for j in index_f:
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
            asymptotes_sr = np.array([k_map_sr, gauge_sr[0, :, 1, -1], gauge_sr[0, :, 2, -1], gauge_sr[0, :, 3, -1]])
            save(asymptotes_sr, 'slow_roll_alpha_f:%.4f' % (alpha_sr_z/f_list[0]), 'Asymptotes')

            stack_j = []
            for i in index_b:
                b_i = b[i]

                k_map = np.array([*map(functools.partial(k, index_f=j, index_b=i), delta)]);
                #print(k_map)

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Raw')
                asymptotes = np.array([k_map, stack[0, :, 1, -1], stack[0, :, 2, -1], stack[0, :, 3, -1]])

                save(asymptotes, 'alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Asymptotes')

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
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])

        for z in index_alpha:
            alpha_z = alpha[z]
            alpha_sr_z = alpha_sr[z]

            stack_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr_z/f_j), 'Asymptotes')
            plt.loglog(stack_sr[0], stack_sr[1], label='Slow roll')

            for i in index_b:
                b_i = b[i]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Asymptotes')

                plt.loglog(stack[0], stack[1], label=r'bf:%.4f' % (b_i * f_j))
                # plt.loglog(stack[0], stack[2], label=r'$\frac{d\mathcal{A}_-}{dN}$')
                plt.legend()

    plt.xlabel(r'$k/a_{end}H_{end}$')
    plt.ylabel(r'$\sqrt{2k}\mathcal{A}_-(|k\eta_{end}|)$')
    return plt.show()


Asymptotic_plot()


def Shredder_B(A_z, k_map, N_index):
    #A_z = stack[0, :, 3, N_index]
    Integrand = 1 / (4 * pi ** 2) * A_z * k_map ** 3
    integrated = trapz(Integrand, x=k_map)

    return integrated


def Shredder_E(stack, k_map, N_index):
    A_prime_z = stack[0, :, 4, N_index]
    Integrand = 1 / (4 * pi ** 2) * A_prime_z * (k_map)

    return trapz(Integrand, x=k_map)


def Shredder_Back(stack, k_map, N_index):
    A_z = stack[0, :, 3, N_index]
    Integrand = k_map ** 2 / (8 * pi ** 2) * A_z

    return trapz(Integrand, x=k_map)


###To reduce integration steps####
N_index = np.arange(0, int(Gauge_steps),1)

def Spooler_rho_B():
    stack_j = []
    for j in index_f:
        item_f = f_list[j]
        alpha = alpha_list(item_f)
        b = b_list(item_f)

        stack_i = []
        for i in index_b:
            item_b = b[i]

            k_map = np.array([*map(functools.partial(k, index_f=j, index_b=i), delta)])

            stack_z = []
            for z in index_alpha:
                item_alpha = alpha[z]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')

                partial_Shred = functools.partial(Shredder_B, stack, k_map)

                pool = Pool(cpu)

                #if __name__ == '__main__':
                    #rho_B_map = np.array(pool.map(partial_Shred, N_index))
                #pool.close()

                rho_B_map = []
                for idx in N_index:
                    A_z = stack[0, :, 3, idx]

                    Shred = Shredder_B(A_z, k_map, idx)
                    rho_B_map.append(Shred)
                rho_B_map = np.array(rho_B_map)
                print(np.shape(rho_B_map))
                #N = stack[0, 0, 0]
                N = np.linspace(stack[0,0,0,0], stack[0,0,0,-1], len(N_index))

                a_par = functools.partial(a_corr, index_f=j, index_b=i)

                a_map_Gauge = np.array([*map(a_par, N)])
                H_map_Gauge = np.array([*map(H, N)])

                # pool = Pool(cpu)

                # if __name__ == '__main__':
                # rho_B_map = np.array(
                # pool.map(functools.partial(rho_B, b=item_b,
                # f=item_f, alpha=item_alpha), N_index))
                # pool.close()
                timer('Finished Rho_B stack alpha/f:%.5f' % (item_alpha/item_f))
                save(rho_B_map, 'alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Rho_B')

    return

Rho_B_dir = 'Rho_B'

sub_path = '%s/%s' % (Master_path, Rho_B_dir)

if os.path.exists(sub_path):
    print('Sub Directory exists')
else:
    os.mkdir(sub_path)
    print('Sub directory does not exist')
    print('Builing new Subdirectory %s' % Rho_B_dir)


rewrite_rho_B = input('Rewrite Rho_B file:')

if rewrite_rho_B == 'T':
    Spooler_rho_B()
    timer('Finish calculating and saving')

else:
    timer('Finish loading')



def Spooler_rho_E():
    stack_j = []
    for j in index_f:
        item_f = f_list[j]
        alpha = alpha_list(item_f)
        b = b_list(item_f)

        stack_i = []
        for i in index_b:
            item_b = b[i]

            k_map = np.array([*map(functools.partial(k, index_f=j, index_b=i), delta)])

            stack_z = []
            for z in index_alpha:
                item_alpha = alpha[z]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')

                partial_Shred = functools.partial(Shredder_E, stack, k_map)

                pool = Pool(cpu)

                if __name__ == '__main__':
                    rho_E_map = np.array(pool.map(partial_Shred, N_index))
                pool.close()

                N = np.linspace(stack[0, 0, 0, 0], stack[0, 0, 0, -1], len(N_index))
                a_par = functools.partial(a_corr, index_f=j, index_b=i)

                a_map_Gauge = np.array([*map(a_par, N)])
                H_map_Gauge = np.array([*map(H, N)])

                # pool = Pool(cpu)

                # if __name__ == '__main__':
                # rho_E_map = np.array(
                # pool.map(functools.partial(rho_E, b=item_b,
                # f=item_f, alpha=item_alpha), N_index))
                # pool.close()
                timer('Finished Rho_E stack alpha/f:%.5f' % (item_alpha / item_f))
                save(rho_E_map, 'alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha / item_f, item_b, item_f), 'Rho_E')

    return


Rho_E_dir = 'Rho_E'

sub_path = '%s/%s' % (Master_path, Rho_E_dir)

if os.path.exists(sub_path):
    print('Sub Directory exists')
else:
    os.mkdir(sub_path)
    print('Sub directory does not exist')
    print('Builing new Subdirectory %s' % Rho_E_dir)


rewrite_rho_E = input('Rewrite Rho_E file:')

if rewrite_rho_E == 'T':
    Spooler_rho_E()
    timer('Finish calculating and saving')

else:
    timer('Finish loading')


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

            k_map = np.array([*map(functools.partial(k, index_f=j, index_b=i), delta)])

            stack_z = []
            for z in index_alpha:

                item_alpha = alpha[z]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')

                partial_Shred = functools.partial(Shredder_Back, stack, k_map)

                pool = Pool(cpu)

                if __name__ == '__main__':
                    Backreaction_map = np.array(pool.map(partial_Shred, N_index))
                pool.close()


                #dN = float((N_final - 2. - N_start) / Gauge_steps)
                y = Down_res(Backreaction_map, 10)

                N = np.linspace(stack[0, 0, 0, 0], stack[0, 0, 0, -1], len(N_index))
                N_Backreaction = np.linspace(N[0], N[-1], len(y))
                H_map_Backreaction = np.array([*map(H, N_Backreaction)])
                a_map_Backreaction = np.array([*map(a, N_Backreaction)])

                dy = np.array(np.diff(y))
                dx = np.diff(N_Backreaction)

                dy_filtered = lowess(abs(dy / dx), N_Backreaction[:-1], frac=0.01)

                Backreaction_filtered = item_alpha / item_f * dy_filtered[:, 1] * H_map_Backreaction[:-1] / a_map_Backreaction[:-1]**3

                Backreaction_stack = np.array([dy_filtered[:, 0], Backreaction_filtered])
                timer('Finished Backreaction stack alpha/f:%.5f' % (item_alpha / item_f))

                save(abs(Backreaction_map),'alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha / item_f, item_b, item_f), 'Backreaction')

    return

Backreaction_dir = 'Backreaction'

sub_path = '%s/%s' % (Master_path, Backreaction_dir)

if os.path.exists(sub_path):
    print('Sub Directory exists')
else:
    os.mkdir(sub_path)
    print('Sub directory does not exist')
    print('Builing new Subdirectory %s' % Backreaction_dir)


rewrite_Backreaction = input('Rewrite Backreaction file:')

if rewrite_Backreaction == 'T':
    Spooler_Bacreaction()
    timer('Finish calculating and saving')

else:
    timer('Finish loading')

timer('Starting slow roll calculations')

def Spooler_rho_B_sr():

    alpha_sr = alpha_list(f_list[0])

    for z in index_alpha:
        k_map = np.array([*map(k_sr, delta)]);

        stack = load('slow_roll_alpha_f:%.4f'%(alpha_sr[z]/f_list[0]), 'Raw')

        partial_Shred = functools.partial(Shredder_B, stack, k_map)

        pool = Pool(cpu)

        if __name__ == '__main__':
            rho_B_map = np.array(pool.map(partial_Shred, N_index))
        pool.close()

        N = np.linspace(stack[0, 0, 0, 0], stack[0, 0, 0, -1], len(N_index))
        a_map_Gauge = np.array([*map(a_corr_sr, N)])

        H_map_Gauge = np.array([*map(H, N)])

        # pool = Pool(cpu)

        # if __name__ == '__main__':
        # rho_B_map = np.array(
        # pool.map(functools.partial(rho_B, b=item_b,
        # f=item_f, alpha=item_alpha), N_index))
        # pool.close()

        save(rho_B_map ,'slow_roll_alpha_f:%.4f'%(alpha_sr[z]/f_list[0]), 'Rho_B')


    return

rewrite_rho_B_sr = input('Rewrite Rho_B SR file:')

if rewrite_rho_B_sr == 'T':
    Spooler_rho_B_sr()
    timer('Finish calculating and saving')

else:
    timer('Finish loading')



def Spooler_rho_E_sr():
    alpha_sr = alpha_list(f_list[0])

    for z in index_alpha:
        k_map = np.array([*map(k_sr, delta)]);

        stack = load('slow_roll_alpha_f:%.4f'%(alpha_sr[z]/f_list[0]), 'Raw')

        partial_Shred = functools.partial(Shredder_E, stack, k_map)

        pool = Pool(cpu)

        if __name__ == '__main__':
            rho_E_map = np.array(pool.map(partial_Shred, N_index))
        pool.close()

        N = np.linspace(stack[0, 0, 0, 0], stack[0, 0, 0, -1], len(N_index))
        a_map_Gauge = np.array([*map(a_corr_sr, N)])

        H_map_Gauge = np.array([*map(H, N)])

        # pool = Pool(cpu)

        # if __name__ == '__main__':
        # rho_B_map = np.array(
        # pool.map(functools.partial(rho_B, b=item_b,
        # f=item_f, alpha=item_alpha), N_index))
        # pool.close()
        print(np.shape(rho_E_map))
        save(rho_E_map , 'slow_roll_alpha_f:%.4f'%(alpha_sr[z]/f_list[0]),'Rho_E')
    return



rewrite_rho_E_sr = input('Rewrite Rho_E SR file:')

if rewrite_rho_E_sr == 'T':
    Spooler_rho_E_sr()
    timer('Finish calculating and saving')

else:
    timer('Finish loading')

def Spooler_Bacreaction_sr():
    alpha_sr = alpha_list(f_list[0])

    for z in index_alpha:
        k_map = np.array([*map(k_sr, delta)]);

        stack = load('slow_roll_alpha_f:%.4f'%(alpha_sr[z]/f_list[0]), 'Raw')

        partial_Shred = functools.partial(Shredder_Back, stack, k_map)

        pool = Pool(cpu)

        if __name__ == '__main__':
            Backreaction_map = np.array(pool.map(partial_Shred, N_index))
        pool.close()

        N = np.linspace(stack[0, 0, 0, 0], stack[0, 0, 0, -1], len(N_index))
        a_map_Gauge = np.array([*map(a_corr_sr, N)])

        H_map_Gauge = np.array([*map(H, N)])

        # pool = Pool(cpu)

        # if __name__ == '__main__':
        # rho_B_map = np.array(
        # pool.map(functools.partial(rho_B, b=item_b,
        # f=item_f, alpha=item_alpha), N_index))
        # pool.close()
        print(np.shape(Backreaction_map))
        save(Backreaction_map, 'slow_roll_alpha_f:%.4f'%(alpha_sr[z]/f_list[0]), 'Backreaction')
    return


rewrite_Backreaction_sr = input('Rewrite Backreaction SR file:')

if rewrite_Backreaction_sr == 'T':
    Spooler_Bacreaction_sr()
    timer('Finish calculating and saving')

else:
    timer('Finish loading')


def rho_plot():
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)

        for i in index_b:
            b_i = b[i]

            for z in index_alpha:
                alpha_z = alpha[z]

                gauge_stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Raw')
                N = np.linspace(gauge_stack[0, i, 0, 0], gauge_stack[0, i, 0, -1], len(N_index))
                rho_B_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Rho_B')
                rho_E_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Rho_E')

                #plt.semilogy(N, H_map ** 2, label=r'$\phi$')
                plt.semilogy(N, abs(rho_B_Spool), label=r'$\rho_B$')
                plt.semilogy(N, abs(rho_E_Spool) , label=r'$\rho_E$')
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
            N = Buffer[0, j, i, 2]

            dVdphi = phi_solved / phi_cmb

            plt.semilogy(N, dVdphi, label=r'$V_{,\phi}$')

            for z in index_alpha:
                alpha_z = alpha[z]

                gauge_stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Raw')
                N_g = np.linspace(gauge_stack[0, i, 0, 0], gauge_stack[0, i, 0, -1], len(N_index))


                Backreaction_Spool = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Backreaction')
                print(np.shape(Backreaction_Spool))
                plt.semilogy(N_g, Backreaction_Spool, label=r'$\frac{\alpha}{f}< E.B >$')

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
