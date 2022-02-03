from main import *
from plotting import *

def potent_plot():
    for x in np.linspace(1, 4, 5):

        potential_plot(x)

    plt.legend()
    plt.show()

    return

#potent_plot()

def k_generator(p,a_stack, len_k, kill_matrix, trigger_matrix):

    stack = []
    for idx_p, item_p in enumerate(p):
        t_i = trigger_matrix[idx_p]
        t_f = kill_matrix[idx_p]
        a_i = mapper(t_i, a_stack[idx_p][0], a_stack[idx_p][1])
        a_f = mapper(t_f, a_stack[idx_p][0], a_stack[idx_p][1])

        k = np.geomspace(0.001*a_i, 10*a_f, len_k)
        stack.append(k)

    return stack

#scal_plot()

#phase_plot(np.linspace(2,10,5))

#args = [3,20,1]
#Vec_plot(args)


p = np.linspace(2,4,1)
alpha = np.linspace(0.1,2, 2)
len_k = 40
x_len = steps/1000       #Integration steps for computing energy density, helicity and backreaction
##Kill Conditions##
splice = steps/100
use_limit_of = 'Phi'        ##Choose based on limiting computation of Phi or dPhi
lim_val_low, lim_val_high = [3, .9]               ##CAREFUL on the value(They are different for phi or dphi based limitors)

Data_set = 'p_%.2f_%.2f_%.1f_alpha_%.2f_%.2f_%.1f_k__%.1f_steps_%.1f'%(p[0], p[-1], len(p),
                                                                     alpha[0], alpha[-1], len(alpha), len_k, steps)

Master_path = '/media/teerthal/Repo/Delayed_oscillation/%s'%(Data_set)

make_dir(Master_path)
phi_spool_args = [p]
Phi_stack = load_or_exec('Phi_stack', 'Phi', Phi_Spooler, Master_path, phi_spool_args )

a_spool_args = [p,Phi_stack]
a_stack = load_or_exec('a_stack', 'a', a_spooler, Master_path, a_spool_args)

trigger_args = [p, Phi_stack, use_limit_of, lim_val_low]
trigger_matrix = initiation_matrix(trigger_args)

kill_matrix_args = [p, Phi_stack, splice, use_limit_of, lim_val_high]
kill_arr = kill_matrix(kill_matrix_args)

k = k_generator(p,a_stack,len_k, kill_arr, trigger_matrix)

scalar_plot(p,Phi_stack, a_stack, kill_arr, trigger_matrix, k)

gauge_spool_args = [p,alpha,k,a_stack,Phi_stack, kill_arr, trigger_matrix, Master_path]
load_or_exec_multi_file(Master_path, 'Raw', Gauge_spooler, gauge_spool_args)

asym_spool_args = [p, alpha, k, a_stack, Phi_stack, Master_path]
asymp_stack = load_or_exec('Asymp_stack', 'Asymptotes', Asymptotes, Master_path, asym_spool_args)

gauge_plot_args = [p,alpha,k,a_stack, Phi_stack, Master_path]
gauge_plot(gauge_plot_args)
asymp_plot([p,alpha,k,a_stack, Phi_stack], asymp_stack)
lim_temp_plot([p, alpha, k, a_stack, Phi_stack], asymp_stack)
hel_temp_plot([p, alpha, k, a_stack, Phi_stack], asymp_stack)

Rho_B_spool_args = [p,alpha,k, x_len, a_stack, Master_path]
load_or_exec_multi_file(Master_path, 'Rho_B', Rho_B_spooler, Rho_B_spool_args)

Rho_E_spool_args = [p,alpha,k, x_len, a_stack, Master_path]
load_or_exec_multi_file(Master_path, 'Rho_E', Rho_E_spooler, Rho_E_spool_args)

Helicity_spool_args = [p,alpha,k, x_len, a_stack, Master_path]
load_or_exec_multi_file(Master_path, 'Helicity', Helicity_spooler, Helicity_spool_args)

