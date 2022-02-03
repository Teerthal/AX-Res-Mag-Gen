from main import *
from plotting import *

def potent_plot():
    for x in np.linspace(1, 4, 5):

        potential_plot(x)

    plt.legend()
    plt.show()

    return

#potent_plot()

def k_generator(p, H_m, a_stack, len_k, kill_matrix, trigger_matrix):

    stack_p = []
    for idx_p, item_p in enumerate(p):
        stack_H = []
        for idx_H, item_H in enumerate(H_m):
            t_i = trigger_matrix[idx_p][idx_H]
            t_f = kill_matrix[idx_p][idx_H]
            print('E folding:',t_f/t_i)
            #a_i = mapper(t_i, a_stack[idx_p][idx_H][0], a_stack[idx_p][idx_H][1])
            #a_f = mapper(t_f, a_stack[idx_p][idx_H][0], a_stack[idx_p][idx_H][1])
            #print('x',t_i, t_f)
            #print(log(a_i), log(a_f))
            #k = np.arange(0.01*a_i, 4*a_f, len_k)# abs(0.01*a_i - 4*a_f)/len_k)
            #k = np.geomspace(exp(-15), 10, len_k)*exp(t_f)# Geomspace for exp ###SET 1####
            #print(k[0],k[-1])
            k = np.geomspace(1e-3, 5, len_k)*exp(t_f)
            stack_H.append(k)
        stack_p.append(stack_H)
    return stack_p

#scal_plot()

#phase_plot(np.linspace(2,10,5))

#args = [3,20,1]
#Vec_plot(args)


####Set 1########Linearly sparsed k####
p = np.linspace(1.,6.5,20)
alpha = np.linspace(0.,5, 50)
len_k = 24*20
x_len = steps/1000       #Integration steps for computing energy density, helicity and backreaction
##Kill Conditions##
splice = 1000
use_limit_of = 'Phi'        ##Choose based on limiting computation of Phi or dPhi
lim_val_low, lim_val_high = [phi_i, 1e-2]
##CAREFUL on the value(They are different for phi or dphi based limitors)
H_m = [1.]
cross_lim = 5
lnA_thresh = 2
max_lnA_thresh = 5


####Set 2########Linearly sparsed k####
p = np.linspace(2.,6.,40)

alpha = np.linspace(0.,5, 50)
alpha = np.array([0,1,2,3,4,5])

len_k = 24*80
x_len = steps/1000       #Integration steps for computing energy density, helicity and backreaction
##Kill Conditions##
splice = 1000
use_limit_of = 'Phi'        ##Choose based on limiting computation of Phi or dPhi
lim_val_low, lim_val_high = [3.8, 1e-2]
##CAREFUL on the value(They are different for phi or dphi based limitors)
H_m = [1.]
cross_lim = 20
lnA_thresh = 2
max_lnA_thresh = 5


Data_set = 'var_a_Bck_%s_H_m_%.3f_%.3f_%.0f_p_%.2f_%.2f_%.1f_alpha_%.2f_%.2f_%.1f_k__%.2f_steps_%.1f_phi_lim_%.3f'%(Bck, H_m[0], H_m[-1], len(H_m), p[0], p[-1], len(p),
                                                                     alpha[0], alpha[-1], len(alpha), len_k, steps, lim_val_high)

#Master_path = '/media/teerthal/Repo/Delayed_oscillation/Simultaneous/%s'%(Data_set)
Master_path = '/str1/Teerthal/Spectator/%s'%(Data_set)


make_dir(Master_path)
phi_spool_args = [p, H_m]
Phi_stack = load_or_exec('Phi_stack', 'Phi', Phi_Spooler, Master_path, phi_spool_args )

a_spool_args = [p, H_m, Phi_stack]
a_stack = load_or_exec('a_stack', 'a', a_spooler, Master_path, a_spool_args)

trigger_args = [p, H_m, Phi_stack, use_limit_of, lim_val_low, lnA_thresh]
trigger_matrix = initiation_matrix(trigger_args)

kill_matrix_args = [p, H_m, Phi_stack, splice, use_limit_of, lim_val_high, cross_lim, trigger_matrix, max_lnA_thresh]
kill_arr = kill_matrix(kill_matrix_args)

k = k_generator(p, H_m, a_stack,len_k, kill_arr, trigger_matrix)
print(np.shape(k))

targ_p = [1,2,3,4,5,6.5]
#targ_p = p
targ_p_idxs = [np.argmin(abs(targ-p)) for targ in targ_p]

scalar_plot(p,H_m, Phi_stack, a_stack, kill_arr, trigger_matrix)
#para_plot(p, k, alpha, Phi_stack)
#scalar_plot2(p,H_m, Phi_stack, a_stack, kill_arr, trigger_matrix, targ_p, targ_p_idxs)

#scalar_animation(p,H_m, Phi_stack, a_stack, kill_arr, trigger_matrix, targ_p, targ_p_idxs)

#peaks_args_2 = [p, H_m, alpha, k, Master_path]
#peak_arr2 = load_or_exec('Peak_arr2', 'Peaks2', peaks_rms_N2, Master_path, peaks_args_2)

plt_intervals = 5
#peaks_args_2_1 = [p, H_m, alpha, k, targ_p, targ_p_idxs, Master_path]
#load_or_exec_multi_file(Master_path, 'Peaks2',peaks_rms_N2_1, peaks_args_2_1)
#contour_slices2([p, alpha, k, H_m, plt_intervals, targ_p, targ_p_idxs, kill_arr, Master_path])

#exit()
gauge_spool_args = [p,alpha,k, H_m, kill_arr, trigger_matrix, Phi_stack, Master_path]
load_or_exec_multi_file(Master_path, 'Raw', Gauge_spooler, gauge_spool_args)

asym_spool_args = [p, H_m, alpha, k, a_stack, Phi_stack, Master_path]
asymp_stack = load_or_exec('Asymp_stack', 'Asymptotes', Asymptotes, Master_path, asym_spool_args)
print(np.shape(asymp_stack))
targ_alpha = [5.]
targ_alpha_idxs = [np.argmin(abs(targ-alpha)) for targ in targ_alpha]



#slice_intervals = 500
#cross_threshold = 5
#peaks_args = [p, alpha, k, slice_intervals, cross_threshold, Master_path]
#peak_arr = load_or_exec('Peak_arr', 'Peaks', peaks_rms_N, Master_path, peaks_args)

#targ_p = [1.,2,3,4,5,6.5]
targ_alpha = [5.]
#targ_alpha = alpha
targ_alpha_idxs = [np.argmin(abs(targ-alpha)) for targ in targ_alpha]


gauge_plot_args = [p,alpha,k, H_m,a_stack, Phi_stack, Master_path, targ_alpha, targ_alpha_idxs]
gauge_plot(gauge_plot_args)


#pre_osc_plot([p, alpha, k, H_m, plt_intervals, targ_alpha, targ_alpha_idxs,targ_p, targ_p_idxs, Phi_stack, kill_arr, Master_path])
#asymp_plot([p,alpha,k, H_m,[targ_alpha,targ_alpha_idxs], kill_arr, Master_path], asymp_stack)
asymp_plot2([p,alpha,k, H_m, [targ_p, targ_p_idxs],[targ_alpha,targ_alpha_idxs], kill_arr, Master_path], asymp_stack)


targ_alpha = [1, 2., 3., 4, 5.]
targ_alpha_idxs = [np.argmin(abs(targ-alpha)) for targ in targ_alpha]

peak_k_plot([p, alpha, k, H_m, [targ_alpha,targ_alpha_idxs], kill_arr, Phi_stack, Master_path], asymp_stack)

#hel_temp_plot([p, alpha, k, H_m, a_stack, Phi_stack, kill_arr], asymp_stack)

#exit()


#contour_slices([p, alpha, k, plt_intervals], peak_arr)

#mean_mu_contour([p, alpha, k, H_m, plt_intervals], peak_arr2)



#mu_evo_plot([p, alpha, k, H_m, plt_intervals, targ_alpha, targ_alpha_idxs], peak_arr2)

#pre_osc_plot([p, alpha, k, H_m, plt_intervals, targ_alpha, targ_alpha_idxs, targ_p, targ_p_idxs, Phi_stack, Master_path])

#Spectrum_mode_plot([Master_path, alpha, p, k, H_m], asymp_stack)

#lim_temp_plot([p, alpha, k, H_m, a_stack, Phi_stack, kill_arr], asymp_stack)


Rho_B_spool_args = [p,alpha,k, x_len, a_stack, Master_path]
load_or_exec_multi_file(Master_path, 'Rho_B', Rho_B_spooler, Rho_B_spool_args)

Rho_E_spool_args = [p,alpha,k, x_len, a_stack, Master_path]
load_or_exec_multi_file(Master_path, 'Rho_E', Rho_E_spooler, Rho_E_spool_args)

Helicity_spool_args = [p,alpha,k, x_len, a_stack, Master_path]
load_or_exec_multi_file(Master_path, 'Helicity', Helicity_spooler, Helicity_spool_args)

