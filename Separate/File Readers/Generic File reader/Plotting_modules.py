from Parameters import *
from gen import *

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
            print(np.shape(N_phi_solved))
            N_kill = Buffer[j, i, 2][-1]

            #plt.subplot(211)
            #plt.semilogy(N_phi_solved, phi_solved)
            # plt.semilogy(slow_roll_stack[2], slow_roll_stack[0])
            plt.plot(N_phi_solved, phi_prime_solved ** 2 / 2, label='bf:%.4f'%(b_i*f_j) )
            #plt.plot(remapped_scalar[j,i,:,2], remapped_scalar[j,i,:,1]**2/2, linestyle=':', label='Interp')


            plt.legend()
            plt.ylabel(r'$\epsilon$')
            plt.xlabel('N')
            #plt.subplot(212)
            #plt.plot(N_phi_solved, phi_solved)


            #plt.semilogy(slow_roll_stack[2], slow_roll_stack[1]**2/2)
    plt.show()
    return




def a_corr(N, index_f, index_b):
    N_phi_solved = Buffer[index_f, index_b, 2]
    N_kill = N_phi_solved[-1]
    return exp(N - N_kill)

def a_corr_sr(N):
    N_kill = slow_roll_stack[2][-1]
    return exp(N - N_kill)


def k(delta, index_f, index_b):
    k = H(N_final) * exp(-delta)
    return k

def k_sr(delta):
    k = H(N_final) * exp(-delta)
    return k

def xi_plot():

    index_f = [0]
    index_b = [0]
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

                    xi =  abs(dphi_remap)
                    kin = k(delta_l,j,i)/a_full

                    plt.semilogy(gauge_N,xi)
                    plt.semilogy(gauge_N, kin, label = 'Delta:%.2f'%delta_l)

    plt.legend()
    return plt.show()


def Gauge_curves():
    for l in index_f:
        item_f = f_list[l]
        b = b_list(item_f)
        alpha = alpha_list(item_f)

        for j in index_b:
            item_b = b[j]

            k_map = np.array([*map(functools.partial(k, index_f=l, index_b=j),delta)])

            for z in [10]:
                item_alpha = alpha[z]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Raw')
                gauge_sr = load('slow_roll_alpha_f:%.4f'%(item_alpha/f_list[0]), 'Raw')
                #plt.loglog(k_map, gauge_sr[0,:,1,-1], label='s bf:%.4f' %(item_b*item_f));plt.loglog(k_map,stack[0,:,1,-1]);plt.show()
                for i in [int(N_start_intervals/2)]:#np.arange(0, delta_intervals, 50):
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



def Gauge_curves_A():
    for l in index_f:
        item_f = f_list[l]
        b = b_list(item_f)
        alpha = alpha_list(item_f)
        #plt.plot([],[],label='sr', linestyle =':')
        for j in index_b:
            item_b = b[j]

            k_map = np.array([*map(functools.partial(k, index_f=l, index_b=j),delta)])

            for z in index_alpha:
                item_alpha = alpha[z]

                stack = load('alpha:%.4f_b:%.4f_f:%.5f' % (item_alpha, item_b, item_f), 'Raw')
                gauge_sr = load('slow_roll_alpha:%.4f'%item_alpha, 'Raw')

                for i in np.arange(0, delta_intervals, 1):

                    plt.semilogy(gauge_sr[0, i, 0], gauge_sr[0, i, 1], linestyle=':')

                    k_i = k_map[i]


                    plt.ylabel(r'$|\mathcal{A}|$')
                    plt.semilogy(stack[0, i, 0], stack[0, i, 1])
                    plt.xlabel('N')
                    plt.legend()


        plt.show()

    return


def Asymptotic_plot():

    linestyle = ['-','--',':']

    #[plt.plot([],[],label=r'f:%.1e'%Decimal(f_list[idx]), linestyle=linestyle[idx]) for idx in [0,1]]

    for j in index_f:

        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])

        for z in [-1]:#np.arange(0,len(alpha),20):
            alpha_z = alpha[z]
            alpha_sr_z = alpha_sr[z]

            stack_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr_z/f_list[0]), 'Asymptotes')
            plt.loglog(stack_sr[0], stack_sr[1], label=r'$bf:0 \quad \alpha/f = %.1f$'%(alpha_sr_z/f_list[0]), linestyle=linestyle[j])

            for i in index_b:
                b_i = b[i]

                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z/f_j, b_i, f_j), 'Asymptotes')

                plt.loglog(stack[0], stack[1], label=r'bf:%.4f' % (b_i * f_j), linestyle=linestyle[j])
                # plt.loglog(stack[0], stack[2], label=r'$\frac{d\mathcal{A}_-}{dN}$')
                plt.legend()

    plt.xlabel(r'$k/a_{end}H_{end}$')
    plt.ylabel(r'$\sqrt{2k}\mathcal{A}_-(|k\eta_{end}|)$')
    return plt.show()


def Asym_alpha_plot():

    stack = Asymp_alpha()
    print(np.shape(stack))
    k_slices = np.shape(stack)[2]

    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])

        colors = cm.rainbow(np.arange(0, k_slices, 1))
        for idx_k in np.arange(0, k_slices-1, 1):

            b_patt = ['--',':', '-.', '-']
            for i in index_b:
                b_i = b[i]

                plt.semilogy(stack[j, idx_k, i, :, 0],stack[j, idx_k, i, :, 1], color = colors[idx_k], linestyle= b_patt[i])
    plt.legend()
    plt.xlabel(r'$\alpha$')
    return plt.show()


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


def lim_temp_plot():

    plot_idx = 1
    for idx_f, item_f in enumerate(f_list):
        for idx_b, item_b in enumerate(b_list(item_f)):
            plt.subplot(2,2,plot_idx)
            plot_idx = plot_idx + 1
            for idx_alpha, item_alpha in enumerate(alpha_list(item_f)):

                #stack_sr = load('slow_roll_alpha_f:%.4f' % (alpha_list(f_list[0]) / item_f), 'Asymptotes')
                stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (item_alpha/item_f, item_b, item_f), 'Asymptotes')

                k = stack[0]
                z_1 = np.zeros((len(alpha_list(item_f)), len(k)))

                for idx_k, item_k in enumerate(k):

                    sub_stack = stack[:,idx_k]
                    #print(np.shape(sub_stack))

                    k_1, A_1, dA_1, norm_A1 = [sub_stack[i] for i in [0, 1, 2, 3]]

                    color = log(A_1)

                    #if color < 0.:
                     #   color = 0.

                    z_1[idx_alpha, idx_k] = color

            z_1_nf = z_1;
            z_1 = np.flip(z_1, 0)

            x = k
            y = alpha_list(item_f)/item_f
            print(np.shape(z_1_nf), len(x), len(y))
            plt.xlabel(r'$k/a_f$')
            plt.ylabel(r'$\alpha$')

            x, y = np.meshgrid(x, y)
            plt.contourf(x, y, z_1_nf, len(alpha_list(item_f)), ticker=ticker.LogLocator())

            plt.xscale('Log')
            #plt.yscale('Log')
            plt.title(r'f:%.2f b:%s' % (item_f, item_b))
            plt.xlabel(r'$k/a_f$')
            plt.ylabel(r'$\alpha$')
            plt.colorbar()

        plt.show()

    return
