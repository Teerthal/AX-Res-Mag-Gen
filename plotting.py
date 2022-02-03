from main import *

def potential_plot(p):

    phi = np.linspace(-10,10, 1000)
    par = functools.partial(V, p=p)
    V_arr = np.array([*map(par,phi)])
    label = 'p:%s ' % (p)
    return plt.plot(phi, V_arr, label = label)

def scalar_plot(p, Phi_Stack, a_stack, kill_arr, trigger_arr, k, f):

    colors = cm.rainbow(np.linspace(0, 1, len(p)*len(f)))
    color_cnt = 0

    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)

    for idx_f, item_f in enumerate(f):

        for idx_p in np.arange(0,len(p),1):
            t_trigger = trigger_arr[idx_f][idx_p]
            t_kill = kill_arr[idx_f][idx_p]
            p_i = p[idx_p]
            sol = Phi_Stack[idx_f][idx_p]
            t = sol[0]
            phi = sol[1]
            dphi = sol[2]

            t_a = a_stack[idx_f][idx_p, 0]
            a = a_stack[idx_f][idx_p, 1]

            max_idx = np.argmax(abs(dphi))
            H_osc = H([dphi[max_idx], phi[max_idx]], p_i, item_f)
            #cross_idx = np.argwhere(np.diff(np.sign(np.real(dphi) - [0] * len(dphi)))).flatten()
            #osc_idx = cross_idx[1]
            #H_osc = H([dphi[osc_idx], phi[osc_idx]], p_i)

            ax0.loglog(t, abs(phi), label=r'$p:%.1f H_{osc}:%.3f$' % (p_i, H_osc), color = colors[color_cnt])
            ax0.loglog([t_kill]*2, [0,2], marker='X', color = colors[color_cnt], linestyle = ':')
            ax0.loglog([t_trigger] * 2, [0, 2], marker='X', color=colors[color_cnt], linestyle=':')
            ax0.set_ylabel(r'$\phi$')
            ax0.legend()
            #ax0.set_xscale('Log')

            par = functools.partial(V, p=p_i)
            V_arr = np.array([*map(par, phi)])

            par_H = functools.partial(H, p=p_i, f= item_f)
            args = np.array(list(zip(phi,dphi)))
            H_arr = np.array([*map(par_H, args)])

            ax1.plot(phi, V_arr, label='p:%s' % (p_i), color = colors[color_cnt])
            ax1.set_xlabel(r'$\phi$')
            ax1.set_ylabel(r'$V$')

            ax1.xaxis.set_label_position('top')
            ax1.yaxis.set_label_position('right')

            #ax2.plot(t, 1/2*(dphi/H_arr) ** 2 / 2)
            ax2.loglog(t, abs(dphi), color = colors[color_cnt])
            ax2.set_ylabel(r'$\epsilon$')
            ax2.set_xlabel(r'$mt$')
            ax2.set_xscale('Log')
            ax2.loglog(t, H_arr, linestyle = '--', color = colors[color_cnt])
            [ax2.loglog(t_a, k[idx_f][idx_p][i]/a, color = colors[color_cnt], linestyle = ':') for i in [0,-1]]

            ax3.semilogx(t_a,log(a/a_i), color = colors[color_cnt])

            ax3.set_ylabel(r'$log(a)$')
            ax3.set_xlabel(r'$mt$')
            ax3.yaxis.set_label_position('right')

            color_cnt = color_cnt+1

    ax2.plot([],[],color='k', label = 'H/m', linestyle = '--')
    ax2.legend()

    plt.legend()
    plt.show()

    return

def phase_plot(p):

    phi_arr = np.linspace(-2,2,10)
    dphi_arr = np.linspace(-1,1,10)

    color_p = cm.rainbow(np.linspace(0,1,len(p)))
    for idx_p in np.arange(0,len(p),1):
        p_i = p[idx_p]
        label = r'$p:%s$' % p_i
        plt.plot([],[],label=label,color=color_p[idx_p])

        for phi_I in phi_arr:
            for dphi_I in dphi_arr:
                initial = [phi_I,dphi_I]
                sol = Scalar_solver([p_i], initial=initial)
                t = sol[0]
                phi = sol[1]
                dphi = sol[2]

                #label = r'$\phi_i:%.2f  \dot{phi}_i:%.2f$'%(phi_I, dphi_I)
                #label = r'$p:%s$'%p_i
                plt.plot(phi, dphi,color=color_p[idx_p])

    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\dot{\phi}$')
    plt.legend()
    plt.show()
    return

def gauge_plot_red(args):
    [p, alpha, k, a_stack, Phi_stack, Master_path] = args
    h = [-1,1]
    h_style = ['--', ':']
    [plt.plot([],[],label = 'h:%s'%(h[i]), linestyle=h_style[i]) for i in [0,1]]

    for idx_h, item_h in enumerate(h):
        style_h = h_style[idx_h]

        for idx_p, item_p in enumerate(p):
            for idx_alpha, item_alpha in enumerate(alpha):
                name = 'alpha:%.5f_h:%s_p:%.3f' % (item_alpha, item_h, item_p)
                stack = load(name, 'Raw', Master_path)
                for idx_k, item_k in enumerate(k[idx_p]):
                    sub_stack = stack[idx_k]

                    t = sub_stack[0]
                    A = sub_stack[1]
                    dA = sub_stack[2]

                    plt.loglog(t, absolute(A), label = r'$h:%s p:%.1f \alpha:%.2f k:%.2f$'%(item_h,item_p,item_alpha,item_k), linestyle =style_h)

    plt.xlabel(r'mt')
    plt.ylabel(r'$|\sqrt{(2k)}\mathcal{A}_h|$')
    #plt.legend()
    plt.show()

    return


def gauge_plot(args):
    [p, alpha, k, a_stack, Phi_stack, Master_path, f] = args
    h = [-1,1]
    h_style = ['-', '--']
    #[plt.plot([],[],label = 'h:%s'%(h[i]), linestyle=h_style[i]) for i in [0,1]]
    #ax = [ 0, 0]
    for idx_f, item_f in enumerate(f):
        for idx_p, item_p in enumerate(p):

            for idx_h, item_h in enumerate(h):
                style_h = h_style[idx_h]
                plt.plot([],[],linestyle=h_style[idx_h], label = r'h:%s'%item_h, color='k')
                for idx_alpha, item_alpha in enumerate(alpha):
                    #item_alpha = alpha[idx_alpha]
                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f'% (item_alpha, item_h, item_p, item_f)
                    stack = load(name, 'Raw', Master_path)
                    plt_k_idxs = np.round(np.linspace(0, len(k[idx_f][idx_p])-1 , 4)).astype(int)
                    color = cm.rainbow(np.linspace(0, 1, len(plt_k_idxs)))
                    for subidx_k, idx_k in enumerate(plt_k_idxs):
                        item_k = k[idx_f][idx_p][idx_k]

                        sub_stack = stack[idx_k]

                        t = sub_stack[0]
                        #phi = sub_stack[1]
                        #dphi = sub_stack[2]
                        A = sub_stack[3]
                        dA = sub_stack[4]

                        sol = Phi_stack[idx_f][idx_p]
                        t_phi = sol[0]
                        phi = sub_stack[1]
                        dphi = sub_stack[2]

                        t_a = a_stack[idx_f][idx_p, 0]
                        a = a_stack[idx_f][idx_p, 1]

                        #print(len(t))
                        plt.subplot(2,2,1+idx_h)#+2*count_alpha)
                        plt.loglog(t, absolute(A), label = r'$ k:%.2f$'%(item_k), color = color[subidx_k])

                        plt.subplot(2,2,3+idx_h)
                        plt.loglog(t, absolute(dphi), label=r'$ k:%.2f$' % (item_k), color=color[subidx_k])
                        plt.loglog(t_a, absolute(item_k/a), color=color[subidx_k], linestyle = ':')

            #plt.title(r'p:%.1f' % item_p)
            plt.xlabel(r'mt')
            plt.ylabel(r'$|\sqrt{(2k)}\mathcal{A}_h|$')
            #plt.legend()
            #ax[0].get_shared_x_axes().join(ax[0], ax[1])
            plt.show()

    return

def asymp_plot(args, stack):
    [p, alpha, k, a_stack, Phi_stack, f] = args
    h = [-1,1]
    h_style = ['--', ':']
    for idx_f, item_f in enumerate(f):
        [plt.plot([],[],label = 'h:%s'%(h[i]), linestyle=h_style[i]) for i in [0,1]]

        for idx_h, item_h in enumerate(h):
            style_h = h_style[idx_h]

            for idx_p, item_p in enumerate(p):
                for idx_alpha, item_alpha in enumerate(alpha):
                    sub_stack = stack[idx_f, idx_h, idx_p, idx_alpha, :]
                    print(np.shape(sub_stack))
                    k = sub_stack[:,0]
                    A = sub_stack[:,1]
                    dA = sub_stack[:,2]

                    plt.loglog(k, absolute(A),
                               label=r'$h:%s p:%.1f \alpha:%.2f$' % (item_h, item_p, item_alpha),
                               linestyle=style_h)
        plt.title(r'$f:%.5f$'%item_f)
        plt.xlabel(r'k/m')
        plt.ylabel(r'$|\sqrt{(2k)}\mathcal{A}_h|$')
        plt.legend()
        plt.show()

    return


def lim_temp_plot(args, stack):
    [p, alpha, k, a_stack, Phi_stack] = args

    h = [-1, 1]

    plot_idx = 1
    for idx_p, item_p in enumerate(p):
        for idx_h, item_h in enumerate(h):
            plt.subplot(len(p), 2, plot_idx)
            plot_idx = plot_idx+1
            z_1 = np.zeros((len(alpha), len(k[idx_p])))

            for idx_alpha in range(len(alpha)):

                alpha_i = alpha[idx_alpha]
                for idx_k in range(len(k[idx_p])):

                    sub_stack = stack[idx_h, idx_p, idx_alpha, idx_k]

                    k_1, A_1, dA_1 = [sub_stack[i] for i in [0,1,2]]
                    x_final = t_f

                    z_1[idx_alpha, idx_k] = log(absolute(A_1))

                    color = log(absolute(A_1))/x_final

                    if color < 0.:
                        color = 0.

                    z_1[idx_alpha, idx_k] = color

            z_1_nf = z_1;
            z_1 = np.flip(z_1, 0)

            x = k[idx_p]
            y = alpha

            plt.xlabel(r'$k/a_f$')
            plt.ylabel(r'$\alpha$')

            x, y = np.meshgrid(x, y)
            plt.contourf(x, y, z_1_nf, 5)

            plt.xscale('Log')
            plt.title(r'p:%.2f h:%s'%(item_p,item_h))
            plt.xlabel(r'$k/a_f$')
            plt.ylabel(r'$\alpha$')
            plt.colorbar()

    plt.show()

    return


def hel_temp_plot(args, stack):
    h = [-1, 1]
    [p, alpha, k, a_stack, Phi_stack] = args
    z = np.zeros((len(alpha), len(k)))
    for idx_p, item_p in enumerate(p):
        for idx_alpha, item_alpha in enumerate(alpha):
            for idx_k, item_k in enumerate(k[idx_p]):

                sub_stack = [stack[i, idx_p, idx_alpha, idx_k] for i in [0,1]]
                x_final = t_f

                A_0 = absolute(sub_stack[0][1])
                A_1 = absolute(sub_stack[1][1])

                diff = abs(A_0 ** 2 - A_1 ** 2)
                sum = abs(A_0 ** 2 + A_1 ** 2)

                z[idx_alpha,idx_k] = diff/sum
    z = np.flip(z,0)
    plt.imshow(z, interpolation='bicubic',
               extent=(k[idx_p][0] , k[idx_p][-1],  alpha[0], alpha[-1]), aspect='auto')
    plt.colorbar()
    plt.xscale('Log')
    plt.show()
    return


def hel_growth_plot(args):
    h  = [-1,1]
    [p, alpha, k, a_stack, Phi_stack, Master_path] = args

    hel = np.zeros((len(alpha), len(k)))
    growth = np.zeros((len(alpha), len(k)))
    plot_idx = 1
    for idx_p, item_p in enumerate(p):
        if len(p)%2 ==0:
            grid = int(len(p)/2)
            plt.subplot(grid, grid, plot_idx)
        else:
            grid = int(len(p) / 2)
            plt.subplot(grid+1, grid, plot_idx)

        plot_idx = plot_idx + 1
        for idx_alpha, item_alpha in enumerate(alpha):
            stack_h = []
            for idx_h, item_h in enumerate(h):
                name = 'alpha:%.5f_h:%s_p:%.3f' % (item_alpha, item_h, item_p)
                stack = load(name, 'Raw', Master_path)
                stack_k = []
                for idx_k, item_k in enumerate(k[idx_p]):

                    A = stack[idx_k][1]

                    stack_k.append(A)
                stack_h.append(stack_k)

            for idx_k,item_k in enumerate(k[idx_p]):
                A = [stack_h[i][idx_k] for i in [0,1]]
                diff = abs(A[0] ** 2 - A[1] ** 2)
                sum = abs(A[0] ** 2 + A[1] ** 2)

                norm = diff / sum

                if norm < 0.1:
                    norm = 0.

                hel[idx_alpha, idx_k] = norm

                color = log(absolute(sqrt(sum))) / t_f

                if color < 0.:
                    color = 0.

                growth[idx_alpha, idx_k] = color
        a_final = a_stack[idx_p][1][-1]
        x = k[idx_p] / a_final
        y = alpha

        x, y = np.meshgrid(x, y)
        plt.contourf(x, y, growth, 20)
        plt.colorbar()
        cs = plt.contour(x, y, hel, levels=[0.5, 0.9], colors=['white', 'black'])
        plt.clabel(cs, fontsize=10, inline=1, fmt='%.1f')

        plt.ylabel(r'$\alpha$')
        plt.xlabel(r'$k/a_f$')
        # plt.colorbar()

    plt.show()

    return


def rho_plot(p, alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix, Phi_Stack):
    h=[-1,1]
    h_style = ['--', '-']
    H_inf = 1e-5

    f_idxs = [0,1]
    p_idxs = [0]
    alpha_idxs = [1, 2, 3,4]
    colors = cm.rainbow(np.linspace(0,1,len(alpha_idxs)))
    color_cnt = 0
    for cnt_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        m = H_inf/item_f
        m =1e-5
        for cnt_p, idx_p, in enumerate(p_idxs):
            item_p = p[idx_p]
            colors_alpha = cm.rainbow(np.linspace(0,1, len(alpha)))
            sol = Phi_Stack[idx_f][idx_p]
            t = sol[0]
            phi = sol[1]
            dphi = sol[2]
            par_H = functools.partial(H, p=item_p, f= item_f)
            args = np.array(list(zip(phi,dphi)))
            H_arr = np.array([*map(par_H, args)])

            color_cnt = 0
            for cnt_alpha, idx_alpha in enumerate(alpha_idxs):
                item_alpha = alpha[idx_alpha]
                name_hel = 'alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                #stack_hel = np.array(load(name_hel, 'Helicity', Master_path));print(np.shape(stack_hel))
                #x_hel = stack_hel[:][0], Hel = stack_hel[:][1]
                for idx_h, item_h in enumerate(h):
                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, item_h, item_p, item_f)
                    name_0 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., item_h, item_p, item_f)
                    t_i = trigger_matrix[idx_f][idx_p]
                    t_f = kill_matrix[idx_f][idx_p];a_f = mapper(t_f, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                    a_map = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in x_B]) / a_i
                    H_map = np.array([mapper(x, t, H_arr) for x in x_B])

                    cross_idx = np.argwhere(np.diff(np.sign(np.absolute(rho_E + rho_B) - H_map ** 2 / m ** 2))).flatten()
                    if len(cross_idx) == 0:
                        eq_idx = -1
                    else:
                        eq_idx = cross_idx[0]

                    stack_B = np.array(load(name, 'Rho_B', Master_path)); stack_B_0 = np.array(load(name_0, 'Rho_B', Master_path))
                    x_B, rho_B = stack_B[:,0], stack_B[:,1]
                    x_B_0, rho_B_0 = stack_B_0[:, 0], stack_B_0[:, 1]

                    stack_E = np.array(load(name, 'Rho_E', Master_path));stack_E_0 = np.array(load(name_0, 'Rho_E', Master_path))
                    x_E, rho_E = stack_E[:, 0], stack_E[:, 1]
                    x_E_0, rho_E_0 = stack_E_0[:, 0], stack_E_0[:, 1]

                    rho_E = rho_E/rho_E_0; rho_B = rho_B/rho_B_0
                    rho = rho_E+rho_B

                    plt.subplot(1,2,1+idx_f)

                plt.loglog(x_B[:eq_idx], rho[:eq_idx], color=colors[color_cnt], linestyle=h_style[idx_h])
                plt.loglog(x_B[eq_idx:], rho[eq_idx:], color=colors[color_cnt], linestyle=h_style[idx_h], alpha = 0.6)

        plt.loglog(x_B, H_map ** 2 / m ** 2, color='k')

    plt.show()
    return


def hel_spooler_2(args):
    [p, alpha, k, a_stack, Phi_stack, Master_path, f, x_len, trigger_matrix, kill_matrix] = args
    h = [-1,1]
    h_style = ['-', '--']
    #[plt.plot([],[],label = 'h:%s'%(h[i]), linestyle=h_style[i]) for i in [0,1]]
    #ax = [ 0, 0]
    for idx_f, item_f in enumerate(f):
        for idx_p, item_p in enumerate(p):
            t_i = trigger_matrix[idx_f][idx_p]
            t_f = kill_matrix[idx_f][idx_p]
            x = np.linspace(t_i, t_f, x_len)
            k = k[idx_f][idx_p]
            for cnt_alpha, idx_alpha in enumerate([-1]):
                item_alpha = alpha[idx_alpha]
                A_h = []
                for idx_h, item_h in enumerate(h):
                    style_h = h_style[idx_h]

                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f'% (item_alpha, item_h, item_p, item_f)
                    stack = load(name, 'Raw', Master_path)

                    A_x = []
                    for subidx_k, idx_k in enumerate(k):
                        item_k = k[idx_k]

                        sub_stack = stack[idx_k]

                        t = sub_stack[0]
                        #phi = sub_stack[1]
                        #dphi = sub_stack[2]
                        A = sub_stack[3]
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
                Integrand = [trapz(1 / (8 * pi ** 2) * k ** 2 * (A_h[0][:][i] ** 2 - A_h[1][:][i] ** 2), x=k) for i in range(x_len)]

                Integrand_2 = [trapz(1 / (8 * pi ** 2) * k ** 2 * (A_h[0][:][i] ** 2 + A_h[1][:][i] ** 2), x=k) for i in range((x_len))]

                norm_int = Integrand/Integrand_2

                plt.plot(x,norm_int, label='f:%.4f alpha:%.2f'%(item_f, item_alpha))

    return


