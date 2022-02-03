from main import *

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')

def potential_plot(p):

    phi = np.linspace(-10,10, 1000)
    par = functools.partial(V, p=p)
    V_arr = np.array([*map(par,phi)])
    label = 'p:%s ' % (p)
    return plt.plot(phi, V_arr, label = label)

def scalar_plot(p, Phi_Stack, a_stack, kill_arr, trigger_arr, k, f):

    colors = cm.rainbow(np.linspace(0, 1, len(p)*len(f)+1));print(len(colors))
    color_cnt = 0

    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
    #ax1 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=4, sharex=ax0)
    #ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)

    for cnt_f, idx_f in enumerate([0,-1]):
        item_f = f[idx_f]
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

            ax0.loglog(t, abs(phi), label=r'$f:%.3f \quad H_{osc}/m:%.3f$' % (item_f, H_osc), color = colors[color_cnt])
            #ax0.loglog([t_kill]*2, [0,2], marker='X', color = colors[color_cnt], linestyle = ':')
            #ax0.loglog([t_trigger] * 2, [0, 2], marker='X', color=colors[color_cnt], linestyle=':')
            ax0.set_ylabel(r'$\phi$')
            ax0.legend()
            #ax0.set_xscale('Log')

            par = functools.partial(V, p=p_i)
            V_arr = np.array([*map(par, phi)])

            par_H = functools.partial(H, p=p_i, f= item_f)
            args = np.array(list(zip(phi,dphi)))
            H_arr = np.array([*map(par_H, args)])

            #ax1.plot(phi, V_arr, label='p:%s' % (p_i), color = colors[color_cnt])
            #ax1.set_xlabel(r'$\phi$')
            #ax1.set_ylabel(r'$V$')

            #ax1.xaxis.set_label_position('top')
            #ax1.yaxis.set_label_position('right')

            #ax2.plot(t, 1/2*(dphi/H_arr) ** 2 / 2)
            #ax2.loglog(t, abs(dphi), color = colors[color_cnt])
            #ax2.set_ylabel(r'$\epsilon$')
            #ax2.set_xlabel(r'$\tilde t$')
            #ax2.set_xscale('Log')
            #ax2.loglog(t, H_arr, linestyle = '--', color = colors[color_cnt])
            #[ax2.loglog(t_a, k[idx_f][idx_p][i]/a, color = colors[color_cnt], linestyle = ':') for i in [0,-1]]

            ax2.semilogx(t_a,log(a/a_i), color = colors[color_cnt])

            ax2.set_ylabel(r'$log(a)$')
            ax2.set_xlabel(r'$\tilde t$')
            #ax2.yaxis.set_label_position('right')

            color_cnt = color_cnt+1

    #ax2.plot([],[],color='k', label = 'H/m', linestyle = '--')
    #ax2.legend()

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
    f_idxs = [0, 12]
    for item_idx_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        for idx_p, item_p in enumerate(p):
            #ax1 = plt.subplot(211)
            #ax2 = plt.subplot(212, sharex=ax1)

            gs = gridspec.GridSpec(3, 1)
            ax1 = plt.subplot(gs[:-1, :])
            ax2 = plt.subplot(gs[-1, :], sharex=ax1)

            for idx_h, item_h in enumerate(h):
                style_h = h_style[idx_h]
                plt.plot([],[],linestyle=h_style[idx_h], label = r'h:%s'%item_h, color='k')
                for cnt_alpha, idx_alpha in enumerate([-1]):
                    item_alpha = alpha[idx_alpha]
                    name = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f'% (item_alpha, item_h, item_p, item_f)
                    stack = load(name, 'Raw', Master_path)
                    plt_k_idxs = np.round(np.linspace(350, len(k[idx_f][idx_p])-450 , 2)).astype(int);plt_k_idxs = np.round(np.linspace(550, 600 , 2)).astype(int)
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
                        a = a_stack[idx_f][idx_p, 1];print(item_k/a[np.argmax(abs(dphi))])

                        #print(len(t))
                        #plt.subplot(2,1,1)#+idx_h)#+2*count_alpha)
                        ax1.loglog(t, absolute(A), label = r'$ k:%.0f$'%(idx_k), color = color[subidx_k], linestyle=h_style[idx_h])
                        #plt.legend()
                        if subidx_k==0:
                        #plt.subplot(2,1,2)#3+idx_h)
                            ax2.loglog(t, absolute(dphi), label=r'$ k:%.2f$' % (item_k), color='green')
                        #ax2.loglog(t_a, absolute(item_k/a), color=color[subidx_k], linestyle = ':')

            #plt.title(r'p:%.1f' % item_p)
            ax2.set_xlabel(r'$\tilde{t}$');ax1.set_title(r'$f/{\rm Mpl}:%.3f$'%item_f)
            ax1.set_ylabel(r'$|\sqrt{(2k)}\mathcal{A}_\pm|$');ax2.set_ylabel(r'$\frac{{\rm d}{\tilde\phi}}{{\rm d}{\tilde{t}}}$', rotation=0,labelpad=20, fontsize=30)

            #ax[0].get_shared_x_axes().join(ax[0], ax[1])
            plt.show()

    return


def H_osc_plot(p, Phi_Stack, a_stack, kill_arr, trigger_arr, k, f):

    p_idxs = [0,11,23]
    colors = cm.rainbow(np.linspace(0, 1, len(p_idxs)))
    color_cnt = 0
    [plt.scatter([],[],color = x, label = 'p:%.1f'%p_i) for (x, p_i) in zip(colors, p[p_idxs])]
    for idx_f, item_f in enumerate(f):

        for cnt_p, idx_p in enumerate(p_idxs):#np.arange(0,len(p),1):
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
            #ax0.set_xscale('Log')

            par = functools.partial(V, p=p_i)
            V_arr = np.array([*map(par, phi)])

            par_H = functools.partial(H, p=p_i, f= item_f)
            args = np.array(list(zip(phi,dphi)))
            H_arr = np.array([*map(par_H, args)])
            print(idx_p);plt.scatter( item_f, H_osc, color = colors[cnt_p] )

    plt.legend()
    plt.show()

    return



def asymp_plot(args, stack):
    [p, alpha, k, a_stack, Phi_stack, f, kill_arr] = args
    h = [-1,1]
    h_style = ['-', '--']
    f_idxs = [3, 7]
    targ_f = [0.01,0.001]
    targ_f = [0.05]

    f_idxs = [np.argmin(abs(targ - f)) for targ in targ_f]
    p_idxs = [0]
    alpha_idxs = [0,4,5]
    targ_alpha = [0.5,0.6,0.7, 0.8]
    targ_alpha = [0.6,1.,2,3]
    alpha_idxs = [np.argmin(abs(targ-alpha)) for targ in targ_alpha]

    colors = cm.rainbow(np.linspace(0,1,len(alpha_idxs)))
    targ_f = [0.01,0.001]    
    f_idxs = [np.argmin(abs(targ-f)) for targ in targ_f]
    
    for cnt_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        #[plt.plot([],[],label = 'h:%s'%(h[i]), linestyle=h_style[i]) for i in [0,1]]
        plt.subplot(1,2,1+cnt_f)

        plt.plot([],[],label = r'$\alpha$', linestyle = '')

        for cnt_alpha, idx_alpha in enumerate(alpha_idxs):
            item_alpha = alpha[idx_alpha];print(cnt_alpha, item_alpha)
            plt.plot([],[],label = '%.1f'%(item_alpha), color = colors[cnt_alpha])
            for cnt_p, idx_p in enumerate(p_idxs):
                item_p = p[idx_p]
                t_kill = kill_arr[idx_f][idx_p]
                a = a_stack[idx_f][idx_p, 1]
                t_a = a_stack[idx_f][idx_p, 0]
                a_f = mapper(t_kill, t_a , a)
                
                t = Phi_stack[idx_f][idx_p][0]
                max_dphi = np.argmax(abs(Phi_stack[idx_f][idx_p][2]))
                t_max = t[max_dphi]
                
                a_osc = mapper(t_max, t_a, a)

                for idx_h, item_h in enumerate(h):
                    style_h = h_style[idx_h]
                    colors_cnt = 0
                    sub_stack = stack[idx_f, idx_h, idx_p, idx_alpha, :]
                    A_0 = stack[idx_f, idx_h, idx_p, 0, :][:,1]
                    print(np.shape(sub_stack))
                    k = sub_stack[:,0]
                    A = sub_stack[:,1]
                    dA = sub_stack[:,2]
                    plt.loglog(k/a_osc, absolute(A)/absolute(A_0),
                               linestyle=style_h, color = colors[cnt_alpha])
                    colors_cnt = colors_cnt+1
                    plt.title(r'$f/{\rm Mpl}:%.4f$'%item_f)
                    plt.xlabel(r'$k(ma_{\rm osc})^{-1}$')
                    plt.ylabel(r'$\sqrt{2k}|\mathcal{A}_h(k, \tilde{t}_f)|$')
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

                norm = diff

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

def rho_plot(p, alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix, Phi_Stack, asymp_stack):
    h=[-1,1]

    H_inf = 1e-8
    targ_f = [0.01]
    
    f_idxs = [np.argmin(abs(targ-f)) for targ in targ_f]
    #f_idxs = [6,12]
    p_idxs = [0]
    targ_alpha = [0.6, 0.7, 0.8]#; targ_alpha = alpha
    #alpha_idxs=range(len(alpha))
    alpha_idxs = [np.argmin(abs(targ-alpha)) for targ in targ_alpha]
    f_idxs = [np.argmin(abs(targ-f)) for targ in targ_f]
    colors = cm.rainbow(np.linspace(0,1,len(alpha_idxs)));colors=['blue','green','red','darkviolet']
    color_cnt = 0
    T_0 = 23.5*1e-14
    Mpl = 2.435 * 1e18  ##Mpl in GeV
    GeV2 = 6.8 * 1e20  ##Gev^2 in Gauss
    T_R = 1e-13;T_R = (90/(100*pi**2))**(1/2)*(H_inf)**(1/2) 
    for cnt_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        #m = H_inf/item_f
        #m =1e-5
        plt.subplot(2,len(targ_f),1+cnt_f)
        plt.xlabel(r'$mt$')
        plt.ylabel(r'$\rho_{\rm em}/m^4$')
        for cnt_p, idx_p, in enumerate(p_idxs):
            item_p = p[idx_p]
            colors_alpha = cm.rainbow(np.linspace(0,1, len(alpha)))
            sol = Phi_Stack[idx_f][idx_p]
            t = sol[0];phi_a = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in t])/mapper(t[0], a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
            phi = sol[1]
            dphi = sol[2]
            par_H = functools.partial(H, p=item_p, f= item_f)
            args = np.array(list(zip(dphi,phi)))
            H_arr = np.array([*map(par_H, args)])
            par_dV = functools.partial(dV, p=item_p)
            #m = H_inf/item_f
            #dV_map = np.array([*map(par_dV, phi)])*item_f/m**2
            id_max = np.argmax(abs(dphi))
            #plt.plot([t[id_max]]*2, [1e-5,1e5], linestyle=':', color = 'grey') ##Onset of oscillation

            color_cnt = 0
            for cnt_alpha, idx_alpha in enumerate(alpha_idxs):

                #plt.subplot(1,2,1+cnt_alpha)
                #plt.xlabel(r'$mt$')
                #plt.ylabel(r'$\rho_{\rm em}/m^4$')

                item_alpha = alpha[idx_alpha]
                plt.plot([],[], color = colors[cnt_alpha], label = r'$\alpha:%.2f$'%(item_alpha))

                #name_hel = 'alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                #name_hel = 'resolved alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                #stack_hel = np.array(load(name_hel, 'Helicity', Master_path));print(np.shape(stack_hel))
                #x_hel = stack_hel[:][0], Hel = stack_hel[:][1]
                #for idx_h, item_h in enumerate(h):
                name_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, -1, item_p, item_f)
                name_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, 1, item_p, item_f)                
                name_0_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., -1, item_p, item_f)
                name_0_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., 1, item_p, item_f)

                #x_hel = [stack_hel[i][0] for i in np.arange(0,len(stack_hel),1)]; Hel = [stack_hel[i][1] for i in np.arange(0,len(stack_hel),1)]
                #x_hel = stack_hel[0,:];Hel=stack_hel[1,:]
                
                stack_B_1, stack_B_2 = np.array(load(name_1, 'Rho_B', Master_path)), np.array(load(name_2, 'Rho_B', Master_path))
                stack_B_0_1, stack_B_0_2 =np.array(load(name_0_1, 'Rho_B', Master_path)),np.array(load(name_0_2, 'Rho_B', Master_path))                

                x_B = stack_B_1[:,0]
                    
                x_B_0_1, rho_B_0_1 = stack_B_0_1[:, 0], stack_B_0_1[:, 1]
                x_B_0_2, rho_B_0_2 = stack_B_0_2[:, 0], stack_B_0_2[:, 1]
                rho_B = stack_B_1[:,1]/rho_B_0_1 + stack_B_2[:,1]/rho_B_0_2#*m**2/H_map**2/(a_i**4)
                    
                stack_E_1 = np.array(load(name_1, 'Rho_E', Master_path));stack_E_0_1 = np.array(load(name_0_1, 'Rho_E', Master_path))
                stack_E_2 = np.array(load(name_2, 'Rho_E', Master_path));stack_E_0_2 = np.array(load(name_0_2, 'Rho_E', Master_path))
                x_E_0_1, rho_E_0_1 = stack_E_0_1[:, 0], stack_E_0_1[:, 1]
                x_E_0_2, rho_E_0_2 = stack_E_0_2[:, 0], stack_E_0_2[:, 1]
                rho_E = stack_E_1[:, 1]/rho_E_0_1 + stack_E_2[:, 1]/rho_E_0_2#*m**2/H_map**2/(a_i**2)

                H_map = np.array([mapper(x, t, H_arr) for x in x_B])

                m = H_inf/item_f*(3*(0.5*absolute(dphi[0])**2 + V(phi[0], item_p))**(-1))**(1/2)

                H_map = sqrt(item_f**2*(0.5*dphi**2+np.array([V(z,item_p) for z in phi])))/sqrt((0.5*dphi[0]**2+V(phi[0],item_p))); H_map=np.array([mapper(x, t, H_map) for x in x_B])
                t_i = trigger_matrix[idx_f][idx_p]
                a_i = mapper(t_i, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                a_map = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in x_B])/a_i; a_map = 1./a_i#print(np.shape(rho_B),np.shape(rho_E),np.shape(H_map))#a_map = 1/a_i
                cross_idx = np.argwhere(np.diff(np.sign(np.absolute(rho_E+rho_B) - 3*H_map**2/m**2 ))).flatten()
                if len(cross_idx) == 0:
                    eq_idx = -1
                else:
                    eq_idx = cross_idx[0]
                dV_map = np.array([mapper(x, phi, t) for x in x_B])

                #stack_Bck_0 = [absolute(np.array(load(name_0_1, 'Backreaction', Master_path))[:,1]), absolute(np.array(load(name_0_2, 'Backreaction', Master_path))[:,1])]
                #stack_Bck = abs((absolute(np.array(load(name_1, 'Backreaction', Master_path))[:,1])/stack_Bck_0[0] - absolute(np.array(load(name_2, 'Backreaction', Master_path))[:,1])/stack_Bck_0[0]))/abs(dV_map*a_map**3)

                #plt.semilogy(x_B, stack_Bck, color = colors[color_cnt], linestyle=':')

                t_i = trigger_matrix[idx_f][idx_p]
                t_f = kill_matrix[idx_f][idx_p];a_f = mapper(t_f, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])

                
                A_0 = [absolute(asymp_stack[idx_f, idx_h, idx_p, 0, :, 1]) for idx_h in [0,1]]
                A = [absolute(asymp_stack[idx_f, idx_h, idx_p, idx_alpha, :, 1])-absolute(A_0[idx_h]) for idx_h in [0,1]]

                max_idx = np.argmax(A[0])
                max_k, max_A = [k[idx_f][idx_p][max_idx], sqrt(absolute(A[0][max_idx])**2+absolute(A[1][max_idx])**2)]

                max_A0 = sqrt(absolute(A_0[0][max_idx])**2+absolute(A_0[1][max_idx])**2)

                #plt.subplot(2,2,1+cnt_f)
                plt.title(r'$f/{\rm Mpl}:%.4f$'%item_f)
                #plt.semilogy(x_B-t_i, absolute(rho_E/rho_E_0*m**2/H_map**2)/(a_i**2), color =colors[color_cnt], linestyle=['-','--'][idx_h])
                #plt.loglog(x_B-t_i, ro_B+rho_E, color=colors[color_cnt], linestyle=h_style[idx_h])
                rho_B_1 = (absolute(stack_B_1[:,1])-absolute(rho_B_0_1))*m**4*Mpl**4;rho_E_1 = (absolute(stack_E_1[:,1])-absolute(rho_E_0_1))*m**4*Mpl**4#*m**2/H_map**2
                rho_B_2 = (absolute(stack_B_2[:,1])-absolute(rho_B_0_2))*m**4*Mpl**4;rho_E_2 = (absolute(stack_E_2[:,1])-absolute(rho_E_0_2))*m**4*Mpl**4#*m**2/H_map**2
                
                plt.subplot(2,1,2)
                plt.semilogy(x_B, abs(rho_B_1-rho_B_2)/abs(rho_B_1+rho_B_2), color = colors[color_cnt])
                plt.subplot(2,len(targ_f),1+cnt_f)


                ratio = (((rho_B_1+rho_B_2)/a_map**4+(rho_E_1+rho_E_2)/a_map**2)*m**2/(3*H_map**2))[-1]
                print(r'$\alpha:$', item_alpha, 'f:',item_f, 'Ratio:%.5f'%ratio)
                a_R = T_0 / (T_R * Mpl)#; a_R = 1e-29  ## a_rT_r = a_0T_r

                if ratio <2:
                    print(r'$B_{gen}:%.4e$' %(sqrt(rho_B_1+rho_B_2)*m**2*Mpl**2*a_map**-2)[-1])
                    print(r'$\Lambda_{gen}:%.4e \quad \Lambda_{gen, peak}:%.4e$' %(2/(m*Mpl), (a_f/(max_k*m*Mpl))))
                    B_k_f = (m ** 2 * (max_k/(a_f/a_i)) ** 2 * (max_A / max_A0)) * Mpl ** 2; print(r'$B_{gen, peak}:%.4e$' %(B_k_f)); 
                    
                    Lambda_k_f=2/(m*Mpl); Lambda_k_f = (a_f/(max_k*m*Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_k_f) ** (1 / 3) * (B_k_f) ** (2 / 3); print(r'$\Lambda_{m,0}:%.4e$' %(Lambda_0))
                    B_0 = 1e-8 * Lambda_0;print(r'$B_{0, peak}:%.4e$' %(B_0)) 
                #a_R = 1e-29 ##Subhramanian entropy conservation
                cross_idx_1 = np.argwhere(np.diff(np.sign(np.absolute(rho_E_1/a_map**2+rho_B_1/a_map**4) - 3*H_map**2*m**2*Mpl**4 ))).flatten()
                cross_idx_2 = np.argwhere(np.diff(np.sign(np.absolute(rho_E_2/a_map**2+rho_B_2/a_map**4) - 3*H_map**2*m**2*Mpl**4 ))).flatten()

                if len(cross_idx_1)==0:
                    cross_idx_1=-1
                    plt.semilogy(x_B[:cross_idx_1], (rho_B_1/a_map**4+rho_E_1/a_map**2)[:cross_idx_1], color=colors[color_cnt], linestyle=':')
                    plt.semilogy(x_B[cross_idx_1:], (rho_B_1/a_map**4+rho_E_1/a_map**2)[cross_idx_1:], color=colors[color_cnt], linestyle=':',alpha=0.5)
                    
                    B_k_f = sqrt((rho_B_1/a_map**4)[cross_idx_1])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl); Lambda_f = (a_f/(max_k*m*Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)


                else:
                    cross_idx_1 =cross_idx_1[0]
                    plt.semilogy(x_B[:cross_idx_1+1], (rho_B_1/a_map**4+rho_E_1/a_map**2)[:cross_idx_1+1], color=colors[color_cnt], linestyle=':')
                    plt.semilogy(x_B[cross_idx_1:], (rho_B_1/a_map**4+rho_E_1/a_map**2)[cross_idx_1:], color=colors[color_cnt], linestyle=':',alpha=0.5)

                    B_k_f = sqrt((rho_B_1/a_map**4)[cross_idx_1])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl); Lambda_f = (a_f/(max_k*m*Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)

                if len(cross_idx_2)==0:
                    cross_idx_2=-1
                    plt.semilogy(x_B[:cross_idx_2], (rho_B_2/a_map**4+rho_E_2/a_map**2)[:cross_idx_2], color=colors[color_cnt], linestyle='-')
                    plt.semilogy(x_B[cross_idx_2:], (rho_B_2/a_map**4+rho_E_2/a_map**2)[cross_idx_2:], color=colors[color_cnt], linestyle='-',alpha=0.5)
                    
                    B_k_f = sqrt((rho_B_2/a_map**4)[cross_idx_2])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl); Lambda_f = (a_f/(max_k*m*Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0
                    
                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)

                else:
                    cross_idx_2 =cross_idx_2[0]
                    plt.semilogy(x_B[:cross_idx_2+1], (rho_B_2/a_map**4+rho_E_2/a_map**2)[:cross_idx_2+1], color=colors[color_cnt], linestyle='-')
                    plt.semilogy(x_B[cross_idx_2:], (rho_B_2/a_map**4+rho_E_2/a_map**2)[cross_idx_2:], color=colors[color_cnt], linestyle='-',alpha=0.5)


                    B_k_f = sqrt((rho_B_2/a_map**4)[cross_idx_2])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl); Lambda_f = (a_f/(max_k*m*Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0
                    
                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)


                #plt.loglog(x_B, rho_B_1/a_map**4+rho_E_1/a_map**2, color=colors[color_cnt], linestyle='--')
                #plt.loglog(x_B, rho_B_2/a_map**4+rho_E_2/a_map**2, color=colors[color_cnt], linestyle='-')                
                #plt.semilogy(x_hel, abs(absolute(np.array(Hel))), color = colors[color_cnt], label=r'$\alpha:%.0f$'%(item_alpha))
                color_cnt = color_cnt + 1
        plt.semilogy(x_B, 3*H_map**2*m**2*Mpl**4,color='k')#;plt.semilogy(t, 3*H_map_T**2/m**2,color='k', linestyle = '--')
        plt.ylim(1e-5,1e5)
    plt.legend()
    plt.show()
    return


def rho_hel_plot(p, alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix, Phi_Stack, asymp_stack):
    h = [-1, 1]

    H_inf = 1e-8
    targ_f = [0.05, 0.01, 0.005]
    targ_f = [0.05]
    f_idxs = [np.argmin(abs(targ - f)) for targ in targ_f]
    # f_idxs = [6,12]
    p_idxs = [0]
    targ_alpha = [1, 1.2,1.3, 1.45]  # ; targ_alpha = alpha
    targ_alpha = [1.2,1.4,1.6,1.8,2,2.6,2.8]
    # alpha_idxs=range(len(alpha))
    alpha_idxs = [np.argmin(abs(targ - alpha)) for targ in targ_alpha]
    f_idxs = [np.argmin(abs(targ - f)) for targ in targ_f]
    colors = cm.rainbow(np.linspace(0, 1, len(alpha_idxs)));
    #colors = ['blue', 'green', 'red', 'darkviolet']
    color_cnt = 0
    T_0 = 23.5 * 1e-14
    Mpl = 2.435 * 1e18  ##Mpl in GeV
    GeV2 = 6.8 * 1e20  ##Gev^2 in Gauss
    T_R = 1e-13;
    T_R = (90 / (100 * pi ** 2)) ** (1 / 2) * (H_inf) ** (1 / 2)

    for cnt_f, idx_f in enumerate(f_idxs):

        ax0 = plt.subplot2grid((3, len(f_idxs)), (0, cnt_f), colspan=1, rowspan = 2)
        ax1 = plt.subplot2grid((3, len(f_idxs)), (2, cnt_f), colspan=1, sharex=ax0)

        item_f = f[idx_f]
        # m = H_inf/item_f
        # m =1e-5
        #plt.subplot(2, len(targ_f), 1 + cnt_f)
        ax1.set_xlabel(r'$mt$')
        ax0.set_ylabel(r'$\rho_{\rm em}/m^4$')
        for cnt_p, idx_p, in enumerate(p_idxs):
            item_p = p[idx_p]
            colors_alpha = cm.rainbow(np.linspace(0, 1, len(alpha)))
            sol = Phi_Stack[idx_f][idx_p]
            t = sol[0];
            phi_a = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in t]) / mapper(t[0],
                                                                                                                  a_stack[
                                                                                                                      idx_f][
                                                                                                                      idx_p][
                                                                                                                      0],
                                                                                                                  a_stack[
                                                                                                                      idx_f][
                                                                                                                      idx_p][
                                                                                                                      1])
            phi = sol[1]
            dphi = sol[2]
            par_H = functools.partial(H, p=item_p, f=item_f)
            args = np.array(list(zip(dphi, phi)))
            H_arr = np.array([*map(par_H, args)])
            par_dV = functools.partial(dV, p=item_p)
            # m = H_inf/item_f
            # dV_map = np.array([*map(par_dV, phi)])*item_f/m**2
            id_max = np.argmax(abs(dphi))
            # plt.plot([t[id_max]]*2, [1e-5,1e5], linestyle=':', color = 'grey') ##Onset of oscillation

            color_cnt = 0
            for cnt_alpha, idx_alpha in enumerate(alpha_idxs):

                # plt.subplot(1,2,1+cnt_alpha)
                # plt.xlabel(r'$mt$')
                # plt.ylabel(r'$\rho_{\rm em}/m^4$')

                item_alpha = alpha[idx_alpha]
                #ax0.plot([], [], color=colors[cnt_alpha], label=r'$\alpha:%.2f$' % (item_alpha))

                # name_hel = 'alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                name_hel = 'resolved alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                stack_hel = np.array(load(name_hel, 'Helicity', Master_path))#;print(np.shape(stack_hel));exit()
                # x_hel = stack_hel[:][0], Hel = stack_hel[:][1]
                # for idx_h, item_h in enumerate(h):
                name_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, -1, item_p, item_f)
                name_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, 1, item_p, item_f)
                name_0_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., -1, item_p, item_f)
                name_0_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., 1, item_p, item_f)

                # x_hel = [stack_hel[i][0] for i in np.arange(0,len(stack_hel),1)]; Hel = [stack_hel[i][1] for i in np.arange(0,len(stack_hel),1)]
                x_hel = stack_hel[0,:];Hel=stack_hel[1,:]

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

                H_map = np.array([mapper(x, t, H_arr) for x in x_B])

                m = H_inf / item_f * (3 * (0.5 * absolute(dphi[0]) ** 2 + V(phi[0], item_p)) ** (-1)) ** (1 / 2)

                H_map = sqrt(item_f ** 2 * (0.5 * dphi ** 2 + np.array([V(z, item_p) for z in phi]))) / sqrt(
                    (0.5 * dphi[0] ** 2 + V(phi[0], item_p)));
                H_map = np.array([mapper(x, t, H_map) for x in x_B])
                t_i = trigger_matrix[idx_f][idx_p]
                a_i = mapper(t_i, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                a_map = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in x_B]) / a_i;
                a_map = 1. / a_i  # print(np.shape(rho_B),np.shape(rho_E),np.shape(H_map))#a_map = 1/a_i
                cross_idx = np.argwhere(
                    np.diff(np.sign(np.absolute(rho_E + rho_B) - 3 * H_map ** 2 / m ** 2))).flatten()
                if len(cross_idx) == 0:
                    eq_idx = -1
                else:
                    eq_idx = cross_idx[0]
                dV_map = np.array([mapper(x, phi, t) for x in x_B])

                # stack_Bck_0 = [absolute(np.array(load(name_0_1, 'Backreaction', Master_path))[:,1]), absolute(np.array(load(name_0_2, 'Backreaction', Master_path))[:,1])]
                # stack_Bck = abs((absolute(np.array(load(name_1, 'Backreaction', Master_path))[:,1])/stack_Bck_0[0] - absolute(np.array(load(name_2, 'Backreaction', Master_path))[:,1])/stack_Bck_0[0]))/abs(dV_map*a_map**3)

                # plt.semilogy(x_B, stack_Bck, color = colors[color_cnt], linestyle=':')

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

                # plt.subplot(2,2,1+cnt_f)
                ax0.set_title(r'$f/{\rm Mpl}:%.4f$' % item_f)
                # plt.semilogy(x_B-t_i, absolute(rho_E/rho_E_0*m**2/H_map**2)/(a_i**2), color =colors[color_cnt], linestyle=['-','--'][idx_h])
                # plt.loglog(x_B-t_i, ro_B+rho_E, color=colors[color_cnt], linestyle=h_style[idx_h])
                rho_B_1 = (absolute(stack_B_1[:, 1]) - absolute(rho_B_0_1)) * m ** 4 * Mpl ** 4;
                rho_E_1 = (absolute(stack_E_1[:, 1]) - absolute(rho_E_0_1)) * m ** 4 * Mpl ** 4*0  # *m**2/H_map**2
                rho_B_2 = (absolute(stack_B_2[:, 1]) - absolute(rho_B_0_2)) * m ** 4 * Mpl ** 4;
                rho_E_2 = (absolute(stack_E_2[:, 1]) - absolute(rho_E_0_2)) * m ** 4 * Mpl ** 4*0  # *m**2/H_map**2

                #plt.subplot(2, 1, 2)
                #plt.semilogy(x_B, abs(rho_B_1 - rho_B_2) / abs(rho_B_1 + rho_B_2), color=colors[color_cnt])
                #plt.subplot(2, len(targ_f), 1 + cnt_f)

                ratio = \
                (((rho_B_1 + rho_B_2) / a_map ** 4 + (rho_E_1 + rho_E_2) / a_map ** 2) * m ** 2 / (3 * H_map ** 2))[-1]
                print(r'$\alpha:$', item_alpha, 'f:', item_f, 'Ratio:%.5f' % ratio)
                a_R = T_0 / (T_R * Mpl)  # ; a_R = 1e-29  ## a_rT_r = a_0T_r

                if ratio < 2:
                    print(r'$B_{gen}:%.4e$' % (sqrt(rho_B_1 + rho_B_2) * m ** 2 * Mpl ** 2 * a_map ** -2)[-1])
                    print(r'$\Lambda_{gen}:%.4e \quad \Lambda_{gen, peak}:%.4e$' % (
                    2 / (m * Mpl), (a_f / (max_k * m * Mpl))))
                    B_k_f = (m ** 2 * (max_k / (a_f / a_i)) ** 2 * (max_A / max_A0)) * Mpl ** 2;
                    print(r'$B_{gen, peak}:%.4e$' % (B_k_f));

                    Lambda_k_f = 2 / (m * Mpl);
                    Lambda_k_f = (a_f / (max_k * m * Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_k_f) ** (1 / 3) * (B_k_f) ** (2 / 3);
                    print(r'$\Lambda_{m,0}:%.4e$' % (Lambda_0))
                    B_0 = 1e-8 * Lambda_0;
                    print(r'$B_{0, peak}:%.4e$' % (B_0))
                    # a_R = 1e-29 ##Subhramanian entropy conservation
                cross_idx_1 = np.argwhere(np.diff(np.sign(np.absolute(
                    rho_E_1 / a_map ** 2 + rho_B_1 / a_map ** 4) - 3 * H_map ** 2 * m ** 2 * Mpl ** 4))).flatten()
                cross_idx_2 = np.argwhere(np.diff(np.sign(np.absolute(
                    rho_E_2 / a_map ** 2 + rho_B_2 / a_map ** 4) - 3 * H_map ** 2 * m ** 2 * Mpl ** 4))).flatten()

                if len(cross_idx_1) == 0:
                    cross_idx_1 = -1
                    ax0.semilogy(x_B[:cross_idx_1], (rho_B_1 / a_map ** 4 + rho_E_1 / a_map ** 2)[:cross_idx_1],
                                 color=colors[color_cnt], linestyle=':')
                    ax0.semilogy(x_B[cross_idx_1:], (rho_B_1 / a_map ** 4 + rho_E_1 / a_map ** 2)[cross_idx_1:],
                                 color=colors[color_cnt], linestyle=':', alpha=0.5)

                    ax1.plot(x_hel[:cross_idx_1], Hel[:cross_idx_1], color=colors[color_cnt])
                    ax1.plot(x_hel[cross_idx_1:], Hel[cross_idx_1:], color=colors[color_cnt], alpha = 0.5, linestyle = ':')

                    B_k_f = sqrt((rho_B_1 / a_map ** 4)[cross_idx_1]) * (m * Mpl) ** 2
                    Lambda_f = 2 / (m * Mpl);
                    Lambda_f = (a_f / (max_k * m * Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:', item_alpha, 'f', item_f, Lambda_0, B_0)


                else:
                    cross_idx_1 = cross_idx_1[0]
                    ax0.semilogy(x_B[:cross_idx_1 + 1], (rho_B_1 / a_map ** 4 + rho_E_1 / a_map ** 2)[:cross_idx_1 + 1],
                                 color=colors[color_cnt], linestyle=':')
                    ax0.semilogy(x_B[cross_idx_1:], (rho_B_1 / a_map ** 4 + rho_E_1 / a_map ** 2)[cross_idx_1:],
                                 color=colors[color_cnt], linestyle=':', alpha=0.5)

                    ax1.plot(x_hel[:cross_idx_1], Hel[:cross_idx_1], color=colors[color_cnt])
                    ax1.plot(x_hel[cross_idx_1:], Hel[cross_idx_1:], color=colors[color_cnt], alpha = 0.5, linestyle = ':')

                    B_k_f = sqrt((rho_B_1 / a_map ** 4)[cross_idx_1]) * (m * Mpl) ** 2
                    Lambda_f = 2 / (m * Mpl);
                    Lambda_f = (a_f / (max_k * m * Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:', item_alpha, 'f', item_f, Lambda_0, B_0)

                if len(cross_idx_2) == 0:
                    cross_idx_2 = -1
                    ax0.semilogy(x_B[:cross_idx_2], (rho_B_2 / a_map ** 4 + rho_E_2 / a_map ** 2)[:cross_idx_2],
                                 color=colors[color_cnt], linestyle='-')
                    ax0.semilogy(x_B[cross_idx_2:], (rho_B_2 / a_map ** 4 + rho_E_2 / a_map ** 2)[cross_idx_2:],
                                 color=colors[color_cnt], linestyle='-', alpha=0.5)

                    B_k_f = sqrt((rho_B_2 / a_map ** 4)[cross_idx_2]) * (m * Mpl) ** 2
                    Lambda_f = 2 / (m * Mpl);
                    Lambda_f = (a_f / (max_k * m * Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:', item_alpha, 'f', item_f, Lambda_0, B_0)

                else:
                    cross_idx_2 = cross_idx_2[0]
                    ax0.semilogy(x_B[:cross_idx_2 + 1], (rho_B_2 / a_map ** 4 + rho_E_2 / a_map ** 2)[:cross_idx_2 + 1],
                                 color=colors[color_cnt], linestyle='-')
                    ax0.semilogy(x_B[cross_idx_2:], (rho_B_2 / a_map ** 4 + rho_E_2 / a_map ** 2)[cross_idx_2:],
                                 color=colors[color_cnt], linestyle='-', alpha=0.5)

                    B_k_f = sqrt((rho_B_2 / a_map ** 4)[cross_idx_2]) * (m * Mpl) ** 2
                    Lambda_f = 2 / (m * Mpl);
                    Lambda_f = (a_f / (max_k * m * Mpl))
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:', item_alpha, 'f', item_f, Lambda_0, B_0)

                # plt.loglog(x_B, rho_B_1/a_map**4+rho_E_1/a_map**2, color=colors[color_cnt], linestyle='--')
                # plt.loglog(x_B, rho_B_2/a_map**4+rho_E_2/a_map**2, color=colors[color_cnt], linestyle='-')
                # plt.semilogy(x_hel, abs(absolute(np.array(Hel))), color = colors[color_cnt], label=r'$\alpha:%.0f$'%(item_alpha))



                color_cnt = color_cnt + 1
                
                
                
        ax0.semilogy(x_B, 3 * H_map ** 2 * m ** 2 * Mpl ** 4,
                     color='k')  # ;plt.semilogy(t, 3*H_map_T**2/m**2,color='k', linestyle = '--')
        ax0.set_ylim(1e50, 1e70)
    ax1.set_ylabel(r'$h_B$')
    plt.legend()
    plt.show()
    return


def end_spooler(gauge_stack, t_i, t_f, x_len, a_stack, idx_p, idx_f,k):
    stack_x = []
    for x in np.linspace(t_i, t_f, x_len):
        Shredder_args = [x,k[idx_f][idx_p], a_stack, idx_p, idx_f]
        Rho_B = Shredder_B(Shredder_args, gauge_stack)
        Rho_E = Shredder_E(Shredder_args, gauge_stack)
        stack_x.append([x, Rho_B, Rho_E])
    stack_x = np.array(stack_x)
    return stack_x



def rho_plot_spec(p, alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix, Phi_Stack):
    h=[-1,1]
    start = time.time()
    H_inf = 1e-5
    targ_f = [0.05]
    
    f_idxs = [np.argmin(abs(targ-f)) for targ in targ_f]
    #f_idxs = [6,12]
    p_idxs = [0]
    targ_alpha = [0.55]
    #alpha_idxs = [1,2,3,4,5,6];alpha_idxs=range(len(alpha))
    alpha_idxs = [np.argmin(abs(targ-alpha)) for targ in targ_alpha]
    f_idxs = [np.argmin(abs(targ-f)) for targ in targ_f]
    colors = cm.rainbow(np.linspace(0,1,len(alpha_idxs)))
    color_cnt = 0
    T_0 = 23.5*1e-14
    Mpl = 2.435 * 1e18  ##Mpl in GeV
    GeV2 = 6.8 * 1e20  ##Gev^2 in Gauss
    T_R = 1e-13;T_R = (90/(100*pi**2))**(1/2)*(H_inf)**(1/2) 
    for cnt_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        m = H_inf/item_f
        #m =1e-5
        plt.subplot(1,2,1+cnt_f)
        plt.xlabel(r'$mt$')
        plt.ylabel(r'$\rho_{\rm em}/m^4$')
        for cnt_p, idx_p, in enumerate(p_idxs):
            item_p = p[idx_p]
            colors_alpha = cm.rainbow(np.linspace(0,1, len(alpha)))
            sol = Phi_Stack[idx_f][idx_p]
            t = sol[0];phi_a = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in t])/mapper(t[0], a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
            phi = sol[1]
            dphi = sol[2]
            par_H = functools.partial(H, p=item_p, f= item_f)
            args = np.array(list(zip(phi,dphi)))
            H_arr = np.array([*map(par_H, args)])
            par_dV = functools.partial(dV, p=item_p)
            m = H_inf/item_f
            dV_map = np.array([*map(par_dV, phi)])*item_f/m**2

            color_cnt = 0
            for cnt_alpha, idx_alpha in enumerate(alpha_idxs):

                #plt.subplot(1,2,1+cnt_alpha)
                #plt.xlabel(r'$mt$')
                #plt.ylabel(r'$\rho_{\rm em}/m^4$')

                item_alpha = alpha[idx_alpha]
                plt.plot([],[], color = colors[cnt_alpha], label = r'$\alpha:%.2f$'%(item_alpha))

                #name_hel = 'alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                #name_hel = 'resolved alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                #stack_hel = np.array(load(name_hel, 'Helicity', Master_path));print(np.shape(stack_hel))
                #x_hel = stack_hel[:][0], Hel = stack_hel[:][1]
                #for idx_h, item_h in enumerate(h):
                name_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, -1, item_p, item_f)
                name_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, 1, item_p, item_f)                
                name_0_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., -1, item_p, item_f)
                name_0_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., 1, item_p, item_f)

                #x_hel = [stack_hel[i][0] for i in np.arange(0,len(stack_hel),1)]; Hel = [stack_hel[i][1] for i in np.arange(0,len(stack_hel),1)]
                #x_hel = stack_hel[0,:];Hel=stack_hel[1,:]

                t_i = trigger_matrix[idx_f][idx_p]
                t_f = kill_matrix[idx_f][idx_p]
                
                gauge_stack_1 = load(name_1,'Raw', Master_path);stack_1 = np.array(end_spooler(gauge_stack_1, t_i, t_f, x_len, a_stack, idx_p, idx_f,k))
                gauge_stack_2= load(name_2,'Raw', Master_path);stack_2 = np.array(end_spooler(gauge_stack_2, t_i, t_f, x_len, a_stack, idx_p, idx_f,k))
                gauge_stack_0_1 = load(name_0_1,'Raw', Master_path);stack_0_1 = np.array(end_spooler(gauge_stack_0_1, t_i, t_f, x_len, a_stack, idx_p, idx_f,k))
                gauge_stack_0_2= load(name_0_2,'Raw', Master_path);stack_0_2 = np.array(end_spooler(gauge_stack_0_2, t_i, t_f, x_len, a_stack, idx_p, idx_f,k))
                gauge_stack_1 = [];gauge_stack_2 = [];gauge_stack_0_1 = [];gauge_stack_0_2 = []
                
                timer('Counter %.0f'%(idx_alpha), start)

                #stack_B_1, stack_B_2 = np.array(load(name_1, 'Rho_B', Master_path)), np.array(load(name_2, 'Rho_B', Master_path))
                #stack_B_0_1, stack_B_0_2 =np.array(load(name_0_1, 'Rho_B', Master_path)),np.array(load(name_0_2, 'Rho_B', Master_path))                
                x_B = stack_1[:,0]
                x_B_0_1, rho_B_0_1 = stack_0_1[:, 0], stack_0_1[:, 1]
                x_B_0_2, rho_B_0_2 = stack_0_2[:, 0], stack_0_2[:, 1]
                
                rho_B = stack_1[:,1]/rho_B_0_1 + stack_2[:,1]/rho_B_0_2#*m**2/H_map**2/(a_i**4)
                    
                #stack_E_1 = np.array(load(name_1, 'Rho_E', Master_path));stack_E_0_1 = np.array(load(name_0_1, 'Rho_E', Master_path))
                #stack_E_2 = np.array(load(name_2, 'Rho_E', Master_path));stack_E_0_2 = np.array(load(name_0_2, 'Rho_E', Master_path))
                x_E_0_1, rho_E_0_1 = stack_0_1[:, 0], stack_0_1[:, 2]
                x_E_0_2, rho_E_0_2 = stack_0_2[:, 0], stack_0_2[:, 2]

                rho_E = stack_1[:, 2]/rho_E_0_1 + stack_2[:, 2]/rho_E_0_2#*m**2/H_map**2/(a_i**2)

                H_map = np.array([mapper(x, t, H_arr) for x in x_B])
                H_map = sqrt(item_f**2*(0.5*dphi**2+np.array([V(z,item_p) for z in phi]))); H_map=np.array([mapper(x, t, H_map) for x in x_B])
                t_i = trigger_matrix[idx_f][idx_p]
                a_i = mapper(t_i, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                a_map = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in x_B])/a_i;#a_map = 1/a_i
                cross_idx = np.argwhere(np.diff(np.sign(np.absolute(rho_E+rho_B) - H_map**2/m**2 ))).flatten()
                if len(cross_idx) == 0:
                    eq_idx = -1
                else:
                    eq_idx = cross_idx[0]
                dV_map = np.array([mapper(x, phi, t) for x in x_B])

                #stack_Bck_0 = [absolute(np.array(load(name_0_1, 'Backreaction', Master_path))[:,1]), absolute(np.array(load(name_0_2, 'Backreaction', Master_path))[:,1])]
                #stack_Bck = abs((absolute(np.array(load(name_1, 'Backreaction', Master_path))[:,1])/stack_Bck_0[0] - absolute(np.array(load(name_2, 'Backreaction', Master_path))[:,1])/stack_Bck_0[0]))/abs(dV_map*a_map**3)

                #plt.semilogy(x_B, stack_Bck, color = colors[color_cnt], linestyle=':')

                t_i = trigger_matrix[idx_f][idx_p]
                t_f = kill_matrix[idx_f][idx_p];a_f = mapper(t_f, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                #plt.subplot(2,2,1+cnt_f)
                plt.title(r'$f/{\rm Mpl}:%.4f$'%item_f)
                #plt.semilogy(x_B-t_i, absolute(rho_E/rho_E_0*m**2/H_map**2)/(a_i**2), color =colors[color_cnt], linestyle=['-','--'][idx_h])
                #plt.loglog(x_B-t_i, ro_B+rho_E, color=colors[color_cnt], linestyle=h_style[idx_h])
                rho_B_1 = (absolute(stack_1[:,1])/absolute(rho_B_0_1));rho_E_1 = (absolute(stack_1[:,2])/absolute(rho_E_0_1))#*m**2/H_map**2
                rho_B_2 = (absolute(stack_2[:,1])/absolute(rho_B_0_2));rho_E_2 = (absolute(stack_2[:,2])/absolute(rho_E_0_2))#*m**2/H_map**2
                ratio = (((rho_B_1+rho_B_2)/a_map**4+(rho_E_1+rho_E_2)/a_map**2)*m**2/(H_map**2))[-1]
                print(r'$\alpha:$', item_alpha, 'f:',item_f, 'Ratio:%.5f'%ratio)
                if ratio <2:
                    print(r'$B_{gen}:%.4e$' %(sqrt(rho_B_1+rho_B_2)*m**2*Mpl**2)[-1])
                    print(r'$\Lambda_{gen}:%.4e$' %(2/(m*Mpl)))
                a_R = T_0 / (T_R * Mpl)#; a_R = 1e-29  ## a_rT_r = a_0T_r
                #a_R = 1e-29 ##Subhramanian entropy conservation
                cross_idx_1 = np.argwhere(np.diff(np.sign(np.absolute(rho_E_1/a_map**2+rho_B_1/a_map**4) - H_map**2/m**2 ))).flatten()
                cross_idx_2 = np.argwhere(np.diff(np.sign(np.absolute(rho_E_2/a_map**2+rho_B_2/a_map**4) - H_map**2/m**2 ))).flatten()

                if len(cross_idx_1)==0:
                    cross_idx_1=-1
                    plt.semilogy(x_B[:cross_idx_1], (rho_B_1/a_map**4+rho_E_1/a_map**2)[:cross_idx_1], color=colors[color_cnt], linestyle='--')
                    plt.semilogy(x_B[cross_idx_1:], (rho_B_1/a_map**4+rho_E_1/a_map**2)[cross_idx_1:], color=colors[color_cnt], linestyle='--',alpha=0.5)
                    
                    B_k_f = sqrt((rho_B_1/a_map**4)[cross_idx_1])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl)
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)


                else:
                    cross_idx_1 =cross_idx_1[0]
                    plt.semilogy(x_B[:cross_idx_1+1], (rho_B_1/a_map**4+rho_E_1/a_map**2)[:cross_idx_1+1], color=colors[color_cnt], linestyle='--')
                    plt.semilogy(x_B[cross_idx_1:], (rho_B_1/a_map**4+rho_E_1/a_map**2)[cross_idx_1:], color=colors[color_cnt], linestyle='--',alpha=0.5)

                    B_k_f = sqrt((rho_B_1/a_map**4)[cross_idx_1])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl)
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)

                if len(cross_idx_2)==0:
                    cross_idx_2=-1
                    plt.semilogy(x_B[:cross_idx_2], (rho_B_2/a_map**4+rho_E_2/a_map**2)[:cross_idx_2], color=colors[color_cnt], linestyle='-')
                    plt.semilogy(x_B[cross_idx_2:], (rho_B_2/a_map**4+rho_E_2/a_map**2)[cross_idx_2:], color=colors[color_cnt], linestyle='-',alpha=0.5)
                    
                    B_k_f = sqrt((rho_B_2/a_map**4)[cross_idx_2])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl)
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0
                    
                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)

                else:
                    cross_idx_2 =cross_idx_2[0]
                    plt.semilogy(x_B[:cross_idx_2+1], (rho_B_2/a_map**4+rho_E_2/a_map**2)[:cross_idx_2+1], color=colors[color_cnt], linestyle='-')
                    plt.semilogy(x_B[cross_idx_2:], (rho_B_2/a_map**4+rho_E_2/a_map**2)[cross_idx_2:], color=colors[color_cnt], linestyle='-',alpha=0.5)


                    B_k_f = sqrt((rho_B_2/a_map**4)[cross_idx_2])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl)
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0
                    
                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)


                #plt.loglog(x_B, rho_B_1/a_map**4+rho_E_1/a_map**2, color=colors[color_cnt], linestyle='--')
                #plt.loglog(x_B, rho_B_2/a_map**4+rho_E_2/a_map**2, color=colors[color_cnt], linestyle='-')                
                #plt.semilogy(x_hel, abs(absolute(np.array(Hel))), color = colors[color_cnt], label=r'$\alpha:%.0f$'%(item_alpha))
                color_cnt = color_cnt + 1
        plt.semilogy(x_B, H_map**2/m**2,color='k')
        plt.ylim(1e-5,1e5)
    plt.legend()
    plt.show()
    return

def rho_plot2(p, alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix, Phi_Stack):
    h=[-1,1]

    #ax1 = plt.subplot(211)
    #ax2 = plt.subplot(212, sharex = ax1)

    H_inf = 1e-5

    f_idxs = [3,4,6]
    p_idxs = [0]
    alpha_idxs = [2]
    colors = cm.rainbow(np.linspace(0,1,len(f_idxs)))
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

            max_idx = np.argmax(abs(dphi))
            H_osc = H([dphi[max_idx], phi[max_idx]], item_p, item_f)

            #color_cnt = 0
            for cnt_alpha, idx_alpha in enumerate(alpha_idxs):
                item_alpha = alpha[idx_alpha]
                name_hel = 'alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                #stack_hel = np.array(load(name_hel, 'Helicity', Master_path));print(np.shape(stack_hel))
                #x_hel = stack_hel[:][0], Hel = stack_hel[:][1]
                #for idx_h, item_h in enumerate(h):
                name_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, -1, item_p, item_f)
                name_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, 1, item_p, item_f)                
                name_0_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., -1, item_p, item_f)
                name_0_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., 1, item_p, item_f)
                
                stack_B_1, stack_B_2 = np.array(load(name_1, 'Rho_B', Master_path)), np.array(load(name_2, 'Rho_B', Master_path))
                stack_B_0_1, stack_B_0_2 =np.array(load(name_0_1, 'Rho_B', Master_path)),np.array(load(name_0_2, 'Rho_B', Master_path))
                    
                x_B = stack_B_1[:,0]
                    
                x_B_0_1, rho_B_0_1 = stack_B_0_1[:, 0], stack_B_0_1[:, 1]
                x_B_0_2, rho_B_0_2 = stack_B_0_2[:, 0], stack_B_0_2[:, 1]
                rho_B = stack_B_1[:,1]/rho_B_0_1 + stack_B_2[:,1]/rho_B_0_2#*m**2/H_map**2/(a_i**4)
                    
                stack_E_1 = np.array(load(name_1, 'Rho_E', Master_path));stack_E_0_1 = np.array(load(name_0_1, 'Rho_E', Master_path))
                stack_E_2 = np.array(load(name_2, 'Rho_E', Master_path));stack_E_0_2 = np.array(load(name_0_2, 'Rho_E', Master_path))
                x_E_0_1, rho_E_0_1 = stack_E_0_1[:, 0], stack_E_0_1[:, 1]
                x_E_0_2, rho_E_0_2 = stack_E_0_2[:, 0], stack_E_0_2[:, 1]
                rho_E = stack_E_1[:, 1]/rho_E_0_1 + stack_E_2[:, 1]/rho_E_0_2#*m**2/H_map**2/(a_i**2)

                H_map = np.array([mapper(x, t, H_arr) for x in x_B])
                t_i = trigger_matrix[idx_f][idx_p]
                a_i = mapper(t_i, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                a_map = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in x_B])/a_i
                cross_idx = np.argwhere(np.diff(np.sign(np.absolute(rho_E/a_map**2+rho_B/a_map**4) - H_map**2/m**2 ))).flatten()
                if len(cross_idx) == 0:
                    eq_idx = -1
                else:
                    eq_idx = cross_idx[0]


                t_i = trigger_matrix[idx_f][idx_p]
                t_f = kill_matrix[idx_f][idx_p];a_f = mapper(t_f, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                #plt.subplot(2,1,1)
                #ax1.set_title(r'$f/{\rm Mpl}:%.4f$'%item_f)
                #plt.semilogy(x_B-t_i, absolute(rho_E/rho_E_0*m**2/H_map**2)/(a_i**2), color =colors[color_cnt], linestyle=['-','--'][idx_h])
                #plt.loglog(x_B-t_i, ro_B+rho_E, color=colors[color_cnt], linestyle=h_style[idx_h])

                rho_B_1 = stack_B_1[:,1]/rho_B_0_1;rho_E_1 = stack_E_1[:,1]/rho_E_0_1
                rho_B_2 = stack_B_2[:,1]/rho_B_0_2;rho_E_2 = stack_E_2[:,1]/rho_E_0_2

                cross_idx_1 = np.argwhere(np.diff(np.sign(np.absolute(rho_E_1/a_map**2+rho_B_1/a_map**4) - H_map**2/m**2 ))).flatten()
                cross_idx_2 = np.argwhere(np.diff(np.sign(np.absolute(rho_E_2/a_map**2+rho_B_2/a_map**4) - H_map**2/m**2 ))).flatten()
                if len(cross_idx_1)==0:
                    cross_idx_1=-1
                else:
                    cross_idx_1 =cross_idx_1[0]
                if len(cross_idx_2)==0:
                    cross_idx_2=-1
                else:
                    cross_idx_2 =cross_idx_2[0]

                plt.loglog(x_B[:cross_idx_1]-t_i, (rho_B_1/a_map**4+rho_E_1/a_map**2)[:cross_idx_1], color=colors[color_cnt], linestyle='--')
                plt.loglog(x_B[:cross_idx_2]-t_i, (rho_B_2/a_map**4+rho_E_2/a_map**2)[:cross_idx_2], color=colors[color_cnt], linestyle='-')
                plt.loglog(x_B[cross_idx_1:]-t_i, (rho_B_1/a_map**4+rho_E_1/a_map**2)[cross_idx_1:], color=colors[color_cnt], linestyle='--',alpha=0.5)
                plt.loglog(x_B[cross_idx_2:]-t_i, (rho_B_2/a_map**4+rho_E_2/a_map**2)[cross_idx_2:], color=colors[color_cnt], linestyle='-',alpha=0.5)

                plt.xlabel(r'$ mt  $')
                plt.ylabel(r'$\rho_{\rm em}/\rho_{\phi}$')


        #ax1.plot(x_B, [1]*len(x_B), color = 'k', linestyle = '-.')
        #plt.subplot(2,1,2)
        plt.loglog(x_B-t_i, H_map**2/m**2, color = colors[color_cnt], label = r'$H_{\rm osc}/m:%.3f$'%H_osc, linestyle = '-.');plt.ylabel(r'$H^2{\rm Mpl}^2/m^4$')
        color_cnt = color_cnt + 1
    plt.legend();plt.show()
    return


def rho_plot_f(p, alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix, Phi_Stack):
    h=[-1,1]

    H_inf = 1e-5
    Mpl = 2.435 * 1e18  ##Mpl in GeV
    GeV2 = 6.8 * 1e20  ##Gev^2 in Gauss

    T_R = 1e-5  ##in Mpl
    T_R = 1e-13 ##10^5GeV
    T_0 = 23.5e-14 ###Present photon temperature in GeV


    f_idxs = [3,4,7,-1]
    f_idxs = range(len(f))
    p_idxs = [0]
    alpha_idxs = [1, 2]
    colors = cm.rainbow(np.linspace(0,1,len(f)))
    color_cnt = 0
    for cnt_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        m = H_inf/item_f
        #m =1e-5
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
                #for idx_h, item_h in enumerate(h):
                name_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, -1, item_p, item_f)
                name_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, 1, item_p, item_f)                
                name_0_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., -1, item_p, item_f)
                name_0_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., 1, item_p, item_f)
                
                stack_B_1, stack_B_2 = np.array(load(name_1, 'Rho_B', Master_path)), np.array(load(name_2, 'Rho_B', Master_path))
                stack_B_0_1, stack_B_0_2 =np.array(load(name_0_1, 'Rho_B', Master_path)),np.array(load(name_0_2, 'Rho_B', Master_path))
                    
                x_B = stack_B_1[:,0]
                    
                x_B_0_1, rho_B_0_1 = stack_B_0_1[:, 0], stack_B_0_1[:, 1]
                x_B_0_2, rho_B_0_2 = stack_B_0_2[:, 0], stack_B_0_2[:, 1]
                rho_B = stack_B_1[:,1]/rho_B_0_1 + stack_B_2[:,1]/rho_B_0_2#*m**2/H_map**2/(a_i**4)
                    
                stack_E_1 = np.array(load(name_1, 'Rho_E', Master_path));stack_E_0_1 = np.array(load(name_0_1, 'Rho_E', Master_path))
                stack_E_2 = np.array(load(name_2, 'Rho_E', Master_path));stack_E_0_2 = np.array(load(name_0_2, 'Rho_E', Master_path))
                x_E_0_1, rho_E_0_1 = stack_E_0_1[:, 0], stack_E_0_1[:, 1]
                x_E_0_2, rho_E_0_2 = stack_E_0_2[:, 0], stack_E_0_2[:, 1]
                rho_E = stack_E_1[:, 1]/rho_E_0_1 + stack_E_2[:, 1]/rho_E_0_2#*m**2/H_map**2/(a_i**2)

                H_map = np.array([mapper(x, t, H_arr) for x in x_B])
                t_i = trigger_matrix[idx_f][idx_p]
                a_i = mapper(t_i, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                a_map = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in x_B])/a_i
                cross_idx = np.argwhere(np.diff(np.sign(np.absolute(rho_E+rho_B) - H_map**2/m**2 ))).flatten()
                if len(cross_idx) == 0:
                    eq_idx = -1
                else:
                    eq_idx = cross_idx[0]


                t_i = trigger_matrix[idx_f][idx_p]
                t_f = kill_matrix[idx_f][idx_p];a_f = mapper(t_f, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                plt.subplot(1,2,1+cnt_alpha)
                plt.title(r'$f/{\rm Mpl}:%.4f$'%item_f)
                #plt.semilogy(x_B-t_i, absolute(rho_E/rho_E_0*m**2/H_map**2)/(a_i**2), color =colors[color_cnt], linestyle=['-','--'][idx_h])
                #plt.loglog(x_B-t_i, rho_B+rho_E, color=colors[color_cnt], linestyle=h_style[idx_h])
                rho_B_1 = stack_B_1[:,1]/rho_B_0_1;rho_E_1 = stack_E_1[:,1]/rho_E_0_1
                rho_B_2 = stack_B_2[:,1]/rho_B_0_2;rho_E_2 = stack_E_2[:,1]/rho_E_0_2
                plt.loglog(x_B, (rho_B_1+rho_E_1)*m**2/item_f**2, color=colors[cnt_f], linestyle='--')
                plt.loglog(x_B, (rho_B_2+rho_E_2)*m**2/item_f**2, color=colors[cnt_f], linestyle='-')
                plt.loglog(x_B, H_map**2/item_f**2, color = colors[cnt_f], linestyle='-.')
                color_cnt = color_cnt + 1
    [plt.plot([],[],color=colors[i], label=r'$f/{\rm Mpl}:%.4f$'%f[i]) for i in range(len(f))]; plt.legend()
    plt.show()
    return


def peak_k_plot(args, asymp_stack):
    [p, f, alpha, k_arr, kill_arr, Phi_stack, Master_path] = args
    h = [-1,1]
    h_style = ['--', ':']
    targ_alpha_idxs = [1,2,3,4];targ_alpha=alpha[targ_alpha_idxs]
    colors_alpha = cm.rainbow(np.linspace(0,1,len(targ_alpha_idxs)))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot([],[],label = r'$\alpha$', linestyle = '')
    [ax1.plot([],[],label = r'%.2f'%(targ_alpha[i]), color = colors_alpha[i]) for i in range(len(targ_alpha_idxs))]

    for idx_alpha, item_idx_alpha in enumerate(targ_alpha_idxs):
    
        for idx_p, item_p in enumerate(p):
            stack_f = []
            for idx_f, item_f in enumerate(f):        

                t = Phi_stack[idx_f][idx_p][0]
                max_dphi = np.argmax(abs(Phi_stack[idx_f][idx_p][2]))
                phi_max = Phi_stack[idx_f][idx_p][1][max_dphi]; dphi_max = Phi_stack[idx_f][idx_p][2][max_dphi];t_max = t[max_dphi]
                H_osc = H([phi_max, dphi_max], item_p, item_f)

                item_alpha = alpha[item_idx_alpha]
                #t_f = kill_arr[idx_p][idx_H]

                A = [absolute(asymp_stack[idx_f, idx_h, idx_p, item_idx_alpha, :, 1]) for idx_h in [0,1]]

                #k = np.array([item[0] for item in sub_stack[0]])/exp(t_f)
                #A = [[item[1] for item in sub_stack[i]] for i in [0,1]]
                #dA = [[item[2] for item in sub_stack[i]] for i in [0,1]]

                k=k_arr[idx_f][idx_p]

                Integrand_diff = k ** 2 * abs(np.array(A[0])**2-np.array(A[1])**2)
                Integrand_sum = (abs(np.array(A[0])**2+np.array(A[1])**2)) * k ** 2
                integrated = trapz(Integrand_diff, x=k)/trapz(Integrand_sum, x=k)
                
                stack_f.append([item_f, integrated, H_osc])
            #print(np.shape(stack_p))
            stack_f = np.array(stack_f)
            ax1.plot(stack_f[:,2], stack_f[:,1], color = colors_alpha[idx_alpha], marker = '+')
    ax1.set_xlabel(r'H_{\rm osc}/m')
    #ax1.set_ylabel()
    ax1.legend()
    plt.show()
    return



def Hel_H_osc_plot(args, asymp_stack):
    [p, f, alpha, k_arr, kill_arr, Phi_stack, Master_path, k_generator] = args
    h = [-1,1]
    h_style = ['--', ':']
    #targ_alpha_idxs = [4,5,6];targ_alpha=alpha[targ_alpha_idxs]
    targ_alpha = [0.6, 0.8]
    targ_alpha_idxs = [np.argmin(abs(targ-alpha)) for targ in targ_alpha]

    colors_alpha = cm.rainbow(np.linspace(0,1,len(targ_alpha_idxs)))

    p_osc = np.linspace(2,8,24)
    f_osc = np.geomspace(5e-1,5e-4, 24*2)
    len_k_osc = 24*20
    alpha_osc = np.linspace(0.,4, 2)    
    steps_osc = 1e5
    splice_osc = steps_osc/100

    Data_set_osc = 'p_%.2f_%.2f_%.1f_f_%.4f_%.4f_%.1f_k__%.1f_steps_%.1f'%(p_osc[0], p_osc[-1], len(p_osc),f_osc[0], f_osc[-1], len(alpha_osc), len_k_osc, steps_osc)
    Master_path_osc = '/work2/Teerthal/Reheating/%s'%(Data_set_osc)
    Phi_stack_osc = load_or_exec('Phi_stack', 'Phi', Phi_Spooler, Master_path_osc, [p_osc,f_osc] )
    a_stack_osc = load_or_exec('a_stack', 'a', a_spooler, Master_path_osc, [p_osc, Phi_stack_osc, f_osc])
    trigger_arr_osc = initiation_matrix([p_osc, Phi_stack_osc, 'Phi', 4., f_osc])
    kill_arr_osc = kill_matrix([p_osc, Phi_stack_osc, splice_osc, 'Phi', .5, f_osc, 50])
    k_osc = k_generator(p_osc,a_stack_osc,len_k_osc, kill_arr_osc, trigger_arr_osc, f_osc)


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    [ax1.plot([],[],label = r'%.2f'%(targ_alpha[i]), color = colors_alpha[i]) for i in range(len(targ_alpha_idxs))]
    ax2 = ax1.twiny()
    for idx_alpha, item_idx_alpha in enumerate(targ_alpha_idxs):

        for idx_p, item_p in enumerate(p):
            stack_f = []
            for idx_f, item_f in enumerate(f):        
    
                t = Phi_stack[idx_f][idx_p][0]
                max_dphi = np.argmax(abs(Phi_stack[idx_f][idx_p][2]))
                phi_max = Phi_stack[idx_f][idx_p][1][max_dphi]; dphi_max = Phi_stack[idx_f][idx_p][2][max_dphi];t_max = t[max_dphi]
                H_osc = H([phi_max, dphi_max], item_p, item_f)
                H_osc = 1/3*item_f*sqrt(0.5*absolute(dphi_max)**2+V(phi_max,item_p))/(sqrt(V(6,item_p)))

                item_alpha = alpha[item_idx_alpha]
                #t_f = kill_arr[idx_p][idx_H]
                k_ids = [0,-1]

                A_0 = [absolute(asymp_stack[idx_f, idx_h, idx_p, 0, :, 1]) for idx_h in [0,1]]
                A = [absolute(asymp_stack[idx_f, idx_h, idx_p, item_idx_alpha, :, 1])/A_0[idx_h] for idx_h in [0,1]]

                k=k_arr[idx_f][idx_p]#[k_ids[0]:k_ids[1]]
                
                rho_B_fn = 1 / (4 * pi ** 2) * (np.array(A[0])**2 - np.array(A[1])**2) * k ** 4
                rho_B_fn2 = 1 / (4 * pi ** 2) * (np.array(A[0])**2 + np.array(A[1])**2) * k ** 4
                intg_rho_B = trapz(rho_B_fn, x=log(k))/trapz(rho_B_fn2, x=log(k))

                pk_idx = np.argmax(A[0]);pk_idx_1 = np.argmax(A[0]);pk_idx_2 = np.argmax(A[1])
                pk_diff = abs(A[0][pk_idx_1]**2 - A[1][pk_idx_2]**2)/abs(A[0][pk_idx_1]**2 + A[1][pk_idx_2]**2)
                wid = 1
                #A = np.array([dum[pk_idx-wid:pk_idx+wid] for dum in A])
                #k = np.array([item[0] for item in sub_stack[0]])/exp(t_f)
                #A = [[item[1] for item in sub_stack[i]] for i in [0,1]]
                #dA = [[item[2] for item in sub_stack[i]] for i in [0,1]]

                #k = np.array(k[pk_idx-wid:pk_idx+wid])
                Integrand_diff = k ** 2 * abs(np.array(A[0])**2-np.array(A[1])**2)
                Integrand_sum = (abs(np.array(A[0])**2+np.array(A[1])**2)) * k ** 2
                integrated = trapz(Integrand_diff, x=k)/trapz(Integrand_sum, x=k)
                
                stack_f.append([item_f, integrated, H_osc, intg_rho_B, pk_diff])
            #print(np.shape(stack_p))
            stack_f = np.array(stack_f)
            ax1.loglog(sqrt(3)*sqrt(2)*stack_f[:,0], stack_f[:,1], color = colors_alpha[idx_alpha], marker = '+')
            ax1.loglog(sqrt(3)*sqrt(2)*stack_f[:,0], abs(stack_f[:,3]), color = colors_alpha[idx_alpha], linestyle = ':', marker = 'o')
            ax1.loglog(sqrt(6)*stack_f[:,0], stack_f[:,4], color = colors_alpha[idx_alpha], linestyle = '--', marker = 'x')
            ax2.loglog(stack_f[:,2], stack_f[:,1], linestyle = '')
    #ax2 = ax1.twinx()
    p_idxs = [0]
    colors_osc = cm.rainbow(np.linspace(0, 1, len(p_idxs)))
    color_cnt = 0
    [plt.scatter([],[],color = x, label = 'p:%.1f'%p_i) for (x, p_i) in zip(colors, p_osc[p_idxs])]
    for id_f, it_f in enumerate(f_osc):

        for cn_p, id_p in enumerate(p_idxs):#np.arange(0,len(p),1):
            t_trigger = trigger_arr_osc[id_f][id_p]
            t_kill = kill_arr_osc[id_f][id_p]
            p_i = p_osc[id_p]
            sosc = Phi_stack_osc[id_f][id_p]
            t_osc = sosc[0]
            phi_osc = sosc[1]
            dphi_osc = sosc[2]

            t_a = a_stack_osc[id_f][id_p, 0]
            a = a_stack_osc[id_f][id_p, 1]

            max_idx = np.argmax(abs(dphi_osc))
            #H_os = H([dphi_osc[max_idx], phi_osc[max_idx]], p_i, it_f)
            H_os = 1/3*it_f*sqrt(0.5*absolute(dphi_osc[max_idx])**2+V(phi_osc[max_idx],p_i))/(sqrt(V(6,p_i)))
            #cross_idx = np.argwhere(np.diff(np.sign(np.real(dphi) - [0] * len(dphi)))).flatten()
            #osc_idx = cross_idx[1]
            #H_osc = H([dphi[osc_idx], phi[osc_idx]], p_i)
            #ax0.set_xscale('Log')
            #parV = functools.partial(V, p=p_i)
            #V_arr = np.array([*map(parV, phi_osc)])
            #par_H = functools.partial(H, p=p_i, f= it_f)
            #args = np.array(list(zip(phi_osc,dphi_osc)))
            #H_arr = 1/3*it_f*sqrt(0.5*absolute(dphi_osc)**2+V(phi_osc,p_i))/(sqrt(V(6,p_i)))
            #ax2.scatter( sqrt(3)*it_f*sqrt(2), H_os, color = colors_osc[cn_p] )
            #ax2.plot(H_os, stack_f[:,1], linestyle='')
    ax2.set_xscale('Log'); ax2.set_yscale('Log')
    ax1.set_ylabel(r'$h_B$')#r'$\langle{\mathfrak{H}}\rangle$')
    ax2.set_xlabel(r'$H_{\rm osc}/m$')
    ax1.set_xlabel(r'$f/{\rm Mpl}$')
    #ax1.set_ylabel()
    ax1.plot([],[],linestyle = '--', marker = 'x', color = 'k', label=r'$\left| \frac{{\rm max}[{\cal A}_+]^{2} - {\rm max}[{\cal A}_-]^{2}} { {\rm max}[{\cal A}_+]^{2} + {\rm max}[{\cal A}_-]^{2}} \right|$') 
    ax1.plot([],[],linestyle = ':', marker = 'o', color = 'k', label = r'$\left| \frac{ \rho_{B_-}^2 - \rho_{B_+}^2}{ \rho_{B_-}^2 + \rho_{B_+}^2}  \right|$')
    ax1.legend()
    plt.show()
    return



def B_present(args, asymp_stack):

    [p, f, alpha, k_arr, kill_arr, Phi_stack, Master_path, k_generator, a_stack] = args

    H_inf = 3.6e-5
    r = 0.1
    rho_phi = 1 / (3 * H_inf ** 2)
    dVdphi = 3 * sqrt(r) / H_inf
    
    Mpl = 2.435 * 1e18  ##Mpl in GeV
    GeV2 = 6.8 * 1e20  ##Gev^2 in Gauss

    T_R = 1e-5  ##in Mpl
    T_R = (90/(100*pi**2))**(1/2)*(H_inf)**(1/2) 
    T_0 = 23.5e-14 ###Present photon temperature in GeV

    #b_styles = ['-','--']

    fig, ax1 = plt.subplots(1,1)#;ax1 = fig.add_subplot(111)

    targ_alpha_idxs = [1,2,3,4,5,6];targ_alpha_idxs = [1,2,3,4,5]
    colors = cm.rainbow(np.linspace(0,1,len(targ_alpha_idxs)))

    for idx_alpha, item_idx_alpha in enumerate(targ_alpha_idxs):
        item_alpha = alpha[item_idx_alpha]
        for idx_p, item_p in enumerate(p):
            stack_f = []
            for idx_f, item_f in enumerate(f):        
                
                m = H_inf/item_f#;m=1e-5

                t = Phi_stack[idx_f][idx_p][0]
                max_dphi = np.argmax(abs(Phi_stack[idx_f][idx_p][2]))
                phi_max = Phi_stack[idx_f][idx_p][1][max_dphi]; dphi_max = Phi_stack[idx_f][idx_p][2][max_dphi];t_max = t[max_dphi]
                H_osc = H([phi_max, dphi_max], item_p, item_f)

                t_kill = kill_arr[idx_f][idx_p]
                a = a_stack[idx_f][idx_p, 1]
                t_a = a_stack[idx_f][idx_p, 0]
                a_f = mapper(t_kill, t_a , a)

                item_alpha = alpha[item_idx_alpha]
                #t_f = kill_arr[idx_p][idx_H]
                k_ids = [0,-1]

                A_0 = [absolute(asymp_stack[idx_f, idx_h, idx_p, 0, :, 1]) for idx_h in [0,1]]
                A = [absolute(asymp_stack[idx_f, idx_h, idx_p, item_idx_alpha, :, 1])-absolute(A_0[idx_h]) for idx_h in [0,1]]

                max_idx = np.argmax(A[0])
                max_k, max_A = [k_arr[idx_f][idx_p][max_idx], sqrt(absolute(A[0][max_idx])**2+absolute(A[1][max_idx])**2)]

                print(r'$\alpha$', item_alpha, 'f', item_f)
                a_R = T_0 / (T_R * Mpl)*(3/100)**(1/3)  ## a_rT_r = a_0T_r
                Lambda_f = 2/(m*Mpl); print(r'$\Lambda_{gen}:%.4e$' %(Lambda_f))  #1 / (max_k * H_inf * Mpl)  ## dimensionless, normalized by GeV^-1
                #Lambda_f = (a_R/(max_k*m*Mpl))
                B_k_f = (m ** 2 * (max_k/a_f) ** 2 * max_A) * Mpl ** 2; print(r'$B_{gen}:%.4e$' %(B_k_f))  ##Dimensionless, normalized by GeV^2

                #a_R = 1e-29 ##Subhramanian entropy conservation
                Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3); print(r'$\Lambda_{0}:%.4e$' %(Lambda_0))
                B_0 = 1e-8 * Lambda_0

                stack_f.append([item_f, Lambda_0, B_0])

            slit = np.array(stack_f)
            ax1.semilogy(slit[:,0], slit[:,2], color=colors[idx_alpha])
            ax1.set_ylabel(r'$B_{m,0}(\rm G)$')
    ax2 = ax1.twinx()
    ax2.semilogy(slit[:,0], slit[:,2]*1e8, linestyle = '')
    ax2.set_ylabel(r'$\lambda_{m,0}(\rm Mpc)$')
    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    plt.show()

    return


def rho_peak_plot(p, alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix, Phi_Stack, asym_stack):
    h=[-1,1]

    H_inf = 1e-5
    targ_f = [0.02,0.011]
    
    f_idxs = [np.argmin(abs(targ-f)) for targ in targ_f]
    #f_idxs = [6,12]
    p_idxs = [0]
    alpha_idxs = [1,2,3,4,5,6]
    colors = cm.rainbow(np.linspace(0,1,len(alpha_idxs)))
    color_cnt = 0
    T_0 = 23.5*1e-14
    Mpl = 2.435 * 1e18  ##Mpl in GeV
    GeV2 = 6.8 * 1e20  ##Gev^2 in Gauss
    T_R = 1e-13;T_R = (90/(100*pi**2))**(1/2)*(H_inf)**(1/2) 
    for cnt_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        m = H_inf/item_f
        #m =1e-5
        plt.subplot(1,2,1+cnt_f)
        plt.xlabel(r'$mt$')
        plt.ylabel(r'$\rho_{\rm em}/m^4$')
        for cnt_p, idx_p, in enumerate(p_idxs):
            item_p = p[idx_p]
            colors_alpha = cm.rainbow(np.linspace(0,1, len(alpha)))
            sol = Phi_Stack[idx_f][idx_p]
            t = sol[0];phi_a = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in t])/mapper(t[0], a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
            phi = sol[1]
            dphi = sol[2]
            par_H = functools.partial(H, p=item_p, f= item_f)
            args = np.array(list(zip(phi,dphi)))
            H_arr = np.array([*map(par_H, args)])
            par_dV = functools.partial(dV, p=item_p)
            m = H_inf/item_f
            dV_map = np.array([*map(par_dV, phi)])*item_f/m**2

            color_cnt = 0
            for cnt_alpha, idx_alpha in enumerate(alpha_idxs):

                #plt.subplot(1,2,1+cnt_alpha)
                #plt.xlabel(r'$mt$')
                #plt.ylabel(r'$\rho_{\rm em}/m^4$')

                item_alpha = alpha[idx_alpha]


                A_0 = [absolute(asymp_stack[idx_f, idx_h, idx_p, 0, :, 1]) for idx_h in [0,1]]
                A = [absolute(asymp_stack[idx_f, idx_h, idx_p, item_idx_alpha, :, 1])-absolute(A_0[idx_h]) for idx_h in [0,1]]

                max_idx = np.argmax(A[0])
                max_k, max_A = [k_arr[idx_f][idx_p][max_idx], sqrt(absolute(A[0][max_idx])**2+absolute(A[1][max_idx])**2)]

                #name_hel = 'alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                #name_hel = 'resolved alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                #stack_hel = np.array(load(name_hel, 'Helicity', Master_path));print(np.shape(stack_hel))
                #x_hel = stack_hel[:][0], Hel = stack_hel[:][1]
                #for idx_h, item_h in enumerate(h):
                name_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, -1, item_p, item_f)
                name_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, 1, item_p, item_f)                
                name_0_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., -1, item_p, item_f)
                name_0_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., 1, item_p, item_f)

                #x_hel = [stack_hel[i][0] for i in np.arange(0,len(stack_hel),1)]; Hel = [stack_hel[i][1] for i in np.arange(0,len(stack_hel),1)]
                #x_hel = stack_hel[0,:];Hel=stack_hel[1,:]
                
                stack_B_1, stack_B_2 = np.array(load(name_1, 'Rho_B', Master_path)), np.array(load(name_2, 'Rho_B', Master_path))
                stack_B_0_1, stack_B_0_2 =np.array(load(name_0_1, 'Rho_B', Master_path)),np.array(load(name_0_2, 'Rho_B', Master_path))                

                x_B = stack_B_1[:,0]
                    
                x_B_0_1, rho_B_0_1 = stack_B_0_1[:, 0], stack_B_0_1[:, 1]
                x_B_0_2, rho_B_0_2 = stack_B_0_2[:, 0], stack_B_0_2[:, 1]
                rho_B = stack_B_1[:,1]/rho_B_0_1 + stack_B_2[:,1]/rho_B_0_2#*m**2/H_map**2/(a_i**4)
                    
                stack_E_1 = np.array(load(name_1, 'Rho_E', Master_path));stack_E_0_1 = np.array(load(name_0_1, 'Rho_E', Master_path))
                stack_E_2 = np.array(load(name_2, 'Rho_E', Master_path));stack_E_0_2 = np.array(load(name_0_2, 'Rho_E', Master_path))
                x_E_0_1, rho_E_0_1 = stack_E_0_1[:, 0], stack_E_0_1[:, 1]
                x_E_0_2, rho_E_0_2 = stack_E_0_2[:, 0], stack_E_0_2[:, 1]
                rho_E = stack_E_1[:, 1]/rho_E_0_1 + stack_E_2[:, 1]/rho_E_0_2#*m**2/H_map**2/(a_i**2)

                H_map = np.array([mapper(x, t, H_arr) for x in x_B])
                H_map = sqrt(item_f**2*(0.5*dphi**2+np.array([V(z,item_p) for z in phi]))); H_map=np.array([mapper(x, t, H_map) for x in x_B])
                t_i = trigger_matrix[idx_f][idx_p]
                a_i = mapper(t_i, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                a_map = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in x_B])/a_i;#a_map = 1/a_i
                cross_idx = np.argwhere(np.diff(np.sign(np.absolute(rho_E+rho_B) - H_map**2/m**2 ))).flatten()
                if len(cross_idx) == 0:
                    eq_idx = -1
                else:
                    eq_idx = cross_idx[0]
                dV_map = np.array([mapper(x, phi, t) for x in x_B])

                stack_Bck_0 = [absolute(np.array(load(name_0_1, 'Backreaction', Master_path))[:,1]), absolute(np.array(load(name_0_2, 'Backreaction', Master_path))[:,1])]
                stack_Bck = abs((absolute(np.array(load(name_1, 'Backreaction', Master_path))[:,1])/stack_Bck_0[0] - absolute(np.array(load(name_2, 'Backreaction', Master_path))[:,1])/stack_Bck_0[0]))/abs(dV_map*a_map**3)

                plt.semilogy(x_B, stack_Bck, color = colors[color_cnt], linestyle=':')

                t_i = trigger_matrix[idx_f][idx_p]
                t_f = kill_matrix[idx_f][idx_p];a_f = mapper(t_f, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                #plt.subplot(2,2,1+cnt_f)
                plt.title(r'$f/{\rm Mpl}:%.4f$'%item_f)
                #plt.semilogy(x_B-t_i, absolute(rho_E/rho_E_0*m**2/H_map**2)/(a_i**2), color =colors[color_cnt], linestyle=['-','--'][idx_h])
                #plt.loglog(x_B-t_i, ro_B+rho_E, color=colors[color_cnt], linestyle=h_style[idx_h])
                rho_B_1 = (absolute(stack_B_1[:,1])/absolute(rho_B_0_1));rho_E_1 = (absolute(stack_E_1[:,1])/absolute(rho_E_0_1))#*m**2/H_map**2
                rho_B_2 = (absolute(stack_B_2[:,1])/absolute(rho_B_0_2));rho_E_2 = (absolute(stack_E_2[:,1])/absolute(rho_E_0_2))#*m**2/H_map**2
                ratio = (((rho_B_1+rho_B_2)/a_map**4+(rho_E_1+rho_E_2)/a_map**2)*m**2/(H_map**2))[-1]
                print(r'$\alpha:$', item_alpha, 'f:',item_f, 'Ratio:%.5f'%ratio)



                print(r'$\alpha$', item_alpha, 'f', item_f)
                a_R = T_0 / (T_R * Mpl)*(3/100)**(1/3)  ## a_rT_r = a_0T_r
                Lambda_f = 2/(m*Mpl);   #1 / (max_k * H_inf * Mpl)  ## dimensionless, normalized by GeV^-1
                Lambda_f = (a_R/(max_k*m*Mpl))
                B_k_f = (m ** 2 * (max_k/a_map[-1]) ** 2 * max_A) * Mpl ** 2;   ##Dimensionless, normalized by GeV^2

                #a_R = 1e-29 ##Subhramanian entropy conservation
                Lambda_k_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3); print(r'$\Lambda_{0}:%.4e$' %(Lambda_0))
                B_k_0 = 1e-8 * Lambda_k_0

                if ratio <2:
                    print(r'$B_{gen}:%.4e$' %(sqrt(rho_B_1+rho_B_2)*m**2*Mpl**2/a_map[-1]**4)[-1])
                    print(r'$\Lambda_{gen}:%.4e$' %(2/(m*Mpl)))
                    print('Peak Only(B_k)')
                    print(r'$\Lambda_{m}:%.4e$' %(Lambda_f))
                    print(r'$B_{gen}:%.4e$' %(B_k_f))
                    print(r'$\lambda_0:%.4e B_0:%.4f$'%(Lambda_k_0, B_k_0))


                a_R = T_0 / (T_R * Mpl)#; a_R = 1e-29  ## a_rT_r = a_0T_r
                #a_R = 1e-29 ##Subhramanian entropy conservation
                cross_idx_1 = np.argwhere(np.diff(np.sign(np.absolute(rho_E_1/a_map**2+rho_B_1/a_map**4) - H_map**2/m**2 ))).flatten()
                cross_idx_2 = np.argwhere(np.diff(np.sign(np.absolute(rho_E_2/a_map**2+rho_B_2/a_map**4) - H_map**2/m**2 ))).flatten()


                if len(cross_idx_1)==0:
                    cross_idx_1=-1
                    plt.semilogy(x_B[:cross_idx_1], (rho_B_1/a_map**4+rho_E_1/a_map**2)[:cross_idx_1], color=colors[color_cnt], linestyle='--')
                    plt.semilogy(x_B[cross_idx_1:], (rho_B_1/a_map**4+rho_E_1/a_map**2)[cross_idx_1:], color=colors[color_cnt], linestyle='--',alpha=0.5)
                    
                    B_k_f = sqrt((rho_B_1/a_map**4)[cross_idx_1])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl)
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)


                else:
                    cross_idx_1 =cross_idx_1[0]
                    plt.semilogy(x_B[:cross_idx_1+1], (rho_B_1/a_map**4+rho_E_1/a_map**2)[:cross_idx_1+1], color=colors[color_cnt], linestyle='--')
                    plt.semilogy(x_B[cross_idx_1:], (rho_B_1/a_map**4+rho_E_1/a_map**2)[cross_idx_1:], color=colors[color_cnt], linestyle='--',alpha=0.5)

                    B_k_f = sqrt((rho_B_1/a_map**4)[cross_idx_1])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl)
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0

                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)

                if len(cross_idx_2)==0:
                    cross_idx_2=-1
                    plt.semilogy(x_B[:cross_idx_2], (rho_B_2/a_map**4+rho_E_2/a_map**2)[:cross_idx_2], color=colors[color_cnt], linestyle='-')
                    plt.semilogy(x_B[cross_idx_2:], (rho_B_2/a_map**4+rho_E_2/a_map**2)[cross_idx_2:], color=colors[color_cnt], linestyle='-',alpha=0.5)
                    
                    B_k_f = sqrt((rho_B_2/a_map**4)[cross_idx_2])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl)
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0
                    
                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)

                else:
                    cross_idx_2 =cross_idx_2[0]
                    plt.semilogy(x_B[:cross_idx_2+1], (rho_B_2/a_map**4+rho_E_2/a_map**2)[:cross_idx_2+1], color=colors[color_cnt], linestyle='-')
                    plt.semilogy(x_B[cross_idx_2:], (rho_B_2/a_map**4+rho_E_2/a_map**2)[cross_idx_2:], color=colors[color_cnt], linestyle='-',alpha=0.5)


                    B_k_f = sqrt((rho_B_2/a_map**4)[cross_idx_2])*(m*Mpl)**2
                    Lambda_f = 2/(m*Mpl)
                    Lambda_0 = 3.3e5 * a_R * (Lambda_f) ** (1 / 3) * (B_k_f) ** (2 / 3)
                    B_0 = 1e-8 * Lambda_0
                    
                    print('alpha:',item_alpha, 'f',item_f, Lambda_0, B_0)


                #plt.loglog(x_B, rho_B_1/a_map**4+rho_E_1/a_map**2, color=colors[color_cnt], linestyle='--')
                #plt.loglog(x_B, rho_B_2/a_map**4+rho_E_2/a_map**2, color=colors[color_cnt], linestyle='-')                
                #plt.semilogy(x_hel, abs(absolute(np.array(Hel))), color = colors[color_cnt], label=r'$\alpha:%.0f$'%(item_alpha))
                color_cnt = color_cnt + 1
        plt.semilogy(x_B, H_map**2/m**2,color='k')
    #plt.legend()
    plt.show()
    return


def rho_contours(p, alpha, k, x_len, a_stack, Master_path, f, trigger_matrix, kill_matrix, Phi_Stack, asymp_stack):
    h = [-1, 1]

    H_inf = 1e-8
    targ_f = [0.01]

    f_idxs = [np.argmin(abs(targ - f)) for targ in targ_f]
    # f_idxs = [6,12]
    p_idxs = [0]

    f_idxs = range(len(f)); alpha_idxs = range(len(alpha))

    #f_idxs = [0,-1]
    #f = f[f_idxs]
    #alpha_idxs = [1,-1]
    #alpha = alpha[alpha_idxs]

    T_0 = 23.5 * 1e-14
    Mpl = 2.435 * 1e18  ##Mpl in GeV
    GeV2 = 6.8 * 1e20  ##Gev^2 in Gauss
    T_R = 1e-13;
    T_R = (90 / (100 * pi ** 2)) ** (1 / 2) * (H_inf) ** (1 / 2)

    z = np.zeros((len(f), len(alpha)))
    Hel_arr = np.zeros((len(f), len(alpha)))
    fig, ax = plt.subplots(2, 2)
    ax1, ax2, ax3, ax4 = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
    colormap = plt.cm.viridis  # or any other colormap

    min = 0

    for cnt_f, idx_f in enumerate(f_idxs):
        item_f = f[idx_f]
        # m = H_inf/item_f
        # m =1e-5
        #plt.subplot(2, len(targ_f), 1 + cnt_f)
        #plt.xlabel(r'$mt$')
        #plt.ylabel(r'$\rho_{\rm em}/m^4$')
        for cnt_p, idx_p, in enumerate(p_idxs):
            item_p = p[idx_p]
            colors_alpha = cm.rainbow(np.linspace(0, 1, len(alpha)))
            sol = Phi_Stack[idx_f][idx_p]
            t = sol[0];

            phi = sol[1]
            dphi = sol[2]
            par_H = functools.partial(H, p=item_p, f=item_f)
            args = np.array(list(zip(dphi, phi)))
            H_arr = np.array([*map(par_H, args)])

            color_cnt = 0
            for cnt_alpha, idx_alpha in enumerate(alpha_idxs[:-1]):

                item_alpha = alpha[idx_alpha]
                #lt.plot([], [], color=colors[cnt_alpha], label=r'$\alpha:%.2f$' % (item_alpha))

                
                name_hel = 'resolved alpha:%.5f_p:%.3f_f_%.5f' % (item_alpha, item_p, item_f)
                stack_hel = np.array(load(name_hel, 'Helicity', Master_path));print(np.shape(stack_hel))
                # x_hel = stack_hel[:][0], Hel = stack_hel[:][1]
                # for idx_h, item_h in enumerate(h):
                name_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, -1, item_p, item_f)
                name_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (item_alpha, 1, item_p, item_f)
                name_0_1 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., -1, item_p, item_f)
                name_0_2 = 'alpha:%.5f_h:%s_p:%.3f_f_%.5f' % (0., 1, item_p, item_f)

                #x_hel = [stack_hel[i][0] for i in np.arange(0,len(stack_hel),1)]; Hel = [stack_hel[i][1] for i in np.arange(0,len(stack_hel),1)]
                x_hel = stack_hel[0,:];Hel=stack_hel[1,:]

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

                H_map = np.array([mapper(x, t, H_arr) for x in x_B])

                m = H_inf / item_f * (3 * (0.5 * absolute(dphi[0]) ** 2 + V(phi[0], item_p)) ** (-1)) ** (1 / 2)

                H_map = sqrt(item_f ** 2 * (0.5 * dphi ** 2 + np.array([V(z, item_p) for z in phi]))) / sqrt(
                    (0.5 * dphi[0] ** 2 + V(phi[0], item_p)));
                H_map = np.array([mapper(x, t, H_map) for x in x_B])
                t_i = trigger_matrix[idx_f][idx_p]
                a_i = mapper(t_i, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1])
                a_map = np.array([mapper(x, a_stack[idx_f][idx_p][0], a_stack[idx_f][idx_p][1]) for x in x_B]) / a_i;
                a_map = 1. / a_i  # print(np.shape(rho_B),np.shape(rho_E),np.shape(H_map))#a_map = 1/a_i
                cross_idx = np.argwhere(
                    np.diff(np.sign(np.absolute(rho_E + rho_B) - 3 * H_map ** 2 / m ** 2))).flatten()
                if len(cross_idx) == 0:
                    eq_idx = -1
                else:
                    eq_idx = cross_idx[0]
                dV_map = np.array([mapper(x, phi, t) for x in x_B])


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

                rho_phi = 3*H_map**2*m**2*Mpl**4


                ratio = \
                abs(((rho_B_1 + rho_B_2) / a_map ** 4 + (rho_E_1 + rho_E_2) / a_map ** 2)/rho_phi )[-1]
                print(item_alpha, item_f, ratio)
                color = ratio
                if color > 1.1:
                    color = 1.
                if color < 1e-6:
                    color = 1e-6
                z[idx_f, idx_alpha] = log(color)


                if color != 0 and log(color) < min:
                    min = log(color)
                    
                avg_hel = np.mean(Hel)
                cross_idx = np.argwhere(np.diff(np.sign(Hel - [0] * len(Hel)))).flatten()

                Hel_arr[idx_f, idx_alpha] = len(cross_idx)
                
    normalize = LogNorm(vmin=min, vmax=1)

    x = alpha
    y = f

    cs = ax2.contourf(x,y,z,25, cmap = colormap, vmin = min, vmax = 1.)#, locator=ticker.LogLocator())
    cbar = fig.colorbar(cs, ax = ax2)
    ax2.contour(x,y,Hel_arr, levels=[1,10], color = ['w', 'red'])
    cs2 = ax4.contourf(x, y, Hel_arr, 20, cmap=colormap)
    cbar2 = fig.colorbar(cs2, ax = ax4)
    
    x,y = np.meshgrid(x,y)
    ax1. scatter(x,y, c=z, cmap = colormap, vmin = min, vmax = 1)
    ax3. scatter(x,y, c=Hel_arr, cmap = colormap)

    ax1.set_yscale('Log'); ax2.set_yscale('Log')
    ax3.set_yscale('Log');
    ax4.set_yscale('Log')
    plt.show()

    return
