from main import *



def potential_plot(p):

    phi = np.linspace(-10,10, 1000)
    par = functools.partial(V, p=p)
    V_arr = np.array([*map(par,phi)])
    label = 'p:%s ' % (p)
    return plt.plot(phi, V_arr, label = label)

def scalar_plot(p, H_m, Phi_Stack, a_stack, kill_arr, trigger_arr):

    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)

    color = cm.rainbow(np.linspace(0,1, len(p)+len(H_m)))
    color_cnt = 0
    for idx_p in np.arange(0,len(p),1):
        for idx_H, item_H in enumerate(H_m):
            t_trigger = trigger_arr[idx_p][idx_H]
            t_kill = kill_arr[idx_p][idx_H]
            p_i = p[idx_p]
            sol = Phi_Stack[idx_p][idx_H]
            t = sol[0]
            phi = sol[1]
            dphi = sol[2]

            ax0.plot(t, phi, label='p:%.2f H_m:%.3f' % (p_i, item_H), color = color[color_cnt])
            ax0.plot([t_kill]*2, [0,2], marker='X', color = color[color_cnt], linestyle = ':')
            ax0.plot([t_trigger] * 2, [0, 2], marker='X', color=color[color_cnt], linestyle=':')
            ax0.set_ylabel(r'$\tilde{\phi}$')
            ax0.legend()
            #ax0.set_xscale('Log')

            par = functools.partial(V, p=p_i)
            V_arr = np.array([*map(par, phi)])

            ax1.plot(phi, V_arr, label='p:%s' % (p_i),color = color[color_cnt])
            ax1.set_xlabel(r'$\tilde{\phi}$')
            ax1.set_ylabel(r'$\tilde{V}$')

            #ax2.plot(t, 1/2*(dphi/H_arr) ** 2 / 2)
            ax2.loglog(t, abs(dphi),color = color[color_cnt])
            ax2.set_ylabel(r'$\epsilon$')
            ax2.set_xlabel(r'$mt')
            ax2.set_ylim([1e-4,10])
            #ax2.set_xscale('Log')

            t_a = a_stack[idx_p, idx_H, 0]
            a = a_stack[idx_p, idx_H, 1]
            ax3.semilogx(t_a,log(a/a_i), color = color[color_cnt])
            ax3.set_ylabel(r'$N$')
            ax3.set_xlabel(r'$mt')

            color_cnt = color_cnt + 1

    plt.legend()
    plt.show()

    return


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

def scalar_animation(p, H_m, Phi_Stack, a_stack, kill_arr, trigger_arr, targ_p, targ_p_idxs):

    fig1 = plt.figure()
    color = cm.rainbow(np.linspace(0,1, len(targ_p)))
    color_cnt = 0
    
    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan = 2)
    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)

    for item_idx_p, idx_p in enumerate(targ_p_idxs):
        item_p = p[idx_p];print(item_p)
        for idx_H, item_H in enumerate(H_m):
            t_trigger = trigger_arr[idx_p][idx_H]
            t_kill = kill_arr[idx_p][idx_H]
            sol = Phi_Stack[idx_p][idx_H]
            t = sol[0]
            phi = sol[1]
            dphi = sol[2]

            ax0.plot(t, abs(phi), color = color[color_cnt])
            #ax0.plot([t_kill]*2, [0,2], marker='X', color = color_p[idx_p], linestyle = ':')
            #ax0.plot([t_trigger] * 2, [0, 2], marker='X', color=color_p[idx_p], linestyle=':')
            ax0.set_ylabel(r'$\tilde{\phi}$')
            ax0.legend()
            #ax0.set_yscale('Log')

            par = functools.partial(V, p=item_p)
            V_arr = np.array([*map(par, phi)])

            l, = ax1.plot([], [], label='p:%.1f' % (item_p),color = color[color_cnt])
            ax1.set_xlabel(r'$\tilde{\phi}$')
            ax1.set_ylabel(r'$\tilde{V}$')
            ax1.legend()
            
            data = np.array([phi, V_arr])#np.array(list(zip(phi, V_arr)));print(np.shape(data))

            line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                   interval=0.050, blit=True)

            par_H = functools.partial(H, p=item_p)
            args = np.array(list(zip(phi,dphi)))
            #H_arr = np.array([*map(par_H, args)])
            #ax2.plot(t, 1/2*(dphi/H_arr) ** 2 / 2)
            ax2.semilogy(t, abs(dphi), color = color[color_cnt])
            ax2.set_ylabel(r'$\epsilon$')
            ax2.set_xlabel(r'$mt')
            #ax2.set_yscale('Log')
            ax2.set_ylim([1e-4,10])

            color_cnt = color_cnt + 1

    plt.legend()
    plt.show()
    return

def scalar_plot2(p, H_m, Phi_Stack, kill_arr, trigger_arr, targ_p, targ_p_idxs):
    color = cm.rainbow(np.linspace(0,1, len(targ_p)))
    color_cnt = 0
    
    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan = 2)
    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)

    for item_idx_p, idx_p in enumerate(targ_p_idxs):
        item_p = p[idx_p];print(item_p)
        for idx_H, item_H in enumerate(H_m):
            t_trigger = trigger_arr[idx_p][idx_H]
            t_kill = kill_arr[idx_p][idx_H]
            sol = Phi_Stack[idx_p][idx_H]
            t = sol[0]
            phi = sol[1]
            dphi = sol[2]

            ax0.plot(t, abs(phi), color = color[color_cnt])
            ax0.plot([t_kill]*2, [0,2], marker='X', color = color[color_cnt], linestyle = ':')
            ax0.plot([t_trigger] * 2, [0, 2], marker='X', color=color[color_cnt], linestyle=':')
            ax0.set_ylabel(r'$\tilde{\phi}$')
            ax0.legend()
            #ax0.set_yscale('Log')

            par = functools.partial(V, p=item_p)
            V_arr = np.array([*map(par, phi)])

            ax1.plot(phi, V_arr, label='p:%.1f' % (item_p),color = color[color_cnt])
            ax1.set_xlabel(r'$\tilde{\phi}$')
            ax1.set_ylabel(r'$\tilde{V}$')
            ax1.legend()
            
            par_H = functools.partial(H, p=item_p)
            args = np.array(list(zip(phi,dphi)))
            #H_arr = np.array([*map(par_H, args)])
            #ax2.plot(t, 1/2*(dphi/H_arr) ** 2 / 2)
            ax2.semilogy(t, abs(dphi), color = color[color_cnt])
            ax2.set_ylabel(r'$\epsilon$')
            ax2.set_xlabel(r'$mt')
            #ax2.set_yscale('Log')
            ax2.set_ylim([1e-4,10])

            color_cnt = color_cnt + 1

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


def para_plot(p, k, alpha, Phi_Stack):

    for idx_p in np.arange(0,len(p),1):
        p_i = p[idx_p]

        sol = Phi_Stack[idx_p]
        t = sol[0]
        phi = sol[1]
        dphi = sol[2]
        par_a = functools.partial(a, p=p_i)
        a_stack = np.array([*map(par_a, t)])

        k_idxs = np.arange(0,len(k[idx_p]), 1)
        k_idxs = [-1]
        k_list = [k[idx_p][i] for i in k_idxs]
        print(k_list)
        for item_k in k_list:

            for item_alpha in [alpha[0], alpha[-1]]:

                para_a = item_k**2/a_stack**2
                para_b = item_k/a_stack*dphi*item_alpha

                plt.plot(t,para_a-para_b, label = r'$p:%.2f$'%p_i)
                plt.plot(t,para_a-para_b, label=r'$p:%.2f$' % p_i)
                #plt.plot(t, para_b, label=r'$p:%.2f$' % p_i)

    plt.xlabel(r'$mt$')
    plt.ylabel(r'$a$')
    plt.legend()
    plt.show()

    return



def gauge_plot(args):
    [p, alpha, k, H_m, a_stack, Phi_stack, Master_path, targ_alpha, targ_alpha_idxs] = args
    h = [-1,1]
    h_style = ['-', '--']
    #[plt.plot([],[],label = 'h:%s'%(h[i]), linestyle=h_style[i]) for i in [0,1]]
    #ax = [ 0, 0]
    p_idxs = range(len(p))#[0, -1]

    #targ_p = [6.5]
    targ_p = [6,6.5]

    p_idxs = [np.argmin(abs(targ-p)) for targ in targ_p]
    for sub_idx_p, idx_p in enumerate(p_idxs):
        item_p = p[idx_p]
        fig=plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex=ax1)

        for idx_H, item_H in enumerate(H_m):
            #ax[idx_p] = plt.subplot(2, 1, 1 + idx_p)#, sharex='col')
            t_max = Phi_stack[idx_p][idx_H][0][np.argmax(abs(Phi_stack[idx_p][idx_H][2]))]
            H_osc = H(item_p, t_max, item_H)

            for idx_h, item_h in enumerate(h):
                style_h = h_style[idx_h]
                ax1.plot([],[],linestyle=h_style[idx_h], label = r'h:%s'%item_h, color='k')
                for count_alpha, idx_alpha in enumerate(targ_alpha_idxs):
                    item_alpha = alpha[idx_alpha]
                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H)
                    stack = load(name, 'Raw', Master_path)
                    #plt_k_idxs = (np.arange(0,len(k[idx_p][idx_H]),1))
                    plt_k_idxs = np.round(np.linspace(650, len(k[idx_p][idx_H])-680, 2)).astype(int)
                    #plt_k_idxs = range(len(k[idx_p][idx_H]))#[12, -1]
                    color = cm.rainbow(np.linspace(0, 1, len(plt_k_idxs)))
                    for subidx_k, idx_k in enumerate(plt_k_idxs):
                        item_k = k[idx_p][idx_H][idx_k]

                        sub_stack = stack[idx_k]

                        t = sub_stack[0]
                        phi = sub_stack[1]
                        dphi = sub_stack[2]
                        A = sub_stack[3]
                        dA = sub_stack[4]
                        par_a = functools.partial(a, p=item_p, H_m=item_H)
                        H_map = np.array([H(p, i, H_m) for i in t]);print(absolute(item_k/(H_map[-1]*exp(t[-1]))))
                        
                        #A_rms = peak_arr[idx_p][idx_H][idx_alpha][idx_h][idx_k][2]
                        #t_cross = peak_arr[idx_p][idx_H][idx_alpha][idx_h][idx_k][4]
                        #t_rms = [(t_cross[i]+t_cross[i+1])/2 for i in np.arange(0,len(t_cross)-1,1)]

                        #print(len(t))
                        #plt.subplot(2,1,1, sharex='col')#+2*count_alpha)
                        ax1.semilogy(t, absolute(A), label = r'$ k:%.1f$'%(idx_k), color = color[subidx_k], linestyle = style_h)
                        #plt.semilogy(t_rms, A_rms, linestyle = '', marker = '+',color = color[subidx_k])
                        ax1.set_title(r'$p:%.1f\qquad H_{osc}/m:%.4f$'%(item_p, H_osc))
                        #plt.subplot(2,1,2, sharex='col')
                        if subidx_k == 0:
                            ax2.semilogy(t, abs(absolute(dphi)), label=r'$ k:%.2f$' % (item_k), color='green')
                        #ax2.semilogy(t, abs(item_k/(exp(t)*H_map))**2,linestyle = ':', color = color[subidx_k])
        #plt.title(r'p:%.1f' % item_p)
        #plt.subplot(2,1,2)
        ax2.set_xlabel(r'$\tilde{t}$')
        ax2.set_ylabel(r'$|d\tilde\phi /d\tilde{t}|$')
        #plt.subplot(2,1,1)
        ax1.set_ylabel(r'$|\sqrt{(2k)}\mathcal{A}_h|$')
        #ax1.legend()
        #ax[0].get_shared_x_axes().join(ax[0], ax[1])
        plt.show()

    return

def asymp_plot(args, stack):
    [p, alpha, k, H_m, [targ_alpha,targ_alpha_idxs], kill_arr, Master_path] = args
    h = [-1,1]
    h_style = ['--', ':']
    [plt.plot([],[],label = 'h:%s'%(h[i]), linestyle=h_style[i],color='k') for i in [0,1]]
    colors_alpha = cm.rainbow(np.linspace(0,1,len(targ_alpha_idxs)))
    [plt.plot([], [], color = colors_alpha[i], label = r'$\alpha %.1f$'%targ_alpha[i]) for i in np.arange(0,len(targ_alpha_idxs),1)]

    for idx_p, item_p in enumerate(p):
        for idx_h, item_h in enumerate(h):
            style_h = h_style[idx_h]

            for idx_H, item_H in enumerate(H_m):
                t_f = kill_arr[idx_p][idx_H]
                for idx_alpha, item_idx_alpha in enumerate(targ_alpha_idxs):
                    item_alpha = alpha[item_idx_alpha]

                    sub_stack = stack[idx_h, idx_p, idx_H, item_idx_alpha]

                    if len(sub_stack)!=0:
                        k = [item[0] for item in sub_stack]
                        A = [item[1] for item in sub_stack]
                        dA = [item[2] for item in sub_stack]
                        #print([k,A])
                        plt.loglog(absolute(k)/exp(t_f), absolute(A),
                                   linestyle=style_h, color=colors_alpha[idx_alpha])

    plt.xlabel(r'$k(ma_{\rm osc})^{-1}$')
    plt.ylabel(r'$|\sqrt{(2k)}\mathcal{A}_h(x_f)|$')
    plt.legend()
    plt.show()

    return


def asymp_plot2(args, stack):
    [p, alpha, k, H_m, [targ_p,targ_p_idxs], [targ_alpha, targ_alpha_idxs],kill_arr, Master_path, Phi_stack] = args
    h = [-1,1]
    h_style = ['-', ':']
    [plt.plot([],[],label = 'h:%s'%(['-','+'][i]), linestyle=h_style[i],color='k') for i in [0,1]]
    colors_p = cm.rainbow(np.linspace(0,1,len(targ_p_idxs)))
    colors_alpha = cm.rainbow(np.linspace(0, 1, len(targ_alpha_idxs))) 
    plt.plot([],[], label = 'p', linestyle = '')
    [plt.plot([], [], color = colors_p[i], label = r'$%.1f$'%targ_p[i])
     for i in np.arange(0,len(targ_p_idxs),1)]
    
    for idx_p, item_p_idx in enumerate(targ_p_idxs):
        item_p = p[item_p_idx]
        for idx_h, item_h in enumerate(h):
            style_h = h_style[idx_h]

            for idx_H, item_H in enumerate(H_m):
                t_f = kill_arr[item_p_idx][idx_H]

                
                t = Phi_stack[item_p_idx][idx_H][0]
                max_dphi = np.argmax(abs(Phi_stack[item_p_idx][idx_H][2]))
                t_max = t[max_dphi]
                H_osc = H(item_p, t_max, item_H)

                for idx_alpha, item_idx_alpha in enumerate(targ_alpha_idxs):
                    item_alpha = alpha[item_idx_alpha]

                    sub_stack = stack[idx_h, item_p_idx, idx_H, item_idx_alpha]

                    if len(sub_stack)!=0:
                        k = [item[0] for item in sub_stack]
                        A = [item[1] for item in sub_stack]
                        dA = [item[2] for item in sub_stack]
                        #print([k,A])
                        plt.loglog(absolute(k)/exp(t_max), absolute(A),
                                   linestyle=style_h, color=colors_p[idx_p])

    plt.xlabel(r'$k(ma_{\rm osc})^{-1}$')
    plt.ylabel(r'$|\sqrt{(2k)}\mathcal{A}_h(x_f)|$')
    plt.legend()
    plt.show()

    return



def Spectrum_mode_plot(args, asym_stack):

    [Master_path, alpha, p, k, H_m] = args

    h = [-1, 1]

    targ_alpha = [0.5, 2.0,4.0]

    plot_alpha_idxs = [np.argmin(abs(targ-alpha)) for targ in targ_alpha]

    colors = cm.rainbow(np.linspace(0, 1, len(plot_alpha_idxs)))

    gs = gridspec.GridSpec(3,3)

    ax0 = plt.subplot(gs[:,:-1])

    ax0.plot([], [], label='h: +1', linestyle='-', color='k')
    ax0.plot([], [], label='h: -1', linestyle='--', color='k')


    for idx_p, item_p in enumerate(p):
        for idx_H, item_h in enumerate(H_m):
            stack_alpha = []
            # for idx_alpha in np.arange(0,len(alpha),2):
            for sub_idx_alpha, idx_alpha in enumerate(plot_alpha_idxs):
                color = colors[sub_idx_alpha]
                alpha_i = alpha[idx_alpha]

                ax0.plot([], [], label='alpha:%.1f' % alpha_i, color=color)

                stack_h = []
                for idx_h in range(len(h)):
                    h_i = h[idx_h]

                    sub_stack = asym_stack[idx_h, idx_p, idx_alpha]
                    if len(sub_stack)!=0:
                        k = sub_stack[:, 0]
                        A = sub_stack[:, 1]
                        dA = sub_stack[:, 2]

                    if h_i == 1:
                        linestyle = '-'
                    else:
                        linestyle = '--'

                    # plt.subplot(211)
                    ax0.loglog(absolute(k), absolute(A), color=color, linestyle=linestyle)
                    # plt.subplot(212)
                    # plt.loglog(k/a(x_final), A_prime, label = 'h:%s'%h_i, color=color, linestyle = linestyle)

    ax0.set_ylabel(r'$\sqrt{2k}|\tilde{\mathcal{A}}_\pm(x_f)|$')
    ax0.set_xlabel(r'$k/ma(x_f)$')

    plt.legend()

    for idx_p, item_p in enumerate(p):
        ax11 = plt.subplot(gs[0:1, -1])
        ax12 = plt.subplot(gs[1:2, -1])
        #ax12.set_xlabel(r'$x$')
        #ax11.set_ylabel(r'$\sqrt{2k}\tilde\mathcal{A}_\pm$')
        ax11.plot([], [], color='k', linestyle='--', label='h:-1')
        ax11.plot([], [], color='k', linestyle='-', label='h:+1')
        # plt.subplots_adjust(wspace=0,hspace=0)
        ax11.legend()

        ax13= plt.subplot(gs[2:3, -1])
        ax13.set_xlabel(r'$x$')
        k_plot_idxs = [int(len(k)/10), int(len(k)/2), -1];print(len(k))
        #k_plot_idxs = [10,-1]
        color_k = cm.rainbow(np.linspace(0, 1, len(k_plot_idxs)))

        #[ax13.plot([], [], color=color_k[i], label='k:%.2e' % Decimal(k[k_plot_idxs[i]]))
        #for i in range(len(k_plot_idxs))]
        ax13.legend()

        for idx_alpha, item_alpha_idx in enumerate(plot_alpha_idxs):
            item_alpha = alpha[item_alpha_idx]

            ax = plt.subplot(gs[idx_alpha:idx_alpha+1,-1])
            ax.yaxis.tick_right()
            title = ax.set_title(r'$\alpha:%.2f$'%targ_alpha[idx_alpha])
            offset = np.array([1.2, 0.5])
            title.set_position(offset)
            title.set_rotation(90)

            for sub_idx_k, idx_k in enumerate(k_plot_idxs):

                for idx_h, item_h in enumerate(h):
                    h_i = h[idx_h]

                    if idx_h == 0:
                        h_style = '--'
                    else:
                        h_style = '-'

                    name = 'alpha:%.5f_h:%s_p:%.3f' % (item_alpha, item_h, item_p)
                    stack = load(name, 'Raw', Master_path)

                    sub_stack = stack[idx_k]

                    t = sub_stack[0]
                    phi = sub_stack[1]
                    dphi = sub_stack[2]
                    A = sub_stack[3]
                    dA = sub_stack[4]

                    ax.loglog(t, absolute(A), linestyle=h_style, color=color_k[sub_idx_k])

    plt.subplots_adjust(hspace=0, wspace =0)

    plt.show()

    return

def lim_temp_plot(args, stack):
    [p, alpha, k, H_m, a_stack, Phi_stack, kill_matrix] = args

    h = [-1, 1]

    plot_idx = 1
    for idx_p, item_p in enumerate(p):
        for idx_H, item_H in enumerate(H_m):
            t_f = kill_matrix[idx_p][idx_H]
            for idx_h, item_h in enumerate(h):
                plt.subplot(len(p)*len(H_m), 2, plot_idx)
                plot_idx = plot_idx+1
                z_1 = np.zeros((len(alpha), len(k[idx_p][idx_H])))

                for idx_alpha in range(len(alpha)):

                    alpha_i = alpha[idx_alpha]
                    for idx_k in range(len(k[idx_p][idx_H])):

                        sub_stack = stack[idx_h, idx_p, idx_H, idx_alpha, idx_k]

                        k_1, A_1, dA_1 = [sub_stack[i] for i in [0,1,2]]
                        x_final = t_f

                        z_1[idx_alpha, idx_k] = log(absolute(A_1))

                        color = log(absolute(A_1))#/x_final

                        if color < 0.:
                            color = 0.

                        z_1[idx_alpha, idx_k] = color

                z_1_nf = z_1;
                z_1 = np.flip(z_1, 0)

                x = k[idx_p][idx_H]/a(x_final,item_p, item_H)
                y = alpha

                plt.xlabel(r'$k/a_f$')
                plt.ylabel(r'$\alpha$')

                x, y = np.meshgrid(x, y)
                plt.contourf(x, y, z_1_nf, 20)

                #plt.xscale('Log')
                plt.title(r'p:%.2f H/m:%.3f h:%s'%(item_p,item_H, item_h))
                plt.xlabel(r'$k/a_f$')
                plt.ylabel(r'$\alpha$')
                plt.colorbar()

    plt.show()

    return


def contour_slices(args, stack):
    h = [-1, 1]

    [p, alpha, k, plt_intervals] = args
    print(np.shape(stack))

    for idx_p, item_p in enumerate(p):

        plot_idx = 0

        #min_col = 0.
        #max_col = norm_eq_spc();
        #print(min_col, max_col)

        fig, axes = plt.subplots(nrows=2, ncols=plt_intervals, sharex='col', sharey='row')

        for idx_h in range(len(h)):
            h_i = h[idx_h]
            for idx_N in np.arange(0,plt_intervals, 1):
                # for item_N in np.arange(0, int(cycles), 1):

                z_1 = np.zeros((len(alpha), len(k[idx_p])))
                hel = np.zeros((len(alpha), len(k[idx_p])))
                x_1 = []

                ax = axes.flat[plot_idx]

                for idx_alpha, item_alpha in enumerate(alpha):

                    alpha_i = alpha[idx_alpha]
                    for idx_k, item_k in enumerate(k[idx_p]):

                        peaks = stack[idx_p][idx_alpha][idx_h][idx_k]

                        item_N = np.round(np.linspace(0, len(peaks[1])-1, plt_intervals)).astype(int)[idx_N]
                        item_N = idx_N

                        A_rm_eqspc = peaks[1][item_N]

                        #if item_N == 1:
                            #color = log(A_rm_eqspc[item_N])#/(2*pi)

                        #else:
                            #color = log(A_rm_eqspc[item_N] )#/ A_rm_eqspc[item_N-1])/(2*pi)

                        #rms_peak_N = rms_peaks_N[idx_f][idx_alpha][idx_k][idx_h]
                        #color = log(absolute(rms_peak_N[5][item_N]))

                        #if color < min_col:
                            #color = 0.

                        z_1[idx_alpha, idx_k] = A_rm_eqspc

                z_1_nf = z_1;
                z_1 = np.flip(z_1, 0)
                y = alpha
                x_1 = k[idx_p]/a(peaks[4][-1], item_p)
                im = ax.contourf(x_1, y, z_1_nf,
                                 10)  # , vmin=min_col)#, vmax= max_col)  # [0.1,0.3,0.4,0.5,0.6,0.75])#, levels, ticker=ticker.LogLocator())
                # ax.contour(x, y, hel, levels=[0.5, 0.9], colors=['black','white'])
                ax.set_title(r'$(%s)N:%.1f$' % (['+','-'][idx_h], item_N))
                # ax.set_xscale('Log')
                # ax.clim(min_col, max_col)
                fig.colorbar(im, ax=ax)
                plot_idx = plot_idx + 1

        #fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
        #fig.colorbar(im, cax=cbar_ax)

        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.xlabel(r'$k/ma_N$')
        plt.ylabel(r'$\alpha$')

        plt.show()

    return


def contour_slices2(args):
    h = [-1, 1]

    [p, alpha, k, H_m, plt_intervals, targ_p, targ_p_idxs, kill_arr, Master_path] = args
    max_col = max_rms_N2(p, H_m, alpha, k, targ_p, targ_p_idxs, Master_path)
    #print(max_col)
    for idx_p in targ_p_idxs:
        item_p = p[idx_p]
        name = 'p:%.3f' % (item_p)
        stack = load(name, 'Peaks2',Master_path)

        for idx_H, item_H in enumerate(H_m):
            t_f = kill_arr[idx_p][idx_H]

            plot_idx = 0

            min_col = 0.
            #max_col = 0.
            print(min_col, max_col)

            cycles = len(stack[idx_H][0][0][0][1])
            #cycles = 10

            fig, axes = plt.subplots(nrows=2, ncols=plt_intervals, sharex='col', sharey='row')
            im = []
            for idx_h in range(len(h)):
                h_i = h[idx_h]
                plt_N = np.round(np.linspace(0, plt_intervals, plt_intervals)).astype(int);print(plt_N)
                for idx_N in plt_N:
                    # for item_N in np.arange(0, int(cycles), 1):
                    item_N = idx_N
                    z_1 = np.zeros((len(alpha), len(k[idx_p][idx_H])))
                    hel = np.zeros((len(alpha), len(k[idx_p][idx_H])))
                    x_1 = []

                    ax = axes.flat[plot_idx]

                    for idx_alpha, item_alpha in enumerate(alpha):

                        alpha_i = alpha[idx_alpha]
                        for idx_k, item_k in enumerate(k[idx_p][idx_H]):

                            peaks = stack[idx_H][idx_alpha][idx_h][idx_k]
                            t_cross = peaks[0][idx_N]

                            if len(peaks[1]) >= plt_intervals:
                                #A_rm_eqspc = log(peaks[2][idx_N+2]/peaks[2][idx_N])
                                #color = A_rm_eqspc
                                mu = peaks[1][idx_N]
                                color = mu
                                #print(color)
                                #A_rm_eqspc = peaks[2][idx_N]
                                #print(A_rm_eqspc)
                                #color = log(A_rm_eqspc)

                                #if item_N == 1:
                                    #color = log(A_rm_eqspc[item_N])#/(2*pi)

                                #else:
                                    #color = log(A_rm_eqspc[item_N] )#/ A_rm_eqspc[item_N-1])/(2*pi)

                                #rms_peak_N = rms_peaks_N[idx_f][idx_alpha][idx_k][idx_h]
                                #color = log(absolute(rms_peak_N[5][item_N]))

                                if color < min_col:
                                    color = 0.

                                #if color < min_col:
                                    #min_col = color
                                #if color > max_col:
                                    #max_col = color

                            else:
                                color = None


                            z_1[idx_alpha, idx_k] = color

                    z_1_nf = z_1;
                    z_1 = np.flip(z_1, 0)
                    y = alpha
                    #x_N = peaks[4][idx_N+1]
                    x_N = t_cross
                    x_1 = k[idx_p][idx_H]/exp(x_N)
                    img = im.append(ax.contourf(x_1, y, z_1_nf,10))
                                     #10, vmin=min_col, vmax= max_col))  # [0.1,0.3,0.4,0.5,0.6,0.75])#, levels, ticker=ticker.LogLocator())
                    # ax.contour(x, y, hel, levels=[0.5, 0.9], colors=['black','white'])
                    ax.set_title(r'$(%s)N:%.1f$' % (['-','+'][idx_h], item_N+1))
                    ax.set_xlim([x_1[0], x_1[-1]])
                    #ax.set_xscale('Log')
                    # ax.clim(min_col, max_col)
                    if img != None:
                        fig.colorbar(img, ax=ax)
                    plot_idx = plot_idx + 1

            #fig.subplots_adjust(right=0.8)
            #cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
            #fig.colorbar(im[5], cax=cbar_ax)

            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            plt.grid(False)
            plt.xlabel(r'$k/ma_N$')
            plt.ylabel(r'$\alpha$')

            plt.show()

    return

def mu_evo_plot(args, stack):
    h = [-1, 1]

    [p, alpha, k, H_m, plt_intervals, targ_alpha, targ_alpha_idxs] = args

    for idx_p, item_p in enumerate(p):
        for idx_H, item_H in enumerate(H_m):
            plot_idx = 0

            # min_col = 0.
            # max_col = norm_eq_spc();
            # print(min_col, max_col)

            cycles = len(stack[idx_p][idx_H][0][0][0][1])

            for idx_h in range(len(h)):
                h_i = h[idx_h]
                plt_N = np.round(np.linspace(0, cycles - 1, plt_intervals)).astype(int);
                plt_N = [int(cycles/6),int(cycles/2),cycles-1]
                color_N = cm.rainbow(np.linspace(0,1,len(plt_N)))
                for count_N, idx_N in enumerate(plt_N):
                    # for item_N in np.arange(0, int(cycles), 1):
                    item_N = idx_N
                    z_1 = np.zeros((len(alpha), len(k[idx_p][idx_H])))
                    hel = np.zeros((len(alpha), len(k[idx_p][idx_H])))
                    x_1 = []

                    for idx_alpha, item_idx_alpha in enumerate(targ_alpha_idxs):
                        item_alpha = alpha[item_idx_alpha]
                        peaks = stack[idx_p][idx_H][item_idx_alpha][idx_h]

                        A_rm_k = [peaks[i][2][idx_N] for i in range(len(k[idx_p][idx_H]))]
                        x_N = peaks[0][4][-1]
                        plt.subplot(2,2,1+idx_alpha)
                        plt.loglog(k[idx_p][idx_H]/x_N, A_rm_k, linestyle = ['--','-'][idx_h], color = color_N[count_N])
                        plt.title(r'$\alpha:%.1f$'%item_alpha)
            [plt.plot([],[],color = color_N[i], label = r'N:%i'%plt_N[i]) for i in np.arange(0,len(plt_N),1)]
            plt.legend()
            plt.show()
    return


def pre_osc_plot(args):
    h = [-1, 1]

    [p, alpha, k, H_m, plt_intervals, targ_alpha, targ_alpha_idxs, targ_p, targ_p_idxs, Phi_stack, kill_arr, Master_path] = args

    color = cm.rainbow(np.linspace(0, 1, len(targ_alpha) * len(H_m) * len(targ_p)))
    color_cnt = 0
    for idx_p, item_p_idx in enumerate(targ_p_idxs):
        item_p = p[item_p_idx]
        for idx_H, item_H in enumerate(H_m):
            t_f = kill_arr[item_p_idx][idx_H]

            t = Phi_stack[item_p_idx][idx_H][0]
            max_dphi = np.argmax(abs(Phi_stack[item_p_idx][idx_H][2]))
            #max_dphi = [np.argmin(abs(max(abs(Phi_stack[idx_p][idx_H][2]))*i - abs(Phi_stack[idx_p][idx_H][2]))) for i in [0.0]]
            t_max = t[max_dphi]

            t_slice = [t_max*0.999, t_max, t_max * 1.00005, t_max*1.0005]

            gs = gridspec.GridSpec(3,len(t_slice))
            ax2 = plt.subplot(gs[-1, :])

            ax2.semilogy(t, abs(Phi_stack[item_p_idx][idx_H][2]))
            [ax2.semilogy([t_i]*2, [1,1e-3],color = 'k', marker = 'x') for t_i in t_slice]

            print(t_slice)
            for idx_alpha, item_idx_alpha in enumerate(targ_alpha_idxs):
                item_alpha = alpha[item_idx_alpha]

                for idx_h, item_h in enumerate(h):

                    name = 'alpha:%.5f_h:%s_p:%.3f_H_%.3f' % (item_alpha, item_h, item_p, item_H)
                    stack = load(name, 'Raw', Master_path)

                    for idx_t, item_t in enumerate(t_slice):

                        ax1 = plt.subplot(gs[:-1, idx_t])
                        #plt.subplot(1,3,idx_t+1)
                        A_k = []
                        for idx_k, item_k in enumerate(k[item_p_idx][idx_H]):
                            A_i = exp(1j*-0.001*item_k)

                            t = stack[idx_k][0]
                            # cross_idx = np.argwhere(np.diff(np.sign(np.real(dphi) - [0] * len(dphi)))).flatten()
                            A = stack[idx_k][3]
                            if len(t) != 0 and t[0] <= item_t:
                                A_max = mapper(item_t, t, A)
                            else:
                                A_max = 1.
    
                            A_k.append(absolute(A_max)-absolute(A_i))
    
                        ax1.loglog(k[item_p_idx][idx_H] / exp(t_f), abs(A_k), linestyle=['--', '-'][idx_h], color = 'blue')#,
                                       #color=color[color_cnt])
                color_cnt = color_cnt + 1
            # [plt.plot([],[],color = color_alpha[i], label = r'$\alpha:%i$'%targ_alpha[i]) for i in np.arange(0,len(targ_alpha),1)]
        ax1.set_title(r'$p:%.1f$'%item_p)
        ax2.set_ylim([1e-4,10])
        plt.legend()
        plt.show()
    return


def mean_mu_contour(args, stack):
    h = [-1, 1]

    [p, alpha, k, H_m, plt_intervals] = args
    print(np.shape(stack))

    for idx_p, item_p in enumerate(p):
        for idx_H, item_h in enumerate(H_m):
            plot_idx = 0

            #min_col = 0.
            #max_col = norm_eq_spc();
            #print(min_col, max_col)

            fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col', sharey='row')

            for idx_h in range(len(h)):
                h_i = h[idx_h]

                z_1 = np.zeros((len(alpha), len(k[idx_p][idx_H])))
                hel = np.zeros((len(alpha), len(k[idx_p][idx_H])))
                # plt.subplot(2, cycles, plot_idx)
                ax = axes.flat[plot_idx]
                plot_idx = plot_idx + 1

                # alpha_lim_idx = np.argmin(abs(alpha - lim_alpha))
                stack_alpha = []
                # for idx_alpha in range(len(alpha)):
                for idx_alpha, item_alpha in enumerate(alpha):

                    alpha_i = alpha[idx_alpha]
                    for idx_k, item_k in enumerate(k[idx_p][idx_H]):
                        peaks = stack[idx_p][idx_H][idx_alpha][idx_h][idx_k]
                        mean_mu = peaks[3]

                        A_rms = peaks[2]
                        color = log(A_rms[-1])#/A_rms[0])

                        #color = mean_mu

                        z_1[idx_alpha, idx_k] = color
                # print(np.shape(z_1))
                z_1_nf = z_1;
                z_1 = np.flip(z_1, 0)
                # plt.subplot(212)

                x = k[idx_p][idx_H] #/ a(x_N, item_p)
                y = alpha

                # plt.imshow(z_1, interpolation = 'sinc',
                #        extent = (x[0],x[-1],y[0],y[-1]), aspect = 'auto')
                # plt.colorbar()

                # x,y = np.meshgrid(x,y)
                # levels = np.arange(0.,1.,0.1)
                im = ax.contourf(x, y, z_1_nf,
                                 10)  # , vmin=min_col)#, vmax= max_col)  # [0.1,0.3,0.4,0.5,0.6,0.75])#, levels, ticker=ticker.LogLocator())
                # ax.contour(x, y, hel, levels=[0.5, 0.9], colors=['black','white'])
                #ax.set_title(r'$(%s)N:%.1f$' % (['+', '-'][idx_h], item_N))
                # ax.set_xscale('Log')
                # ax.clim(min_col, max_col)
                fig.colorbar(im, ax=ax)

            #fig.subplots_adjust(right=0.8)
            #cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
            #fig.colorbar(im, cax=cbar_ax)

            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            plt.grid(False)
            plt.xlabel(r'$k/ma_N$')
            plt.ylabel(r'$\alpha$')

            plt.show()

    return


def hel_temp_plot(args, stack):
    h = [-1, 1]
    [p, alpha, k, H_m, a_stack, Phi_stack, kill_arr] = args

    fig, axes = plt.subplots(nrows=1, ncols=5, sharex='col', sharey='row')

    plot_idx = 0
    for idx_p in [0,5,15,20,-1]:
        for idx_H, item_H in enumerate(H_m):
            t_f = kill_arr[idx_p][idx_H]
            a_f = exp(t_f)

            ax = axes.flat[plot_idx]
            plot_idx = plot_idx + 1

            z = np.zeros((len(alpha), len(k[idx_p][idx_H])))

            for idx_alpha, item_alpha in enumerate(alpha):
                for idx_k, item_k in enumerate(k[idx_p][idx_H]):

                    sub_stack = [np.array(stack[i, idx_p, idx_H, idx_alpha])[idx_k,1] for i in [0,1]]

                    A_0 = absolute(sub_stack[0])
                    A_1 = absolute(sub_stack[1])

                    diff = abs(A_0 ** 2 - A_1 ** 2)
                    sum = abs(A_0 ** 2 + A_1 ** 2)

                    z[idx_alpha,idx_k] = diff/sum
        #z = np.flip(z,0)
        x = k[idx_p][idx_H]/a_f
        y = alpha
        ax.contourf(x, y, z,10, vmin = 0, vmax = 1)
        #ax.colorbar()
        plt.xscale('Log')
    plt.show()
    return


def peak_k_plot(args, asymp_stack):
    [p, alpha, k_arr, H_m, [targ_alpha,targ_alpha_idxs], kill_arr, Phi_stack, Master_path] = args
    h = [-1,1]
    h_style = ['--', ':']
    colors_alpha = cm.rainbow(np.linspace(0,1,len(targ_alpha_idxs)))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    [ax1.plot([],[],label = r'%.1f'%(targ_alpha[i]), color = colors_alpha[i]) for i in np.arange(0,len(targ_alpha), 1)]
    ax2 = ax1.twiny()
    for idx_alpha, item_idx_alpha in enumerate(targ_alpha_idxs):
    
        for idx_H, item_H in enumerate(H_m):
            stack_p = []
            for idx_p, item_p in enumerate(p):        

                t = Phi_stack[idx_p][idx_H][0]
                max_dphi = np.argmax(abs(Phi_stack[idx_p][idx_H][2]))
                t_max = t[max_dphi]
                H_osc = H(item_p, t_max, item_H)

                item_alpha = alpha[item_idx_alpha]
                t_f = kill_arr[idx_p][idx_H]

                sub_stack = [asymp_stack[idx_h, idx_p, idx_H, item_idx_alpha] for idx_h in [0,1]]

                k = np.array([item[0] for item in sub_stack[0]])/exp(t_f)
                A = [[item[1] for item in sub_stack[i]] for i in [0,1]]
                dA = [[item[2] for item in sub_stack[i]] for i in [0,1]]

                Integrand_diff = k ** 2 * abs(np.array(A[0])**2-np.array(A[1])**2)
                Integrand_sum = (abs(np.array(A[0])**2+np.array(A[1])**2)) * k ** 2
                integrated = trapz(Integrand_diff, x=k)/trapz(Integrand_sum, x=k)
                
                stack_p.append([item_p, integrated, H_osc])
            #print(np.shape(stack_p))
            stack_p = np.array(stack_p);print(stack_p[:,2][0], stack_p[:,2][-1])
            ax1.loglog(stack_p[:,2], stack_p[:,1], color = colors_alpha[idx_alpha], marker = '+')
            ax2.loglog(stack_p[:,0], stack_p[:,1], linestyle = '')
    ax2.invert_xaxis()
    #ax1.set_xlim(stack_p[:,2][-1],stack_p[:,2][0])
    ax1.legend()
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
        plt.contourf(x, y, growth, 5)
        plt.colorbar()
        cs = plt.contour(x, y, hel, levels=[0.5, 0.9], colors=['white', 'black'])
        plt.clabel(cs, fontsize=10, inline=1, fmt='%.1f')

        plt.ylabel(r'$\alpha$')
        plt.xlabel(r'$k/a_f$')
        # plt.colorbar()

    plt.show()

    return
