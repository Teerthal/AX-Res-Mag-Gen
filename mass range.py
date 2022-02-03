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
import os
from shutil import copyfile
import matplotlib.cm as cm
import matplotlib.image as mpimg
#import cv2
from matplotlib.colors import LogNorm

# from numdifftools import Derivative
flatten = np.ndarray.flatten
# from scipy.interpolate import UnivariateSpline
# from scipy import interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'
    , 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

start = time.time()

plt.rcParams['axes.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['animation.ffmpeg_path'] ='/usr/bin/ffmpeg'

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

def cond(T):

    sigma = 100*T

    return sigma

def H_rad(T):

    h = 0.7
    omega_rad = 2.471*1e-5/h**2
    H_0 = 2.1331e-42*h
    T_0 = 0.2348e-3

    H = H_0*(omega_rad*T/T_0)

    return H

def plot_para():

    T = np.geomspace(1e6,1e12, 1e4)

    cond_map = np.array([*map(cond,T)])
    H_map = np.array([*map(H_rad,T)])

    plt.loglog(T, cond_map, label = r'$\sigma_{el}$')
    #plt.loglog(T, H_map, label = r'$H_\gamma$')

    plt.xlabel(r'$T$(eV)')
    plt.legend()
    plt.show()

    return

#plot_para()

def B_max_hel(lamda_0):

    M_PL = 2.4e18

    T_0 = 0.2348e-3*1e-9
    z_rec = 1100
    Mpc = 1e39*1/6.4
    GeV_2 = 1/6.8*1e20

    B = lamda_0*(T_0**4/M_PL)*Mpc*GeV_2

    return B


def B_present_hel():

    T_0 = 0.2348e-3*1e-9
    M_PL = 2.4e18

    return

def B_present_ad(lamda_0,f,m):

    B = 1.5e-57*(f/m)*1/(lamda_0)*1e9

    return  B

def mash_plot():

    lamda_0 = 1
    T = np.geomspace(1e3,1e12, 1e4)*1e-9

    m_min = cond(T[0])*100
    m_max = cond(T[-1])*100

    f = 1e15

    cond_map = np.array([*map(cond,T)])
    H_map = np.array([*map(H_rad,T)])

    plt.loglog(T, cond_map, label = r'$\sigma_{el}$[GeV]')

    plt.loglog([T[0], T[-1]], [B_present_ad(lamda_0,f,m_min)]*2, label = r'$B_{*,0}(m=100\sigma(T_{min}))$[nG]')
    plt.loglog([T[0], T[-1]], [B_present_ad(lamda_0,f,m_max)]*2, label = r'$B_{*,0}(m=100\sigma(T_{max}))$[nG]')

    plt.loglog([T[0], T[-1]], [B_max_hel(1)]*2, label=r'$B_{hel,0}$[G]')
    #    plt.loglog([T[0],T[-1]], [1,1], label=r'GeV')

    plt.xlabel(r'$T$(GeV)')

    plt.legend()
    plt.show()
    return

#mash_plot()

def rho(T):

    T_0 = 0.2348e-3*1e-9
    a = T_0/T
    h = 0.7

    omega_r_0 = 9.05e-5
    omega_m_0 = 0.3
    rho_c = 8.1*h**2*1e-47      #in GeV^4
    rho = omega_r_0*rho_c/a**4+omega_m_0*rho_c/a**3

    rho = T**4

    return rho

def mass_range_plot():
    T = np.geomspace(1,1e20, 1e4)
    rho_map = np.array([*map(rho,T)])*1e36
    r = 1
    f = 1e27
    print(max(rho_map),min(rho_map))
    m_max = rho_map**0.5/f*r**-0.5
    m_min = 100*T

    H_map = np.array([*map(H_rad,T)])
    m_max_2 = H_map*1e6*1e9

    plt.plot(T,m_min, label=r'$\sigma=100T$')
    plt.plot(T,m_max, label=r'$\sqrt{(\rho_*)}/f$')
    plt.plot(T,m_max_2, label=r'check')

    plt.fill_between(T,m_min,m_max, where=m_max>=m_min)
    plt.xscale('Log')
    plt.yscale('Log')
    plt.legend()

    plt.show()
    return

#mass_range_plot()

def mass_f(f,T,r):
    T_eq = 1e-9
    m = (rho(T))**0.5*r**0.5/f*T_eq**0.5
    return m

def mass_f_plot():
    r=1e-4
    GeV = 1/1e9
    T = np.array([ 1e-1,1e5])      #T in GeV
    f = np.geomspace(1e2,1e16,1e4)            #f in GeV
    colors = cm.rainbow(np.linspace(0, 1, len(T)))

    for idx_T in range(len(T)):
        t = T[idx_T]

        plt.plot([],[], label='Log(T/GeV):%.1f'%np.log10(t), color=colors[idx_T])

        par = functools.partial(mass_f, T=t, r=r)
        arr = np.array([*map(par,f)])
        plt.loglog(f,arr, color=colors[idx_T])

        m_min = [100*t]*len(f)
        plt.loglog(f, m_min, color=colors[idx_T])
        #plt.fill_between(f,m_min,arr, where=m_min<=arr, color = colors[idx_T], alpha=0.5)

    #plt.fill_between([f[0], f[-1]], [100 * (T[0]), 100 * (T[0])], [100 * (T[-1]), 100 * (T[-1])], alpha=.5)
    plt.yscale('Log')
    plt.xscale('Log')
    plt.xlabel('f[GeV]')
    plt.ylabel('m[GeV]')
    plt.legend()
    plt.show()

    return

#mass_f_plot()

def mass_f(T,f,r):
    T_eq = 1e-9
    m = (rho(T))**0.5*r**0.5/f*(T_eq/T)**0.5
    m = T_eq**(1/2)*T**(3/2)/f
    m = T**(3/2)*1/f*8*1e-5

    if T>1.9*1e-9:
        m = 0.81*1/f*T**2

    else:
        m = 3.7e-5*1/f*T**(3/2)

    return m


def f_mass(T,m):
    T_eq = 1e-9

    if T>1.9*1e-9:
        f = 0.81*1/m*T**2

    else:
        f = 3.7e-5*1/m*T**(3/2)

    return f

def sigma_T(T):

    #print((0.511*1e-3*100/9)**2)

    if T >= (0.511*1e-3*100/9)**2:#0.511*1e-3:#
        sigma = 100*T
        sigma = (0.1/T - 0.001/T)**(-1)

    #elif 1 > T >= 1e-4:
        #sigma = 100*T

    else:
        sigma = 0.4*4*pi*(1e-9)**0.5/(T**1.5)
        sigma = 3.2*10**4*T**(3/2)
        sigma = T**(3/2)/(0.511*1e-3)
        print((0.511*1e-3))
    #sigma = 3.2*10**4*T**(3/2)+100*T/9

    return sigma

def H(T):
    T_0 = 0.2348e-3*1e-9
    H_0 = 1e-42
    omega_M = 0.27
    omega_rel = 8.24*1e-5
    omega_lambda = 0.73

    H = H_0 * sqrt( omega_rel*(T/T_0)**4 + omega_M*(T/T_0)**3 + omega_lambda )
    return H

def mass_T_plot():
    r = 1
    GeV = 1 / 1e9
    T_0 = 0.2348e-3 * 1e-9
    f = np.array([1])  # T in GeV
    T = np.geomspace(T_0, 1e-3, 1e4)  # f in GeV
    colors = cm.rainbow(np.linspace(0, 1, len(f)))
    T_eq = 1e-9
    sig_map = np.array([*map(sigma_T, T)]);plt.loglog(T,sig_map , linewidth = 2.)
    #T_1 = np.geomspace(1e-6, T[-1], 1e4); T_2 = np.geomspace(T[0], 0.511*1e-3, 1e4)
    #sig_1 = (0.1/T_1 - 0.001/T_1)**(-1); sig_2 = T_2**(3/2)/(0.511*1e-3)
    #plt.loglog(T_1, sig_1, color = 'green');plt.loglog(T_2, sig_2, color = 'green')

    H_map = np.array([*map(H, T)]); plt.loglog(T, H_map, linewidth = 2. )
    #plt.loglog(T, 3.2*10**4*T**(3/2)+100*T/9)
    print()
    for idx_f in range(len(f)):
        F = f[idx_f]

        plt.plot([], [], label='Log(f/GeV):%.1f' % np.log10(F), color=colors[idx_f] , linewidth = 2.)

        par = functools.partial(mass_f, f=F, r=r)
        arr = np.array([*map(par, T)])
        m = T_eq ** 0.5 * T ** 1.5 / F
        plt.loglog(T, arr, color=colors[idx_f] , linewidth = 2.)
        plt.loglog(T, 3.7e-5*1/f*T**(3/2), color = colors[idx_f], linestyle = '--', linewidth = 2.)

        lim_T_0 = mass_f(T_0, F, r=r)
        lim_T_f = H_map[-1]

        plt.fill_between(T, sig_map, arr, where = arr >= sig_map, interpolate=True, facecolor='lightblue')#, hatch='+')
        plt.fill_between(T, arr, [arr[-1]*1e4]*len(arr), where= [arr[-1]*1e4]*len(arr)>=arr, interpolate=True, facecolor = 'white', hatch = '/')
        plt.fill_between(T, arr, H_map, where=arr >= H_map, interpolate=True,
                         facecolor='lightgreen')
        plt.fill_between(T, arr, [lim_T_f]*len(T), where = arr >= [lim_T_f]*len(T), interpolate=True,
                         facecolor='green')



    m_min = 10* T
    #plt.loglog(T, m_min, color='k', linestyle='--', label=r'$\sigma$')

    plt.yscale('Log')
    plt.xscale('Log')
    plt.xlabel('T[GeV]')
    plt.ylabel('m[GeV]')
    #plt.legend()
    plt.show()

    return

mass_T_plot()


def mass_f_DM(T,f,r):
    T_eq = 1e-9
    m = (rho(T))**0.5*r**0.5/f*(T_eq/T)**0.5
    m = T_eq**(1/2)*T**(3/2)/f
    m = T**(3/2)*1/f*8*1e-5

    m = 3.7e-5*1/f*T**(3/2)

    return m


def allowed_mass():

    img = mpimg.imread("/home/teerthal/Dropbox/CMB_injection/ALP_constraint_2.png")
    print(img)
    #plt.imshow(img, extent=[-15, 10.5, -17.2, 0.25])
    omega_a = 0.1
    T_BBN = 1e6*1e-9
    T_CMB = 2.725 * 8.62*1e-5 * 1e-9
    T_Rec = 1e-9
    f = np.geomspace(1e-6,1e18,1e4)     #in GeV
    H_osc = 0.1     ##Parametric amplification efficiency parameter H_osc/m


    exc_arr = []; exc_arr_2 = []
    max_arr_BBN = []
    max_arr_rec = []
    for F in f:

        lim_T_f = H(T_BBN)/H_osc; lim_T_f_2 = H(T_Rec)/H_osc
        exc_arr.append(lim_T_f); exc_arr_2.append(lim_T_f_2)

        max_mass = mass_f_DM(T_BBN, F,omega_a)
        max_arr_BBN.append(max_mass)

        max_mass_2 = mass_f_DM(T_Rec, F, omega_a)
        max_arr_rec.append(max_mass_2)

    min_1_m = 1e7
    max_1_f = (min_1_m**3*1e-2/6.582*1e16/64*pi)**-0.5
    print(max_1_f)

    max_2_m = 10
    min_2_f = (min_1_m**3*1e24/6.582*1e16/64*pi)**-0.5
    print(min_2_f)

    max_arr_rec = np.array(max_arr_rec)*1e9; max_arr_BBN = np.array(max_arr_BBN)*1e9
    exc_arr = np.array(exc_arr)*1e9; exc_arr_2 = np.array(exc_arr_2)*1e9

    plt.fill_between(np.geomspace(min_1_m, max_2_m, 100), max_1_f, min_2_f, interpolate=True, facecolor='white' , hatch='/')

    plt.loglog([min_1_m]*2, [max_1_f, min_2_f], color = 'k')
    plt.loglog([min_1_m, max_2_m], [max_1_f]*2, color= 'k')

    plt.loglog([max_2_m]*2, [max_1_f, min_2_f], color = 'k')
    plt.loglog([min_1_m, max_2_m], [min_2_f]*2, color = 'k')

    plt.loglog(max_arr_rec, 1/f, color = 'blue'); plt.loglog(exc_arr_2, 1/f, color='blue')
    plt.loglog(max_arr_BBN, 1 / f, color='green'); plt.loglog(exc_arr, 1/f, color='green')

    plt.fill_betweenx(1 / f,exc_arr, max_arr_BBN,  where=exc_arr <= max_arr_BBN, interpolate=True, color='lightgreen', alpha=0.5, hatch='solid')
    plt.fill_betweenx(1/f, exc_arr_2, max_arr_rec,  where=max_arr_rec>=exc_arr_2, interpolate=True, color='lightblue', facecolor = 'red', hatch='solid')

    plt.xscale('Log')
    plt.yscale('Log')
    plt.show()
    return

#allowed_mass()


def allowed_mass_T():

    img = mpimg.imread("/home/teerthal/Dropbox/CMB_injection/ALP_constraint_2.png")
    print(img)
    #plt.imshow(img, extent=[-15, 10.5, -17.2, 0.25])
    omega_a = 0.1
    T_BBN = 1e6*1e-9
    T_CMB = 2.725 * 8.62*1e-5 * 1e-9
    T_Rec = T_CMB
    f = np.geomspace(1e-6,1e18,1e4)     #in GeV
    H_osc = 0.1     ##Parametric amplification efficiency parameter H_osc/m

    H_rec = H(T_Rec)
    H_BBN = H(T_BBN)

    T_range = np.geomspace(T_Rec, T_BBN, 1e1)
    H_range = np.array([ H(x)/H_osc for x in T_range ])*1e9
    f_range = np.array([f_mass(t, m) for (t,m) in zip(T_range,H_range)])*1e9

    for T in T_range:

        exc_arr = []; exc_arr_2 = []
        max_arr_BBN = []
        max_arr_rec = []

        arr = []
        for F in f:

            lim_T_f = H(T_BBN)/H_osc; lim_T_f_2 = H(T_Rec)/H_osc
            exc_arr.append(lim_T_f); exc_arr_2.append(lim_T_f_2)

            max_mass = mass_f(T_BBN, F,omega_a)
            max_arr_BBN.append(max_mass)

            max_mass_2 = mass_f(T_Rec, F, omega_a)
            max_arr_rec.append(max_mass_2)

            exc = H(T)/H_osc
            ED_bound = mass_f(T, F, 0.1)
            arr.append([exc ,ED_bound])
        stack  =np.array(arr)
        #plt.fill_betweenx(1/f, stack[:, 0]*1e9, stack[:,1]*1e9, where= stack[:,0]<=stack[:,1], color = 'lightblue')

    min_1_m = 1e7
    max_1_f = (min_1_m**3*1e-2/6.582*1e16/64*pi)**-0.5
    print(max_1_f)

    max_2_m = 10
    min_2_f = (min_1_m**3*1e24/6.582*1e16/64*pi)**-0.5
    print(min_2_f)

    max_arr_rec = np.array(max_arr_rec)*1e9; max_arr_BBN = np.array(max_arr_BBN)*1e9
    exc_arr = np.array(exc_arr)*1e9; exc_arr_2 = np.array(exc_arr_2)*1e9

    #plt.fill_between(np.geomspace(min_1_m, max_2_m, 100), max_1_f, min_2_f, interpolate=True, facecolor='white' , hatch='/')

    #plt.loglog([min_1_m]*2, [max_1_f, min_2_f], color = 'k')
    #plt.loglog([min_1_m, max_2_m], [max_1_f]*2, color= 'k')

    #plt.loglog([max_2_m]*2, [max_1_f, min_2_f], color = 'k')
    #plt.loglog([min_1_m, max_2_m], [min_2_f]*2, color = 'k')

    #plt.fill_betweenx(1/f, max_arr_rec, max_arr_BBN, color='tomato')

    plt.loglog(H_range, 1/f_range)

    #plt.loglog(max_arr_rec, 1/f, color = 'blue');
    plt.loglog([exc_arr_2[0], max_arr_BBN[0]], [1/f[0]]*2)
    plt.loglog(exc_arr_2, 1/f)
    plt.loglog(max_arr_BBN, 1 / f);# plt.loglog(exc_arr, 1/f, color='green')

    x1 = np.append( H_range , exc_arr_2 ); x2 = np.append([exc_arr_2[0], max_arr_BBN[0]], max_arr_BBN)
    y1 = np.append(1/f_range, 1/f); y2 = np.append([1/f[0]]*2, 1/f)
    plt.fill(x1,y1, color = 'lightblue'); plt.fill(x2,y2, color = 'lightblue')

    #plt.fill_betweenx(1 / f,exc_arr, max_arr_BBN,  where=exc_arr <= max_arr_BBN, interpolate=True, color='lightgreen', alpha=0.5, hatch='solid')
    #plt.fill_betweenx(1/f, exc_arr_2, max_arr_rec,  where=max_arr_rec>=exc_arr_2, interpolate=True, color='lightblue', facecolor = 'red', hatch='solid')

    plt.xscale('Log')
    plt.yscale('Log')
    plt.show()
    return

allowed_mass_T()

def t_g(T):
    T_0 = 0.2348e-3*1e-9
    a_f = T_0 / T
    a_i = 1e-30
    a = np.geomspace(a_i, a_f,1e5)
    H_0 = 1e-42     #GeV
    integrand = 1/(H_0*sqrt(0.3/a**3))
    t = trapz(integrand,a)

def axial_dilution():

    #GeV units only

    s = 1/6.58*1e25             #seconds in GeV_-1

    rho_ini = (m*f)**2

    t_decay = 130*m**-3*1e-24*f**2*s

    a_ratio = (1-t_decay/t_g(T))**2
    return



