import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

plt.rcParams['axes.labelsize'] = 20
pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

p = 2

N_final = 70.
N_initial = 0.

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

delta = 6

def a(N):
    return exp(N)


def phi(N):
    return sqrt(2 * p * (N_final - N_initial) + phi_final ** 2)


def phi_prime(N):
    return -p / phi(N)


def epsilon(N):
    return (phi_prime(N)) ** 2 / 2.

def H(N):
    return H_initial * sqrt((3. - epsilon(N)) / 2. * (phi(N) / phi_initial) ** 2)

def N_k(N_start):
    return N_start + delta

def k(N_start):
    k = a(N_start)*H(N_start)*exp(N_k(N_start) - N_start)


file = open('/work/Teerthal/Gauge_Evolution/Integral/stack_1/stack_1.npy', 'rb')
stack = np.load(file)

#Loop parameters

for f_index in [0]:
    f = 0.1

    for b_index in [0, 1, 2]:
        b = [0, .25 / f, .5 / f]
        b = b[b_index]

        Integrate = []
        for N_start_index in np.arange(0, int(24*50), 1):
            N_start = np.linspace(45., 62., 24*50)
            N_start = N_start[N_start_index]

            temp = np.array([int(item) for item in stack[N_start_index, f_index, b_index, 0, 0]])

            dump = []
            N_dump = []
            for N in np.linspace(45., 68., 100):

                N_dump.append(N)

                if int(N) in temp:
                    index = int(N / 68. * 50000)

                    N_add = stack[N_start_index, f_index, b_index, 0, index]
                    A_1 = stack[N_start_index, f_index, b_index, 0, 1, index]
                    A_2 = stack[N_start_index, f_index, b_index, 1, 1, index]

                    dump.append(np.array(N, A_1, A_2))

                else:

                    index = 100

                    A_1 = stack[N_start_index, f_index, b_index, 0, 1, index]
                    A_2 = stack[N_start_index, f_index, b_index, 1, 1, index]

                    dump.append(np.array([N, A_1, A_2]))


            print(np.shape(dump))
            print(np.shape(N_dump))
            exit()
            Integrand = 1/(4*pi)**2*exp(4*(N_k(N_start)-N))*H(N)**4*(dump[1]**2-1+dump[2]**2)/k(N_start)
            Integrate.append(np.array([k(N_start), Integrand]))

        rho_B = trapz(Integrate[1], x=Integrate[0])
