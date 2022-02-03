import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

plt.rcParams['axes.labelsize'] = 20
pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

file = open('/home/teerthal/Repository/Gauge_Evolution/Integral/stack_1/dump.npy', 'rb')
stack = np.load(file)
print(np.shape(stack))

p = 2

N_final = 70.
N_initial = 0.

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

delta = 6

def a(N):
    return exp(N)/exp(70)


def phi(N):
    return sqrt(2 * p * (N_final - N_initial) + phi_final ** 2)


def phi_prime(N):
    return -p / phi(N)


def epsilon(N):
    return (phi_prime(N)) ** 2 / 2.

def H(N):
    return H_initial * sqrt((3. - epsilon(N)) / 2. * (phi(N) / phi_initial) ** 2)


for b_index in [0,1,2]:
    Integrated = []

    temp_1 = []; temp_2=[]
    for N_index in np.arange(0, 100, 1):
        N = np.linspace(45., 68., 100)
        N = N[N_index]
        temp_1.append(N)
        Integrand = []
        k_list = []

        A1=[];A2=[]
        for N_start_index in np.arange(0, 24 * 50, 1):
            N_start = np.linspace(45., 62., 24 * 50)
            N_start = N_start[N_start_index]
            A_1 = stack[0, b_index, N_start_index, N_index, 0]
            A_2 = stack[0, b_index, N_start_index, N_index, 1]

            A1.append(stack[0, b_index, N_start_index, 99, 0])
            A2.append(stack[0, b_index, N_start_index, 99, 0])


            N_k = N_start + 6
            k = a(N) * H(N) * exp(N_k - N)

            Integrand.append(1 / (4 * pi) ** k**4 * (A_1 ** 2-1 + A_2 ** 2-1) / k)

            k_list.append(k)


        plt.plot(k_list, A1)
        plt.plot(k_list, A2)
        plt.show()

        Integrand = np.array([Integrand])[0]
        temp_2.append(Integrand[1000])
        k_list = np.array([k_list])
        Integral = trapz(Integrand, x=k_list)
        Integrated.append(np.array([N,Integral]))
    plt.semilogy(temp_1,temp_2)
    plt.show()
    Integrated = np.array([Integrated])[0]
    print(np.shape(Integrated))

    plt.plot(Integrated[:, 0], Integrated[:, 1], label = b_index)
plt.show()