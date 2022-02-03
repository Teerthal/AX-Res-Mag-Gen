import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 20
pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

file = open('/home/teerthal/Repository/Gauge_Evolution/Separate/0207/separate.npy', 'rb')
stack = np.load(file)
print(np.shape(stack))

p = 2

f = 0.1

b = 0

N_final = 70.
N_initial = 0.

delta = 8

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1


def a(N):
    return exp(N)


def phi(N):
    return sqrt(2 * p * (N_final - N) + phi_final ** 2)


def phi_prime(N):
    return -p / phi(N)

def epsilon(N):
    return (phi_prime(N)) ** 2 / 2.


def H(N):
    return abs(H_initial * sqrt((3. - epsilon(N)) / (3. - epsilon(N_initial)) * (phi(N) / phi_initial) ** 2))


def k(N_k):
    k = a(N_final) * H(N_final) * exp(N_k - N_final)
    return  k


k_list = np.array([*map(k, stack[:,0,0,0,0])])

b=0
for f_index in [3]:
    f = np.array([0.1,0.05,0.025,0.01])
    f = f[f_index]
    for b_index in [0,1,2]:
        b = np.array([0,0.2,.3,.4,.5])
        b = b[b_index]
        plt.loglog(k_list/(a(N_final)*H(N_final)), stack[:, f_index, b_index, 0, 1], label= 'Mod:%s'%b)

plt.title(r'$\alpha/f:20$ f:%s'%f)
plt.xlabel(r'${k}/{a_fH_f}$')
plt.ylabel(r'$\sqrt{2k}|\mathcal{A}_\pm|$')
plt.legend()
plt.show()