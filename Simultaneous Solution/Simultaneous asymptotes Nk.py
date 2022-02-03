import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

p = 2
f = 0.1
alpha = 10*f

N_final = 70
N_stop = 70
phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

asyms = []
for N_initial in [0,65]:

    def a(N):
        return exp( N)


    def phi(N):
        return sqrt(2*p*(N_final-N_initial)+phi_final**2)
    print('phi initial:',phi(N_initial))
    def phi_prime(N):
        return -p / phi(N)


    def epsilon(N):
        return (phi_prime(N)) ** 2 / 2.


    def H(N):
        return H_initial * sqrt((3. - epsilon(N)) / 2. * (phi(N) / phi_initial) ** 2)


    def k_ins(N):
        return alpha / f * a(N) * abs(phi_prime(N)) * H(N) / a(N_final)

    asyms_delta = []
    for delta in np.linspace(6,8,1):
        N_k = N_initial + delta

        k = exp(N_k - N_initial)

        print('k', 'N_k')
        print([k, N_k])

        asyms_b = []
        for b in [0, .1/f, .2 / f, .5/f]:

            Offset = b*f*10
            asyms_h = []
            for h in [-1,1]:

                def ODE(N, u):

                    eps = 1 / 2 * u[1] ** 2
                    return np.array(
                        [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p),
                         u[3],
                         -(1 - epsilon(0)) * u[3] - (
                             exp(-2 * (N - N_k)) - exp(-(N - N_k)) * h * alpha / f * u[1]) * u[2]])


                r = ode(ODE).set_integrator('zvode', method='bdf', order=5, rtol=1e-8, atol=1e-8, nsteps=1e6)

                # Initial Condition
                t_initial = -1
                A_initial = exp(-1j * k * t_initial)
                A_prime_initial = -1j * A_initial * k
                print(A_prime_initial)
                init = np.array([phi(N_initial), phi_prime(N_initial), A_initial, A_prime_initial]);
                print(phi_prime(N_initial));
                print(phi(N_initial))
                r.set_initial_value(init, N_initial)

                u = []
                t = []
                while r.successful() and r.t <= N_stop+Offset:
                    r.integrate(N_stop+Offset, step=True)
                    u.append(r.y)
                    t.append(r.t)

                Phi = np.array([item[0] for item in u])
                Phi_prime = np.array([item[1] for item in u])
                A = absolute(np.array([item[2] for item in u]))
                A_prime = np.array([item[3] for item in u])
                N = np.array(t)
                plt.subplot(313)
                plt.plot(N, Phi)
                plt.subplot(311)
                plt.plot(N, abs(alpha/f*Phi_prime), label=b)
                plt.legend()
                plt.subplot(312)
                plt.plot(N, A)
                asyms_h.append(np.array([N_k, A[-1]]))
            asyms_b.append(np.array(asyms_h))
        asyms_delta.append(np.array(asyms_b))
    asyms.append(np.array(asyms_delta))
plt.show()
asyms = np.array(asyms)
print(np.shape(asyms))

plt.semilogy(asyms[:,1,0,0],asyms[:,1,0,1], label = '0')
plt.semilogy(asyms[:,0,0,0],asyms[:,0,0,1],label = '1')
plt.show()