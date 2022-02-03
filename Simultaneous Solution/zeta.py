import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
from multiprocessing import Pool

start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;
sin = np.sin;
cos = np.cos;
exp = np.exp;
log = np.log;
abs = np.abs;
absolute = np.absolute;
sqrt = np.sqrt


def system(N_start):
    p = 2

    asyms_f = []
    stack_f = []
    for f in [.1]:
        print('f:%s' % f)
        alpha = 10 * f;
        print('alpha/f:%s' % (alpha / f))

        N_final = 70.
        N_initial = 0.

        Offset = 8;
        print('Offset:%s' % Offset)

        def N_stop(b, f):
            return N_final + b * f * Offset

        phi_initial = sqrt(282)
        phi_final = p / sqrt(2)

        H_initial = 1

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

        def k_ins(N):
            return alpha / f * a(N) * abs(phi_prime(N)) * H(N) / a(N_final)

        asyms_F = []
        stack_F = []
        for F in np.linspace(0.001,1, 10):

            Phi_Prime = 1/F
            delta = 8.

            print('delta:%f' % delta)

            N_k = N_start + delta

            k = exp(N_k - N_start)

            asyms_h = []
            stack_h = []
            for h in [-1]:

                def ODE(N, u, arg):

                    return np.array(
                        [u[1],
                         -(1 - epsilon(0)) * u[1] - (
                             exp(-2 * (N - N_k)) - exp(-(N - N_k)) * h * alpha / f * arg(N)) * u[0]])

                r = ode(ODE).set_integrator('zvode').set_f_params(Phi_Prime)

                # Initial Condition
                t_initial = -1
                A_initial = exp(-1j * k * t_initial)
                A_prime_initial = -1j * A_initial * k

                init = np.array([A_initial, A_prime_initial]);

                r.set_initial_value(init, N_start)

                u = []
                t = []
                while r.successful() and r.t <= N_final - 2:
                    r.integrate(N_final, step=True)
                    u.append(r.y)
                    t.append(r.t)

                A = absolute(np.array([item[0] for item in u]))
                A_prime = np.array([item[1] for item in u])
                N = np.array(t)

                file_temp = open(
                    '/work/Teerthal/Gauge_Evolution/EVO/EVO_1/EVO_%.6f_%d_%.3f_%.3f.npy' % (N_start, h, F, f), 'wb')
                np.save(file_temp, np.array([N, A, A_prime]))
                file_temp.close()

                asyms_h.append(np.array([N_k, A[-1]]))

            asyms_F.append(np.array(asyms_h))

        asyms_f.append(np.array(asyms_F))

    return np.array(asyms_f)


N_start = np.linspace(45., 60., 24)

pool = Pool(24)

if __name__ == '__main__':
    p = np.array(pool.map(system, N_start))
    pool.close()

print('Shape of p:', np.shape(p))

file = open('/work/Teerthal/Gauge_Evolution/EVO/EVO_1/asyms_1.npy', 'wb')
np.save(file, p)
file.close()

end = time.time()
print('Elapsed Time:%s' % (end - start))