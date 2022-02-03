import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

p = 2
f_max = 1
alpha = 20*f_max

N_final = 70.
N_initial = 0.

def N_stop(b):
    return N_final + b*f_max*10

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 1

asyms_b = []
for b in [0.]:

    def a(N):
        return exp( N)

    def phi(N):
        return sqrt(2*p*(N_final-N_initial)+phi_final**2)
    print('phi initial:',phi(N_initial))

    def f(N):
        return f_max
    print()
    def phi_prime(N):
        return -p / phi(N)


    def epsilon(N):
        return (phi_prime(N)) ** 2 / 2.


    def H(N):
        return H_initial * sqrt((3. - epsilon(N)) / 2. * (phi(N) / phi_initial) ** 2)


    def k_ins(N):
        return alpha / f(N) * a(N) * abs(phi_prime(N)) * H(N) / a(N_final)

    def Scalar_ODE(N, u):
        eps = 1 / 2 * u[1] ** 2
        return np.array(
            [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f_max)) * (eps - 3) / (u[0] ** p)])

    solver = ode(Scalar_ODE).set_integrator('vode')
    Scalar_init = np.array([phi(N_initial),phi_prime(N_initial)])
    solver.set_initial_value(Scalar_init,N_initial)

    #Scalar ODE solving steps

    steps = 5e3
    dt = (N_stop(b)-N_initial)/steps

    temp_1=[];temp_2=[]
    while solver.successful() and solver.t<=N_stop(b):
        solver.integrate(solver.t + dt)
        temp_1.append(solver.t)
        temp_2.append(solver.y)

    plt.subplot(211)
    plt.plot(temp_1,np.array([item[0] for item in temp_2]))
    plt.subplot(212)
    plt.plot(temp_1,np.array([item[1] for item in temp_2]))
    plt.show()

    def Phi(N):
        phi_solved = np.array([item[0] for item in temp_2])
        index_N = int(N/N_final*steps)

        return phi_solved[index_N]

    def Phi_Prime(N):
        phi_prime_solved = np.array([item [1] for item in temp_2])
        index_N = int(N/N_final*steps)

        return phi_prime_solved[index_N]


    asyms_delta = []
    for delta in np.linspace(6,8,1):
        asyms_Nstart = []
        for N_start in [60.]:

            N_k = N_start + delta

            k = exp(N_k - N_start)

            print('k', 'N_k')
            print([k, N_k])

            asyms_h = []

            for h in [-1,1]:
                def ODE(N, u, arg):

                    return np.array(
                        [u[1],
                         -(1 - epsilon(0)) * u[1] - (
                             exp(-2 * (N - N_k)) - exp(-(N - N_k)) * h * alpha / f_max * arg(N)) * u[0]])

                r = ode(ODE).set_integrator('zvode').set_f_params(Phi_Prime)

                # Initial Condition
                t_initial = -1
                A_initial = exp(-1j * k * t_initial)
                A_prime_initial = -1j * A_initial * k
                print(A_prime_initial)
                init = np.array([A_initial, A_prime_initial]);
                print(phi_prime(N_start));
                print(phi(N_start))
                r.set_initial_value(init, N_start)

                u = []
                t = []

                Gauge_steps = 1e4
                dt = (N_final-2-N_start)/Gauge_steps
                while r.successful() and r.t <= N_final-2:
                    r.integrate(r.t+dt)
                    u.append(r.y)
                    t.append(r.t)

                A = absolute(np.array([item[0] for item in u]))
                A_prime = np.array([item[1] for item in u])
                N = np.array(t)
                print(np.shape(N))
                plt.plot(N, A, label = 'h:%s'%h)
                plt.xlabel(r'$\mathcal{N}$')
                plt.ylabel(r'$\sqrt{2k}\mathcal{A}_{\pm}$')
                end = time.time()
                print('Elapsed time:%s', end - start)
                asyms_h.append(np.array([N_k, A[-1]]))
        asyms_Nstart.append(np.array(asyms_h))
    asyms_b.append(np.array(asyms_Nstart))
plt.legend()
plt.show()
exit()
asyms = np.array(asyms_b)
print(np.shape(asyms))

plt.semilogy(asyms[:,1,0,0],asyms[:,1,0,1], label = '0')
plt.semilogy(asyms[:,0,0,0],asyms[:,0,0,1],label = '1')
plt.show()