import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

pi = np.pi
sqrt = np.sqrt
cos = np.cos
sin = np.sin
exp = np.exp
log = np.log
arctan = np.arctan

bigstack = []
for f in np.linspace(0.5,0.001,7):
    stack = []
    runs = 7

    for lamda in np.linspace(4e-3, 2e-3, runs):

        # Parameters: x:(Phi), x0:(Pivot scale Phi), y:(first order scalar field term)
        p = 2
        N_p = 10
        V_p = (8.6e-3) ** 4
        phi_p = sqrt(2 * p * N_p + p ** 2 / 2)
        mu = V_p ** (1 / (4 - p)) / ((phi_p) ** (p / (4 - p)))
        b = lamda ** 4 / (mu ** (4 - p) * f)
        phi00 = sqrt(282)
        phi0f = p / sqrt(2)
        steps = 10e5
        print(b, mu)


        # Solver
        def system(N, u):
            eps = u[1] ** 2
            return [[u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p)]]


        r = ode(system).set_integrator('lsoda', method='bdf', nsteps=steps, rtol=10e-10, atol=10e-10,
                                       with_jacobian=False)

        for phi00 in np.linspace(sqrt(282), 80, 1):
            for phi_prime_0 in np.linspace(0, 1, 1):
                # Time slice
                N0 = 0
                Nf = -(phi0f ** 2 - phi00 ** 2) / (2 * p)  # Setting the initial phi value as the free parameter
                print(phi00)
                N = np.linspace(N0, Nf, steps)

                # Anlaytic solution
                phi_0 = np.linspace(phi00, phi0f, steps)
                Amp = ((6 * b * (phi_p ** (2 - p)) * (f ** 2)) / (
                    p * sqrt(36 * (phi_p ** 2) * (f ** 2) + (3 * (p - 2) * (f ** 2) - 2 * p) ** 2)))
                print(Amp)

                # Initial condition
                init_beta = np.array([phi00, phi_prime_0])
                r.set_initial_value(init_beta, N0)
                print(init_beta)

                u = [];
                t = []
                dt = (Nf - N0) / steps
                print(dt)
                while r.successful() and N0 <= r.t <= Nf:
                    r.integrate(r.t + dt)
                    u.append(r.y);
                    t.append(r.t)

                # Compiling solution in a list and removing the additional entries from the arrays
                sol_0 = [item[0] for item in u]
                sol_1 = [item[1] + 100 * lamda for item in u]
                sol_0.pop()
                sol_1.pop()
                t.pop()
                print(len(t), len(sol_0), len(sol_1))
                stack.append([t, sol_0, sol_1])
    bigstack.append(stack)

    stack = np.array(stack)
    print(np.shape(stack))

print(np.shape(bigstack))
plt.show()
import pickle, pprint

my_data = np.array(bigstack)
path = '/Users/clab/Desktop/Repository/uni/C/Magnetogenesis/Full scalar solution parameter variation/Data Dump/lamda(4,2,7)f(0.5,0.001,7)stdIC.pkl'
output = open(path, 'wb')
pickle.dump(my_data, output)
output.close()

