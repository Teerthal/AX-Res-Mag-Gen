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

for p in [2]:
    N_p = 1
    V_p = (2e-3) ** 4
    phi_p = sqrt(2 * p * N_p + p ** 2 / 2)
    mu = V_p ** (1 / (4 - p)) / ((phi_p) ** (p / (4 - p)))

    # Parameters: x:(Phi), x0:(Pivot scale Phi), y:(first order scalar field term)
    bigstack = []
    biglist = []
    for b in np.linspace(0.1,100,7):
        stack = []
        list = []
        for f in [sqrt(1/b)]:
            phi00 = sqrt(282)
            phi0f = p / sqrt(2)
            steps = 10e5
            print('b=', b)
            print('bf^2=', b * f ** 2)

            for phi00 in [sqrt(282)]:
                # Solver
                def system(N, u):
                    eps = u[1] ** 2
                    return [
                        [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p)]]


                r = ode(system).set_integrator('lsoda', method='bdf', nsteps=steps, rtol=10e-10, atol=10e-10,
                                               with_jacobian=False)

                runs = 1
                for phi_prime_0 in np.linspace(0, 1, runs):
                    # Time slice
                    N0 = 0
                    Nf = -(phi0f ** 2 - phi00 ** 2) / (2 * p)
                    # and the corresponding initial phi value
                    print(phi00)
                    N = np.linspace(Nf, N0, steps)
                    T = exp(-N)
                    T0 = exp(-N0)
                    Tf = exp(-Nf)

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
                    sol_1 = [item[1] for item in u]
                    sol_0.pop()
                    sol_1.pop()
                    t.pop()
                    print(len(sol_0), len(sol_1), len(N))

                    stack.append([t, sol_0, sol_1])
                    list.append(f)


        print('stack shape',np.shape(stack))

        bigstack.append(stack)
        biglist.append(list)
print('bigstack',np.shape(bigstack))
print('list of f:', biglist)

#DataDump
import pickle

my_data = np.array(bigstack)
path = '/Users/clab/Desktop/Repository/uni/C/Magnetogenesis/Full scalar solution parameter variation/Data Dump/f(s.t 0.1^2<bf^2<0.005^2,7intervals)b(10,100,7)stdIC.pkl'
output = open(path, 'wb')
pickle.dump(my_data, output)
output.close()
