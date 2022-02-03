import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint

pi = np.pi
sqrt = np.sqrt
cos = np.cos
sin = np.sin
exp = np.exp
log = np.log
arctan = np.arctan

for p in [2]:

    # Parameters: x:(Phi), x0:(Pivot scale Phi), y:(first order scalar field term)
    lamda = 8.e-4
    f = 0.05
    N_p = 1
    V_p = (1e-3) ** 4
    phi_p = sqrt(2 * p * N_p + p ** 2 / 2)
    mu = V_p ** (1 / (4 - p)) / ((phi_p) ** (p / (4 - p)))
    b = lamda ** 4 / (mu ** (4 - p) * f)

    #Numerically optimum b and f parameters
    b = 10.
    f = 0.01
    phi00 = sqrt(282)
    phi0f = p / sqrt(2)
    steps = 1e5
    print('b=',b)
    print('bf^2=',b*f**2)
    print('mu =', mu)
    print('Mass term:', mu**(4-p)*p)

for Ni in [0]:
    Nk = Ni + 10
    # Additional parameters for solving A
    for h in [-1, 1]:
        k = 1
        alpha = 4 * 0.04  # Keeping the derivative of coupling constant term as 1
        f_prime = - 8*alpha / (4 * f)

        N0 = 0
        Nf = -(phi0f ** 2 - phi00 ** 2) / (2 * p)  # and the corresponding initial phi value
        eps_0 = (-(phi00-phi0f)/(N0-Nf))**2
        print('eps_0=',eps_0)

        def system(N, u):
            eps = u[1] ** 2
            V = mu ** (4 - p) * u[0] ** p
            H = sqrt(V / (3 - eps))
            N = N - Nk
            return [u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p),
                    u[3],
                    -(1 - eps_0) * u[3] - (exp(-2 * N * (1 - eps_0)) + exp(-N * (1 - eps_0))*(1-eps_0) * h * f_prime * u[1]) * u[
                        2]]


        r = ode(system).set_integrator('vode', method='bdf', order=5, atol=1e-6, rtol=1e-6, max_step=1e8)

        # e-folding slice

        print(phi00)
        N_i = np.linspace(Ni, Nf, steps)

        # Anlaytic solution
        phi_0 = phi0f + (N_i - Nf) * (phi0f - phi00) / (Nf - N0)
        phi_0_prime = -sqrt(eps_0)

        Amp = ((6 * b * (phi_p ** (2 - p)) * (f ** 2)) / (
            p * sqrt(36 * (phi_p ** 2) * (f ** 2) + (3 * (p - 2) * (f ** 2) - 2 * p) ** 2)))

        # Inhomogeneous
        theta = arctan(-(3 * (p - 2) * f ** 2 - 2 * p) / (6 * phi_p * p))
        Amp_1 = ((12 * b * (phi_p ** (2 - p)) * (f ** 2)) /
                 (36 * (phi_p ** 2) * (f ** 2) + (3 * (p - 2) * (f ** 2) - 2 * p) ** 2))
        Amp_sin = 3 * (p - 2) * f ** 2 / (2 * p) - 1
        Amp_cos = 3 * phi_p * f / p
        phi_1 = Amp_1 * (Amp_sin * sin(phi_0 / f) + Amp_cos * cos(phi_0 / f))
        phi_1_prime = Amp_1 / f * (Amp_sin * cos(phi_0 / f) - Amp_cos * sin(phi_0 / f))

        phi = phi_0 + phi_1
        print(Amp)
        phi_prime = phi_0_prime+phi_1_prime
        # Setting Initial conditions at the end of inflation

        Ni_index = int(Ni / Nf * steps)
        Ni_index = 0
        phi_i = phi[Ni_index]
        phi_prime_i = phi_prime[Ni_index]
        print('Phi_initial', 'Phi_prime_initial')
        print(phi_i, phi_prime_i)

        # Vector potential initial conditions

        mag = exp((-Ni+Nk)*(1-eps_0))/(1-eps_0)
        A0 = sqrt(sin(mag) ** 2)
        A0 = 1
        A00 = - A0

        print('Vector initial values:', A0, A00)
        # Declaring initial conditions
        init = [phi_i, phi_prime_i, A0, A00]
        r.set_initial_value(init, Ni)

        # Solving loop
        u = [];
        t = []
        dt = (Nf - N0) / steps
        print(dt)
        while r.successful() and r.t <= Nf-15 :
            r.integrate(Nf, step=True)
            u.append(r.y);
            t.append(r.t)

        # Compiling solution in a list and removing the additional entries from the arrays
        phi = [item[0] for item in u]
        phi_prime = [item[1] for item in u]

        A = [item[2] for item in u]
        A_prime = [item[3] for item in u]

        print('t', 'Phi', 'Phi_prime', 'A', 'A_prime')
        print(len(t), len(phi), len(phi_prime), len(A), len(A_prime))

        eps = np.array(phi_prime) ** 2
        x = exp((-np.array(t)+Nk)*(1-eps_0))/(1-eps_0)
        Factor = np.abs(f_prime*sqrt(eps)*-exp(-(np.array(t)-Nk))/x)

        plt.subplot(411)
        plt.plot(t, phi, label=h)
        plt.title(['Ni:%d' % Ni, 'Nk:%d' % Nk])
        plt.legend()
        plt.subplot(412)
        plt.plot(t, np.array(phi_prime) ** 2)
        plt.subplot(413)
        plt.plot(t, np.abs(A))

        plt.subplot(414)
        plt.plot(x, Factor)

    plt.show()