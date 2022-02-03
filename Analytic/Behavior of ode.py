import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
start = time.time()
plt.rcParams['axes.labelsize'] = 20

p=2

pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

x_initial = 1e2
x_final = 1e-6

N_initial = 0
N_final = 70

b=100
f = 0.01
alpha = 2*f*10

delta_N_steps = 1

N_horizon = 50
Asymptotes = []
Analytic_asymptote = []
for delta_N in np.linspace(20,5,delta_N_steps, dtype=complex):

    Asymptotes_h=[]
    Analytic_asymptote_h = []

    for h in [-1,1]:

        xi_pivot = x_final    #####Have to check#####

        N_pivot = N_final - delta_N

        phi_0_pivot = sqrt(p ** 2 / 2 + 2 * p * (N_final - N_pivot))

        eps_0_pivot = p / (p + 4 * (N_final - N_pivot))  # value of eps_o at pivot scale
        dphi_0 = sqrt(2 * eps_0_pivot)  # derivative of phi_0 w.r.t N

        xi_slow_roll = alpha / (2 * f) *dphi_0
        Analytic_asymptote_h.append(exp(pi * xi_slow_roll) / sqrt(2 * pi * xi_slow_roll))
        a1 = []
        b1 = []
        def ODE(x, u):
            phi_0 = sqrt(p ** 2 / 2 + 2 * p * (N_final - N_pivot) + 2 * p / (1 - eps_0_pivot) * log(x / xi_pivot))
            a1.append(x)
            # Oscillation
            Amp_1 = ((12 * b * (phi_0_pivot ** (2 - p)) * (f ** 2)) /
                     (36 * (phi_0_pivot ** 2) * (f ** 2) + (3 * (p - 2) * (f ** 2) - 2 * p) ** 2))
            Amp_sin = 3 * (p - 2) * f ** 2 / (2 * p) - 1
            Amp_cos = 3 * phi_0_pivot * f / p
            phi_1 = Amp_1 * (Amp_sin * sin(phi_0 / f) + Amp_cos * cos(phi_0 / f))
            phi_1_prime = Amp_1 / f * (Amp_sin * cos(phi_0 / f) - Amp_cos * sin(phi_0 / f))

            mod = phi_1_prime  # referes to the scalar field oscillatory contribution

            phi = phi_1 + phi_0
            b1.append(phi_0)
            dphi = dphi_0 * (1 + mod)  # Total dphi/dN
            eps = 1 / 2 * dphi ** 2  # Total epsilon

            xi = alpha / (2 * f) * dphi

            return np.array([u[1], -(1 - 2 * h * xi / x)*u[0]])

        r = ode(ODE).set_integrator('zvode', method = 'bdf', order = 5, atol =1e-6, rtol = 1e-6, nsteps = 1e6)

        A_initial = exp(1j * x_initial)
        dA_dx_initial = 1j * exp(1j * x_initial)
        init = np.array([A_initial, dA_dx_initial], dtype=complex)
        r.set_initial_value(init, x_initial)

        u = []
        t = []
        while r.successful() and x_final <= r.t <= x_initial:
            r.integrate(x_final, step=True)
            u.append(r.y)
            t.append(r.t)

        A = np.array([item[0] for item in u],dtype=complex)
        dA_dx_initial = np.array([item[1] for item in u],dtype=complex)
        t = np.array(t)
        A_absolute = np.array([absolute(item[0]) for item in u])
        print('Length', 't', 'A')
        print(len(t), len(A))

        plt.semilogx(t,A_absolute, label = h)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\sqrt{2k}|\mathcal{A}_h$|')
        mean = np.mean(absolute(A)[int(len(A)*99/100):len(A)]);print(mean)
        std = np.std(absolute(A)[int(len(A)*99/100):len(A)]); print(std)

        Asymptotes_h.append([mean, std])
    Asymptotes.append([Asymptotes_h])
    Analytic_asymptote.append([Analytic_asymptote_h])
plt.legend(prop={'size': 10})
plt.show()
print('Shape of Asymptote stack:',np.shape(Asymptotes))
print('Shape of Analytic Asymptotic stack', np.shape(Analytic_asymptote))
end = time.time()
print('Elapsed time:', end-start)

Asymptotes = np.array(Asymptotes)
Analytic_asymptote = np.array(Analytic_asymptote)

delta_N = np.linspace(1,5,delta_N_steps)

plt.semilogy(delta_N, Asymptotes[:,0,1,0], label = '+Num')
plt.semilogy(delta_N, Asymptotes[:,0,0,0], label = '-Num')
plt.xlabel(r'$\mathcal{N}_f - \mathcal{N}_*$')
plt.ylabel(r'$|\mathcal{A}_h|$')
plt.title('b:%d' %b)
plt.legend()
plt.show()


# Analytic expression for the Asymptotes (for constant xi)
delta_N = np.linspace(1, 5,delta_N_steps)
x = np.linspace(x_initial,x_final,delta_N_steps)
xi_pivot = 1e-4    #####Have to check#####

N_pivot = N_final - delta_N

phi_0_pivot = sqrt(p ** 2 / 2 + 2 * p * (N_final - N_pivot))

eps_0_pivot = p / (p + 4 * (N_final - N_pivot))  # value of eps_o at pivot scale
dphi_0 = sqrt(2 * eps_0_pivot)  # derivative of phi_0 w.r.t N
phi_0 = sqrt(p ** 2 / 2 + 2 * p * (N_final - N_pivot) + 2 * p / (1 - eps_0_pivot) * log(x / xi_pivot))

# Oscillation
Amp_1 = ((12 * b * (phi_0_pivot ** (2 - p)) * (f ** 2)) /
                     (36 * (phi_0_pivot ** 2) * (f ** 2) + (3 * (p - 2) * (f ** 2) - 2 * p) ** 2))
Amp_sin = 3 * (p - 2) * f ** 2 / (2 * p) - 1
Amp_cos = 3 * phi_0_pivot * f / p
phi_1 = Amp_1 * (Amp_sin * sin(phi_0 / f) + Amp_cos * cos(phi_0 / f))
phi_1_prime = Amp_1 / f * (Amp_sin * cos(phi_0 / f) - Amp_cos * sin(phi_0 / f))

mod = phi_1_prime  # referes to the scalar field oscillatory contribution

phi = phi_1 + phi_0
dphi = dphi_0 * (1 + mod)  # Total dphi/dN
eps = 1 / 2 * dphi ** 2  # Total epsilon

xi = alpha / (2 * f) * dphi
xi_slow_roll = alpha / (2 * f) * dphi_0

Analytic_asymptote_1 = exp(pi * xi) / sqrt(2 * pi * xi)

plt.subplot(311)
plt.semilogy(delta_N, Analytic_asymptote_1, label = 'Ana')
plt.ylabel(r'$|\mathcal{A}_h|$')
plt.legend()
plt.title(['Analytic expression outside ODE', 'b:%d' %b])

plt.subplot(312)
plt.semilogy(delta_N, Analytic_asymptote[:,0,0])
plt.ylabel(r'$|\mathcal{A}_h|$')
plt.title(['Analytic expression inside ODE', 'b:%d' %b])

plt.subplot(313)
plt.plot(delta_N, xi, label = 'Full')
plt.plot(delta_N, xi_slow_roll, label = 'Slow Roll')
plt.xlabel(r'$N_f - N_pivot$')
plt.ylabel(r'$\xi$')
plt.legend()

plt.show()