import matplotlib.pyplot as plt
import numpy as np

exp = np.exp;sqrt=np.sqrt;log =np.log; sin = np.sin; cos = np.cos

p=2

x_initial = 1e3
x_final = 1e-2

N_initial = 0
N_final = 70

b = 100
f = 0.01
alpha = 2*f*10

delta_N = 50
xi_pivot = 2; print(xi_pivot)

N_pivot = N_final - delta_N

steps =1e6
N = np.linspace(N_initial, N_final, steps)
x = np.linspace(x_initial,x_final,steps)

phi_0_pivot = sqrt(p ** 2 / 2 + 2 * p * (N_final - N_pivot))

eps_0_pivot = p / (p + 4 * (N_final - N_pivot))  # value of eps_o at pivot scale
dphi_0 = sqrt(2 * eps_0_pivot)  # derivative of phi_0 w.r.t N

phi_0 = sqrt(p ** 2 / 2 + 2 * p * (N_final - N_pivot) + 2 * p / (1 - eps_0_pivot) * log(x/xi_pivot))

# Oscillation
Amp_1 = ((12 * b * (phi_0_pivot ** (2 - p)) * (f ** 2)) /
         (36 * (phi_0_pivot ** 2) * (f ** 2) + (3 * (p - 2) * (f ** 2) - 2 * p) ** 2))
Amp_sin = 3 * (p - 2) * f ** 2 / (2 * p) - 1
Amp_cos = 3 * phi_0_pivot * f / p
phi_1 = Amp_1 * (Amp_sin * sin(phi_0 / f) + Amp_cos * cos(phi_0 / f))
phi_1_prime = Amp_1 / f * (Amp_sin * cos(phi_0 / f) - Amp_cos * sin(phi_0 / f))

mod = phi_1_prime  # referes to the scalar field oscillatory contribution

phi = phi_1+phi_0
dphi = dphi_0 * (1 + mod)  # Total dphi/dN
eps = 1 / 2 * dphi ** 2  # Total epsilon

eps_0 = 1/2*dphi_0**2
eps_1 = 1/2*(dphi_0*phi_1_prime)**2

xi = alpha / (2 * f) * dphi

plt.subplot(311)
plt.semilogx(x, eps)
plt.subplot(312)
plt.semilogx(x, eps_1/eps_0)
plt.subplot(313)
plt.semilogx(x, xi)

plt.show()

