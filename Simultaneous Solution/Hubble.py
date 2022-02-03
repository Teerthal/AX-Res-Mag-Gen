import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

exp = np.exp; log = np.log; sin = np.sin; sqrt = np.sqrt; abs = np.abs

p = 2

N_initial = -70
N_final = 0

a_final = 1

f = 0.05
alpha = 8*f
b = 0

phi_initial = sqrt(282)
phi_final = p / sqrt(2)
phi_prime_initial = -0.1

H_final = 1

f_prime = alpha/f


def ODE(N, u):
    eps = u[1] ** 2
    return [[u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p)]]


r = ode(ODE).set_integrator('vode', method='bdf', order=5, atol=1e-6, rtol=1e-6,
                               with_jacobian=False)

init = np.array([phi_initial, phi_prime_initial])
r.set_initial_value(init, N_initial)

u=[];t=[]
while r.successful() and r.t <= N_final:
    r.integrate(N_final, step=True)
    u.append(r.y)
    t.append(r.t)

phi = np.array([item[0] for item in u])
phi_prime = np.array([item[1] for item in u])
N = np.array(t)

epsilon = 1/2*phi_prime**2
epsilon_initial = 1/2*phi_prime[50]**2
H = H_final*sqrt(2/(3-epsilon)*(phi/phi_final)**2)
H_initial = H_final*sqrt(2/(3-epsilon_initial)*(phi_initial/phi_final)**2)

a = a_final*exp(N)
a_initial = a_final*exp(N_initial)

xi = alpha/f*phi_prime
xi_initial = alpha/f*phi_prime[50]
xi_final = alpha*sqrt(2)/f

k_ins = abs(xi)*a*H
k_ins_min = xi_initial*a_initial*H_initial;print('k_min:%str'%k_ins_min)
k_ins_max = xi_final*a_final*H_final;print('k_max:%str'%k_ins_max)
k = np.linspace(k_ins_min,k_ins_max, len(N))

plt.subplot(311)
plt.plot(N, phi)
plt.subplot(312)
plt.plot(N,phi_prime)
plt.subplot(313)
plt.plot(N, H)

plt.show()

plt.subplot(211)
plt.plot(N,(k/(a*H))**2)
plt.subplot(212)
plt.semilogy(N,(k_ins/(a*H))**2)
plt.show()

h = -1

def Gauge(Ni,u):
    eps = u[1] ** 2
    a = a_final * exp(N)
    xi = alpha / f * u[1]
    H = H_final * sqrt(2 / (3 - u[1] ** 2) * (u[0] / phi_final) ** 2)
    k_ins = abs(xi) * a * H
    k = k_ins*0

    return [[u[1], (eps - 3) * u[1] + (p * u[0] ** (p - 1) - b * sin(u[0] / f)) * (eps - 3) / (u[0] ** p)],
            u[3], -(1-eps)*u[3]-((k/(a*H))**2 - (h*alpha*k/(f*a*H))*u[1])*u[2]]

ode = ode(Gauge).set_integrator('zvode', method='bdf', order=5, atol=1e-6, rtol=1e-6,
                            with_jacobian=False)

init = np.array([phi_initial,phi_prime_initial, exp(1j), -1j*exp(1j)])
ode.set_initial_value(np.array([phi_initial,phi_prime_initial, exp(1j), -1j*exp(1j)]),N_initial)

u1 = []
t1 = []
while ode.successful() and ode.t <= N_final:
    ode.integrate(N_final, step=True)
    u1.append(ode.y)
    t1.append(ode.t)

phi = np.array([item[0]] for item in u1)
phi_prime = np.array([item[1] for item in u1])
A = np.array([item[2] for item in u1])
N = np.array(t1)

plt.plot(N, np.absolute(A))
plt.show()