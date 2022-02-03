import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt

p=2

t_i = -1
t_f = -exp(-60)

f = 0.01
b = 0
m = 1e-12

phi_initial = sqrt(282)
phi0f = p / sqrt(2)
phi_prime_initial = 0.1


def ODE(t, u):
    epsilon = 1/2*(p/u[0]**2)
    V = m*(u[0]**(4-p)+b*f*cos(u[0]/f))
    H_f = sqrt(m*phi0f**(4-p)/2)
    H = sqrt(V/3)
    a = -1/(1-epsilon)*(H_f/H)*(1/t)
    dV_dphi = m*((4-p)**u[0]**(3-p) - b*sin(u[0]/f))
    return np.array([u[1], -2*a*(H/H_f)*u[1]-a**2*dV_dphi/H**2])

r = ode(ODE).set_integrator('vode', method='bdf', order=5, atol=1e-6, rtol=1e-6, nsteps=1e6)

init = np.array([phi_initial, phi_prime_initial])
r.set_initial_value(init, t_i)

u = []
t = []

while r.successful() and t_i <= r.t <= t_f:
    r.integrate(t_f, step=True)
    u.append(r.y)
    t.append(r.t)

phi = np.array([item[0] for item in u])
phi_prime = np.array([item[1] for item in u])
t = np.array(t)

print(np.shape(t))

plt.subplot(211)
plt.semilogx(t-t_i, phi)
plt.subplot(212)
plt.semilogx(t-t_i, phi_prime)
plt.show()