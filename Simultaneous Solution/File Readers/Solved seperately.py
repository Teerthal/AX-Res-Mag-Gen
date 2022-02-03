import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import time as time
start = time.time()
plt.rcParams['axes.labelsize'] = 20

pi = np.pi;sin = np.sin; cos = np.cos; exp = np.exp; log = np.log; abs = np.abs; absolute = np.absolute; sqrt = np.sqrt


p = 2
f = 0.01
b = 0.8/f
alpha = 7*f

N_initial = 0.
N_final = 70.

phi_initial = sqrt(282.)
phi_final = p / sqrt(2.)

H_initial = 1.

steps = 1e6

def a(N):
    return exp(N_initial+N)

def phi(N):
    return phi_final + (N - N_final) * (phi_final - phi_initial) / (N_final - N_initial)

def phi_prime(N):
    return -p/phi(N)

def epsilon(N):
    return (phi_prime(N))**2 / 2.

def H(N):
    return H_initial*sqrt((3.-epsilon(N)) / 2. * (phi(N) / phi_initial) ** 2)

def k_ins(N):
    return alpha/f*a(N)*abs(phi_prime(N))*H(N)/a(N_final)

def ODE(N,u):
    eps = 0.5*u[1]**2
    return [u[1], (eps - 3.) * u[1] + (p * u[0] ** (p - 1.) - b * sin(u[0] / f)) * (eps - 3.) / (u[0] ** p)]

r = ode(ODE).set_integrator('vode')
IC = np.array([phi(N_initial),phi_prime(N_initial)])
r.set_initial_value(IC, N_initial)
dt = abs(N_final-N_initial)/steps;print(dt)
u=[];t=[];
while r.successful() and N_initial<=r.t<=N_final:
    r.integrate(r.t+dt)
    u.append(r.y)
    t.append(r.t)
u.pop();t.pop()
u1 = np.array([item[0] for item in u])
u2 = np.array([item[1] for item in u])
t = np.array(t)
print(np.shape(u1))

def i(N):
    return N / N_final * steps

def Phi(N):
    return u1[int(i(N))]

def Phi_prime(N):
    return u2[int(i(N))]

def Epsilon(N):
    return 1./2.*Phi_prime(N)**2

def Xi(N):
    return alpha/f*Phi_prime(N)

for h in [-1,1]:

    for k in [k_ins(N_final),100*k_ins(N_final)]:

        def A_ODE(N, z):
            return np.array([z[1],
                    -(1 - Epsilon(N)) * z[1] - ((k / (a(N) * H(N))) ** 2 -
                    (h * alpha * k / (f * a(N) * H(N))) * Phi_prime(N)) *
                    z[0]])

        A_sol = ode(A_ODE).set_integrator('vode')

        # Initial Condition
        t_initial = -1
        A_initial = 1 / sqrt(2 * k) * exp(-1j * k * t_initial)
        A_prime_initial = -1j * k * A_initial / (a(N_initial) * H_initial)
        init = np.array([A_initial,A_prime_initial],dtype=complex)

        A_sol.set_initial_value(init,N_initial)

        z=[];l=[]
        while A_sol.successful() and N_initial <= A_sol.t <= N_final-40:
            A_sol.integrate(A_sol.t+dt)
            z.append(A_sol.y)
            l.append(A_sol.t)

        A = np.array([absolute(item[0]) for item in z])
        N = np.array(l)

        print(np.shape(A))
        plt.plot(N, A)
plt.show()