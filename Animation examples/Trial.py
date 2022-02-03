import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

pi = np.pi;
sin = np.sin;
cos = np.cos;
exp = np.exp;
log = np.log;
abs = np.abs;
absolute = np.absolute;
sqrt = np.sqrt

p = 2

F = 0.0025
alpha_0 = 10.

N_final = 70.
N_initial = 0.

Offset = 5;

phi_initial = sqrt(282)
phi_final = p / sqrt(2)

H_initial = 0.5

b_i = 0.01
b_f = 0.2
b_list_size = 3

steps = int(5.e5)
Gauge_steps = int(3.e5)

N_start = 60.

steps = Gauge_steps*N_final/(N_final-N_start)

cpu = 24

N_start_intervals = int(500)
delta_intervals = N_start_intervals

alpha_i = 15.
alpha_f = 20.
alpha_list_size = 1

phi_cmb = 15

# PARAMETERS FOR LOOPS
####################################

# f_list = np.array([0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.0001])
f_list = np.array([0.0001])
####################################

index_b = np.arange(0, b_list_size, 1)
index_f = np.arange(0, len(f_list), 1)
index_alpha = np.arange(0, alpha_list_size, 1)


######################

#Setting epsilon limits for killing slow roll and non slow roll cases
eps_limit_ns = 1.5
eps_limit_slow = 1.
eps_overhead = 1.
sparce_res = 10000                #The number with which the scalar solution is split into for computing the mean
                                #\epsilon
##################

# Parameters for computation start point and wavemodes
####################################

delta = np.geomspace(0.001, 5., delta_intervals)

####################################

#End time for gauge solver
N_gauge_off = 0.

xi_samp_inter = 0.1     #For sampling xi around itme of interest to take 'non-oscillating' del phi for computing
                            #.....corresponding k

Data_set = '1'

Master_path = '/home/teerthal/Repository/Gauge_Evolution/MK_1/%s' %Data_set


def load(name, directory):
    path = '%s/%s/%s.npy' % (Master_path, directory, name)
    file = open(path, 'rb')
    stack = np.load(file)
    file.close()
    return stack


def b_list(item_f):
    b = np.linspace(b_i, b_f, b_list_size) / item_f
    return b


def alpha_list(item_f):
    alpha = np.linspace(alpha_i, alpha_f, alpha_list_size) * item_f

    return alpha


Buffer = load('Phi', 'Phi')


[j,i] = [0, 2]

N_phi_solved = Buffer[0, j, i, 2]
phi_solved = Buffer[0, j, i, 0]
phi_prime_solved = Buffer[0, j, i, 1]

size = len(N_phi_solved)

data = np.array([N_phi_solved, phi_prime_solved ** 2. / 2.])

fig, ax = plt.subplots()

line, = ax.plot([], [],)

ax.set_xlim(0,72)
ax.set_ylim(-0.1, 2.)

xdat, ydat = [], []

def data_gen(i):
    return data[:,0:i]

def animate(i):

    line.set_data(data[..., 0:i])

    return line,

spacing = np.arange(0, size, int(size/100))

line_ani = animation.FuncAnimation(fig, animate, spacing,
           interval=1, blit=True, repeat=False)

plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

line_ani.save('/home/teerthal/Repository/lines.mp4', writer=writer)