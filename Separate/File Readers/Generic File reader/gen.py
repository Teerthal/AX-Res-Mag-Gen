from Parameters import *

def b_list(item_f):
    b = np.linspace(b_i, b_f, b_list_size) / item_f
    return b


def alpha_list(item_f):
    alpha = np.linspace(alpha_i, alpha_f, alpha_list_size) * item_f

    return alpha

def save(data, name, directory):
    path = '%s/%s/%s.npy' % (Master_path, directory, name)

    file = open(path, 'wb')
    np.save(file, data)
    file.close()
    return


def load(name, directory):
    path = '%s/%s/%s.npy' % (Master_path, directory, name)
    file = open(path, 'rb')
    stack = np.load(file)
    file.close()
    return stack

def a(N):
    return exp(N - N_final)


def phi(N):
    return sqrt(2 * p * (N_final - N) + phi_final ** 2)


def phi_prime(N):
    return -p / phi(N)


def H(N):
    H_constant = H_initial
    return H_constant

def k(delta):
    k = H(N_final) * exp(-delta)
    return k

def Asymp_alpha():
    stack_j = []
    for j in index_f:
        f_j = f_list[j]
        b = b_list(f_j)
        alpha = alpha_list(f_j)
        alpha_sr = alpha_list(f_list[0])

        stack_k = []
        for idx_k in np.arange(0, delta_intervals, 150):

            stack_i = []
            for i in index_b:
                b_i = b[i]

                stack_alpha = []

                for z in np.arange(0, len(alpha), 1):
                    alpha_z = alpha[z]
                    alpha_sr_z = alpha_sr[z]

                    stack_sr = load('slow_roll_alpha_f:%.4f' % (alpha_sr_z / f_j), 'Asymptotes')

                    stack = load('alpha_f:%.4f_b:%.4f_f:%.5f' % (alpha_z / f_j, b_i, f_j), 'Asymptotes')

                    stack_alpha.append([alpha_z/f_j, stack[1, idx_k]])
                stack_i.append(stack_alpha)
            stack_k.append(stack_i)
        stack_j.append(stack_k)
    return np.array(stack_j)


Buffer = load('Phi', 'Phi')

remapped_scalar_name = 'Remapped_scalar'
remapped_scalar_dir = 'Phi'

remapped_scalar = load(remapped_scalar_name,remapped_scalar_dir)

sr_name = 'slow_roll_stack'
sr_dir = 'Phi'

slow_roll_stack = load(sr_name, sr_dir)

raw_dir = 'Raw'

remapped_scalar_sr_name = 'Remapped_SR'
remapped_scalar_sr_dir = 'Phi'

remapped_scalar_sr = load(remapped_scalar_sr_name, remapped_scalar_sr_dir)


