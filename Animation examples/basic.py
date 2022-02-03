"""
=========================
Simple animation examples
=========================

This example contains two animations. The first is a random walk plot. The
second is an image animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig1 = plt.figure()

x = np.linspace(-np.pi, np.pi, 100)
data = np.array([x, np.sin(x)])
l, = plt.plot([], [], 'r-')
plt.xlim(-4,4)
plt.ylim(-1, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, frames=100, fargs=(data, l),
                                   interval=1e-10, blit=False, repeat=False)

# To save the animation, use the command: line_ani.save('lines.mp4')

# To save this second animation with some metadata, use the following command:
# im_ani.save('im.mp4', metadata={'artist':'Guido'})

plt.show()


