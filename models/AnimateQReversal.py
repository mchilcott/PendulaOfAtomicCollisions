import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('https://raw.githubusercontent.com/mchilcott/PendulaOfAtomicCollisions/main/presentation/plots.mplstyle')

def fano(q,x):
    return (q + x)**2 / ((1+x**2)*(1+q**2))


fig, ax = plt.subplots()

xlims = (-8, 8) 

x = np.linspace(*xlims, 500)

ln, = ax.plot([], [])

divider=make_axes_locatable(ax)
delta_ax=divider.append_axes('right', size='4%', pad="2%")

delta_ax.yaxis.set_label_position("right")
delta_ax.yaxis.tick_right()

delta_ax.set_ylabel(r"$\delta_{bg}$")
delta_ax.set_yticks([-np.pi/2, 0, np.pi/2])
delta_ax.set_yticklabels(["$-\pi/2$", '0', '$\pi/2$'])
delta_ax.set_xticks([])


ax.set_xticklabels([])
ax.set_yticklabels([])

delta_ln, = delta_ax.plot([],[], linewidth=6)


def init():
    ax.set_xlim(xlims)
    ax.set_ylim(-0.1, 1.1)
    delta_ax.set_xlim([0,1])
    delta_ax.set_ylim([-np.pi/2, np.pi/2])
    return ln,

def update(frame):
    q = np.tan(frame)
    ln.set_data(x, fano(q,x))
    delta_ln.set_data([0,1], [frame, frame])
    return ln,

FA = ani.FuncAnimation(fig, update, frames=np.linspace(-np.pi/2, np.pi/2, 256)[1:-1],
                       init_func=init, blit=True, interval=1, repeat=False)

FA.save("AnimatedQSweep.mp4", writer=ani.FFMpegWriter(fps=50), dpi=400)
plt.close()
