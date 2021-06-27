import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.cm as cm

plt.style.use('../presentation/plots.mplstyle')

# Graphics proterties
pend_equil = [1.5, 1.25]
pend_length = 1.75
drive_offset = -0.2

viewport_y = [0,3]
viewport_x = [0,3]

show_drive = True

# Pendulum Properties
omega_dr = 0.4
omega_1 = 1
gamma = 0.7

response = 1/(omega_1**2 + 1j * gamma * omega_dr - omega_dr**2)
# Normalize to max amplitude
amp1 = np.abs(response)**2 * (gamma)**2
phase1 = np.angle(response)



# Let's try to get an animation going
fig, ax = plt.subplots(figsize=(3,5))

pend, = ax.plot([],[], 'o', ms=30)
string, = ax.plot([],[], color=pend.get_color(), alpha=0.6)
trace, = ax.plot([],[], color=pend.get_color(), alpha=0.6)
if show_drive:
    drive, = ax.plot([],[], alpha=0.4)
    force = ax.annotate(
        '',
        xy=(pend_equil[0], pend_equil[1]+drive_offset),
        xytext=(pend_equil[0], pend_equil[1]+drive_offset),
        arrowprops = {
            'arrowstyle': "->",
            'color' : drive.get_color(),
            'linewidth' : 2
        }
    )

def init():
    ax.set_ylim(viewport_y)
    ax.set_xlim(viewport_x)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return pend, string, trace, drive
        
def animate(frame):
    
    d1 = amp1 * np.sin(2*np.pi*omega_dr*frame + phase1)
    p1_y = pend_equil[1] + 0.3 * d1**2
    p1_x = pend_equil[0] + d1

    pend.set_data([p1_x], [p1_y])
    string.set_data([pend_equil[0], p1_x], [pend_equil[1]+pend_length, p1_y])
    
    x = np.linspace(0,1,50)
    
    trace.set_data(pend_equil[0] + amp1 * np.sin(2*np.pi*omega_dr*(frame - x) + phase1), 1-x)

    if show_drive:
        drive.set_data(pend_equil[0] + np.sin(2*np.pi*omega_dr*(frame-x)), 1-x)
        force.xy=(pend_equil[0] + np.sin(2*np.pi*omega_dr*(frame)), pend_equil[1]+drive_offset)
        return pend, string, trace, drive, force
    return pend, string, trace

frames = int(round(100/omega_dr))
actor = ani.FuncAnimation(fig, animate, init_func=init, frames=np.linspace(0,1/omega_dr, frames)[1:-2],
                          blit=True, interval=1000/50, repeat=False)
actor.save("Pendulum.mp4", writer=ani.FFMpegWriter(fps=50), dpi=200)



if False:
    omega_dr = np.linspace(0.5, 1.5, 100)
    omega_1 = 1
    gamma = 0.4

    response = 1/(omega_1**2 + 1j * gamma * omega_dr - omega_dr**2)
    # Normalize to max amplitude
    amp1 = np.abs(response)**2 * (gamma)**2
    phase1 = np.angle(response)

    plt.figure()
    plt.plot(omega_dr, amp1)
    plt.xlabel("Drive Frequency")
    plt.ylabel("Amplitude")
    #plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    plt.tight_layout()
    plt.savefig("PendulumAmp.svg")

    plt.figure()
    plt.plot(omega_dr, phase1)
    plt.xlabel("Drive Frequency")
    plt.ylabel("Phase")
    #plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.savefig("PendulumPhase.svg")
    
    plt.show()
    
