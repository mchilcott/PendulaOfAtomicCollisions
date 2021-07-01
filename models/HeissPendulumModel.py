 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as ani
import matplotlib.cm as cm
import matplotlib

plt.style.use('../presentation/plots.mplstyle')

"""Fano Coupling

This is an attempt to explore the Fano resonance that arises from two
coupled oscillators.

We follow the model in stacks.iop.org/PhysScr/74/259.

This paper fixes the relative detuning of the two resonators, and
examines the system while changing the drive energy. (Presumably
analytically.)

We wish to fix the drive energy, and vary the relative detuning, with
one very broad resonance, and one small resonance, and see what the
shape does.

"""


# y = [dx1, x1, dx2, x2]

plt.rc('text', usetex=True)
plt.rc('font',**{'serif':['Palatino']})
plt.rc('text.latex', preamble="\\usepackage{siunitx}")

plt.rc('font', size=14)
plt.rc('axes', labelsize="large")


t_eval = 0


def run_expt(omega_drive, omega_osc, gamma, nu, a=1):

    omega1 = omega_osc[0]
    omega2 = omega_osc[1]
    nu12=nu
    gamma1 = gamma[0]
    gamma2 = gamma[1]
    omega = omega_drive

    c = np.array([1,0,0,0])
    
    M_system = np.array([
        [-gamma1, -omega1**2,        0,       -nu12],
        [      1,          0,        0,           0],
        [      0,      -nu12,  -gamma2,  -omega2**2],
        [      0,          0,         1,          0],
        ])

    M = (-1j * omega * np.eye(4) - M_system)
    dM_omega = (-1j * np.eye(4))
    Minv = np.linalg.inv(M)
    S = np.matmul(Minv, c)

    c1 = np.abs(S[1])
    c2 = np.abs(S[3])

    d1 = np.imag(np.log(S[1] / c1))
    d2 = np.imag(np.log(S[3] / c2))

    det1 = np.linalg.det(M)

    # Derivative of determinant:
    # \frac{d \det(A)}{d x} = \det(A) \tr\left(A^{-1} \frac{d A}{d \alpha}\right)
    det2 = det1 * np.trace(Minv*dM_omega)

    s = S[1]

    # c1,2 are the amplitudes of the (first, second) oscillators
    # d1,2 are the relative phases of the oscillators
    # det1 is the determiniaint of (-iomega - M_system), which is zero at resonances (poles)
    # det2 is the derivative of this
    return c1, c2, d1, d2, np.log(np.abs(det1)),  np.log(np.abs(det2)), s

def tests():
    c1,c2, = run_expt(1, (1,1.5), (1,0.1), 0)

    if c2 != 0:
        print("Uncoupled Failed")
    
    plt.plot(data.t, data.y.transpose())


def run_sweep(param):
    
    omega = np.linspace(0.5, 2, 300)
    c=[]

    for om in omega:
        c1 = run_expt(om, (1, param), (0.15,0.02), 0.3)
        c.append(c1)

    return omega, np.array(c)

if False:
    omega, c = run_sweep(1.5)

    plt.plot(omega, c[:,0])
    #plt.plot(omega, c[:,1])
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"Amplitude")
    plt.tight_layout()
    plt.savefig("PendulaAmp1.svg")

    plt.figure()
    plt.plot(omega, c[:,2])
    #plt.plot(omega, c[:,3])
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"Phase")
    plt.tight_layout()
    plt.savefig("PendulaPhase1.svg")
    
    plt.show()
    
if False:

    fig = plt.figure()
    omega, c = run_sweep(1)

    ln, = plt.plot(omega,c[:,0])
    plt.axvline(1.3, c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"Amplitude")
    plt.tight_layout()
    def animate(frame):
        omega, c = run_sweep(frame)
        ln.set_data(omega, c[:, 0])
        return ln,
    
    actor = ani.FuncAnimation(fig, animate, frames=np.linspace(1.0,1.6, 101)[:-1],
                              blit=True, interval=1000/24, repeat=True)#False)
    actor.save("PendulaSweep.mp4", writer=ani.FFMpegWriter(fps=24), dpi=200)

    plt.show()

if False:
    omega_2 = np.linspace(1.0, 1.6, 300)
    c=[]

    for om in omega_2:
        c1 = run_expt(1.3, (1, om), (0.15,0.02), 0.3)
        c.append(c1)
    c = np.array(c)

    plt.plot(omega_2, c[:,0], c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    #plt.plot(omega, c[:,1])
    plt.xlabel(r"$\omega_2$")
    plt.ylabel(r"Amplitude")
    plt.tight_layout()
    plt.savefig("PendulaSweepAmp.svg")

    plt.show()
    
    
if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    re = np.linspace(0.5, 1.5, 100)
    im = np.linspace([-0.5],[0.5],101)
    omega = re + 1j * np.repeat(im, len(re), axis=1)
    
    c=[]
    for om in omega.ravel():
        c1 = run_expt(om, (1, 1.2), (0.05,0.05), 0.1)
        c.append(c1[0])

    X,Y = np.meshgrid(re, im)
    Z = np.array(c).reshape(omega.shape)

    Z[(Z > 20) | (Z < -10)] = 0
    
    ax.plot_surface(X,Y,Z, alpha=0.6, cmap="viridis", rstride=1, cstride=1)

if False:
    plt.figure(figsize=(5,5.5))
    ax=plt.subplot(1,1,1)
    
    re = np.linspace(0.5, 1.5, 100)
    im = np.linspace([-0.5],[0.5],101)
    omega = re + 1j * np.repeat(im, len(re), axis=1)
    
    c=[]
    for om in omega.ravel():
        c1 = run_expt(om, (1, 1.2), (0.05,0.05), 0.2)
        c.append(c1[0])

    Z = np.array(c).reshape(omega.shape)

    Z_safe = Z.copy()
    Z_safe[(Z > 15) | (Z < -10)] = np.nan

    dx = np.diff(re[:2])[0]
    dy = np.diff(im[:2], axis=0)[0,0]
    
    plt.imshow(Z_safe, origin='lower', extent=(re[0]-0.5*dx, re[-1]+0.5*dx, im[0,0]-0.5*dy, im[-1,0]+0.5*dy))
    plt.axhline(0, alpha=0.7)

    divider=make_axes_locatable(ax)
    split=divider.append_axes('bottom', size='20%', pad=0)
    split.plot(re, Z[im.ravel()==0, :].ravel())
    split.set_xlim((re[0]-0.5*dx, re[-1]+0.5*dx))
    plt.tight_layout()
    
if False:
    param = 0
    fig = plt.figure(figsize=(5,5.5))
    ax=plt.subplot(1,1,1)
    
    re = np.linspace(0.5, 1.5, 101)
    im = np.linspace([-0.5],[0],51)
    omega = re + 1j * np.repeat(im, len(re), axis=1)

    c=[]
    # Widths
    gamma = (0.05, 0.05)
    # Coupling Strengths
    nu = 0.4
    
    for om in omega.ravel():
        c1 = run_expt(om, (1, 1.2), gamma, nu)
        c.append(c1[param])

    Z = np.array(c).reshape(omega.shape)

    Z_safe = Z.copy()
    Z_safe[(Z > 20) | (Z < -10)] = np.nan

    dx = np.diff(re[:2])[0]
    dy = np.diff(im[:2], axis=0)[0,0]
    
    img = plt.imshow(np.log(Z), origin='lower', extent=(re[0]-0.5*dx, re[-1]+0.5*dx, im[0,0]-0.5*dy, im[-1,0]+0.5*dy))
    dot, = plt.plot([],[], '.', color='tab:orange')

    divider=make_axes_locatable(ax)
    split=divider.append_axes('bottom', size='20%', pad=0)
    ln, = split.plot(re, Z[im.ravel()==0, :].ravel())
    split.set_xlim((re[0]-0.5*dx, re[-1]+0.5*dx))
    plt.tight_layout()

    def animate(frame):
        c=[]
        for om in omega.ravel():
            c1 = run_expt(om, (1, frame), gamma, nu)
            c.append(c1[param])

        Z = np.array(c).reshape(omega.shape)
        
#        Z_safe = Z.copy()
#        Z_safe[(Z > 20) | (Z < -10)] = np.nan

        img.set_data(np.log(Z))
        ln.set_data(re, Z[im.ravel()==0, :].ravel())
        dot.set_data([frame], [0])
        ax.axhline(0, alpha=0.7)
        return img, ln, dot
    
    actor = ani.FuncAnimation(fig, animate, frames=np.linspace(0.6, 1.4, 50),
                              blit=True, interval=1000/24, repeat=False)
    actor.save("Working.mp4", writer=ani.FFMpegWriter(fps=24), dpi=200)
    
def padeIIi(z, f_z, p_order, q_order):
    """This function returns a pade approximant for a function f where
    
        f_z = f(z)
    
    The polynomials are returned as the tuple (P, Q) in the usual form,
    and can be evaluated by
    
        f(z2) = np.polyval(P,z2) / np.polyval(Q, z2)

    The zeros and poles of the function can be found using np.roots
    on P and Q respectively.

    Please see (Theory of Resonances, V. I. Kuklin, et. al). In
    this book, this is a Type II approximant, found using method (i).
    """

    if len(z) != len(f_z):
        raise ValueError("z, f_z should be the same length.")
    
    #################
    # Equation (2.25)
    
    dim = max(p_order, q_order) + 1
    # z_ij
    index_range = np.arange(dim)
    I,J = np.meshgrid(index_range, index_range)
    exponent = I+J
    z_ij = np.sum(z ** exponent[:,:,None], axis=2)
    
    #  Now f_ij, and f^(2)_ij
    f_ij = np.sum(f_z * z ** exponent[:,:,None], axis=2)
    f2_ij =  np.sum(f_z**2 * z ** exponent[:,:,None], axis=2)
    
    # Finally, we get to (2.24), which we cast in matrix form to solve:
    #   M * poly = F
    # Where poly = [p coeffs, q coeffs]
    
    M = np.concatenate(
        (
            np.concatenate(
                (f_ij[1:q_order+1, 0:p_order+1] , -f2_ij[1:q_order+1, 1:q_order+1]),
                axis=1,
            ),
            np.concatenate(
                ( z_ij[0:p_order+1, 0:p_order+1] , -f_ij[0:p_order+1, 1:q_order+1]),
                axis=1
            )
        ))
    
    F = np.concatenate((f2_ij[1:q_order+1, 0], f_ij[0:p_order+1, 0]))
    
    poly, residuals, rank, s = np.linalg.lstsq(M, F, rcond=0)
    
    P = poly[0:p_order+1][::-1]
    Q = np.concatenate(([1], poly[-(q_order):]))[::-1]
    
    return (P,Q)

if False:
    z = np.arange(10)
    f_z = x**4

    P,Q = padeIIi(z,f_z, 7,3)


    plt.subplot(2,1,1)
    plt.plot(z,f_z)
    z2 = np.linspace(min(z), max(z), 100)
    plt.plot(z2, np.polyval(P,z2) / np.polyval(Q, z2))

    plt.subplot(2,1,2)

    extent=10
    
    im = np.linspace([-extent],[extent],101)
    z2 = np.linspace(-extent,extent,100) + 1j * np.repeat(im, len(re), axis=1)

    f_z2 = (np.polyval(P,z2.ravel()) / np.polyval(Q, z2.ravel())).reshape(z2.shape)
    
    plt.imshow(np.abs(f_z2))
    
    
if False:
    # Pade Approximation for analtic continuation

    # Generate true data
    re = np.linspace(0.5, 1.5, 100)
    im = np.linspace([-0.5],[0.5],101)
    omega = re + 1j * np.repeat(im, len(re), axis=1)
    
    c=[]
    for om in omega.ravel():
        c1 = run_expt(om, (1, 1.2), (0.05,0.05), 0.1)
        c.append(c1[-1])

    Z = np.array(c).reshape(omega.shape)

    Z_safe = np.abs(Z)
    #Z_safe[(Z > 20) | (Z < -10)] = np.nan
    
    fig = plt.figure()
    ax = fig.add_subplot(211)

    dx = np.diff(re[:2])[0]
    dy = np.diff(im[:2], axis=0)[0,0]
    
    img = ax.imshow(Z_safe, origin='lower', extent=(re[0]-0.5*dx, re[-1]+0.5*dx, im[0,0]-0.5*dy, im[-1,0]+0.5*dy))

    #######
    # Pade Approximant

    real_line = Z[im.ravel()==0, :].ravel()
    P,Q = padeIIi(re, real_line, 5,5)

    z2 = omega
    f_z2 = (np.polyval(P,z2.ravel()) / np.polyval(Q, z2.ravel())).reshape(z2.shape)

    ax = fig.add_subplot(212)
    ax.imshow(np.abs(f_z2), origin='lower', extent=(re[0]-0.5*dx, re[-1]+0.5*dx, im[0,0]-0.5*dy, im[-1,0]+0.5*dy))

    ylim = plt.ylim()
    xlim = plt.xlim()
    
    poles = np.roots(Q)
    plt.plot(np.real(poles), np.imag(poles), 'o')
    
    plt.ylim(ylim)
    plt.xlim(xlim)




if False:

    doNu = False

    plt.figure(figsize=(4,3))
    
    def beutler_fano(B, A, delta_bg, gamma, B0, c):
        y = A * np.sin(delta_bg + np.arctan((gamma / 2.0) / (B - B0)))**2 + c
        return y
    
    omega_drive = np.linspace(0.8, 1.2, 150)
    omega_2 = np.linspace(0.5, 1.5, 80)

    if doNu:
        param_range = np.linspace(0.01, 0.2,60)
        sm = cm.ScalarMappable(matplotlib.colors.Normalize(vmin=min(param_range), vmax=max(param_range)), cmap='viridis_r')
    else:
        # Gamma
        param_range = np.logspace(-1, -2.3,70)
        sm = cm.ScalarMappable(matplotlib.colors.LogNorm(vmin=min(param_range), vmax=max(param_range)), cmap='viridis')

    #plt.figure()
    for param in param_range:
        gamma = []
        bg_phase = []
        Res_pos = []
        diff = []
    
        for drive in omega_drive:
            c = []
            c2 = []
            for second in omega_2:
                if doNu:
                    data = run_expt(drive, (1, second), (0.01,0.05), param)
                else:
                    data = run_expt(drive, (1, second), (param,0.05), 0.1)
                c.append(data[0])
                c2.append(data[1])
                
            mean = np.mean(c)
            max_dev_pos = np.argmax(np.abs(c - mean))
            max_dev = np.abs(c[max_dev_pos] - mean)
            
            p0 = [  max_dev, np.pi/2,  0.1, drive,  mean]
            lo = [        0,       0,  0, 0,  0]
            up = [       200,   np.pi, 15, 2,  200]
            popt, _= scipy.optimize.curve_fit(beutler_fano, omega_2, c, p0=p0, bounds=(lo,up))
            
            
            
            gamma.append(popt[2])
            bg_phase.append(popt[1])
            Res_pos.append(popt[3])
            diff.append(np.sum(c - beutler_fano(omega_2, *popt)))

        #ax = plt.subplot(3,1,1)
        plt.plot(omega_drive, Res_pos, c=sm.to_rgba(param), linewidth=2)
        #ax = plt.subplot(3,1,2)
        #plt.plot(omega_drive, bg_phase, c=sm.to_rgba(param), linewidth=2)
        #ax = plt.subplot(3,1,3)
        #plt.plot(omega_drive, gamma, c=sm.to_rgba(param), linewidth=2)


    plt.ylabel("Resonance Position")
    plt.xlabel(r"$\omega_2$")

    

    cbar = plt.colorbar(sm)

    if doNu:        
        cbar.set_label(r"$\nu$")
    else:
        cbar.set_label(r"$\gamma_1$")
    plt.tight_layout()

    if doNu:
        plt.savefig("OpenChanChangeNu.svg")
    else:
        plt.savefig("OpenChanChangeGamma.svg")

if False:
    omega_drive = np.linspace(0.7, 1.3, 150)
    omega_2 = np.linspace(0.5, 1.5, 80)

    c_all = []
    
    for drive in omega_drive:
        c=[]
        for om in omega_2:
            c1 = run_expt(drive, (1, om), (0.01, 0.05), 0.2)
            c.append(c1[0])
        c_all.append(c)


    ln, = plt.plot([],[])
    plt.xlim((min(omega_2), max(omega_2)))
    plt.ylim((np.min(np.array(c_all)),np.max(np.array(c_all))))
        
    def animate(frame):

        ln.set_data(omega_2, c_all[frame])

        return ln,
    
    actor = ani.FuncAnimation(fig, animate, frames=range(len(omega_drive)),
                              blit=True, interval=1000/12, repeat=False)

if True:
    
    def beutler_fano(B, A, delta_bg, gamma, B0, c):
        y = A * np.sin(delta_bg + np.arctan((gamma / 2.0) / (B - B0)))**2 + c
        return y
    
    omega_drive = np.linspace(0.8, 1.2, 150)
    omega_2 = np.linspace(0.5, 1.5, 80)


    gamma = []
    bg_phase = []
    Res_pos = []
    diff = []
    
    for drive in omega_drive:
        c = []
        c2 = []
        for second in omega_2:
            data = run_expt(drive, (1, second), (0.05,0.05), 0.1)
            c.append(data[0])
            c2.append(data[1])
                    
        mean = np.mean(c)
        max_dev_pos = np.argmax(np.abs(c - mean))
        max_dev = np.abs(c[max_dev_pos] - mean)
        
        p0 = [  max_dev, np.pi/2,  0.1, drive,  mean]
        lo = [        0,       0,  0, 0,  0]
        up = [       200,   np.pi, 15, 2,  200]
        popt, _= scipy.optimize.curve_fit(beutler_fano, omega_2, c, p0=p0, bounds=(lo,up))
        
            
        
        gamma.append(popt[2])
        bg_phase.append(popt[1])
        Res_pos.append(popt[3])
        diff.append(np.sum(c - beutler_fano(omega_2, *popt)))
        
    # Do the 3 plots
    _, ax = plt.subplots()
    ax.plot(omega_drive, Res_pos)
    ax.plot([0.8, 1.2], [0.8, 1.2], alpha=0.6)
    ax.set_ylabel('Fano Profile Position')
    ax.set_xlabel('Drive Frequency')
    plt.savefig('PendulaPosition.svg')
    
    _, ax = plt.subplots()
    ax.plot(omega_drive, gamma)
    ax.set_ylabel('Fano Profile Width')
    ax.set_xlabel('Drive Frequency')
    plt.savefig('PendulaWidth.svg')
    
    _, ax = plt.subplots()
    ax.plot(omega_drive, bg_phase)
    ax.set_ylabel('Background Phase')
    ax.set_xlabel('Drive Frequency')
    plt.savefig('PendulaPhase.svg')

if False:
    # Let's try to get an animation going
    fig, ax = plt.subplots()

    
    p1, = ax.plot([],[], 'o', ms=30)
    p2, = ax.plot([],[], 'o', ms=20)
    s1, = ax.plot([],[], color=p1.get_color(), alpha=0.6)
    s2, = ax.plot([],[], color=p2.get_color(), alpha=0.6)
    spr, = ax.plot([],[], linewidth=2, alpha=0.6)

    tr1, = ax.plot([],[], color=p1.get_color(), alpha=0.6)
    tr2, = ax.plot([],[], color=p2.get_color(), alpha=0.6)
    
    omega = 1
    
    
    #amp1 = 0.8
    #amp2 = 0.3
    #phase1 = 0
    #phase2 = np.pi/2
    gamma1 = 0.1
                    
    amp1, phase1, amp2, phase2 = run_expt(omega_drive=omega, omega_osc=(1, 1.5), gamma=(gamma1, 0.1), nu=0.5)[:4]

    amp1 /= 1.2/gamma1
    amp2 /= 1.2/gamma1
    
    def init():
        ax.set_ylim([0,3])
        ax.set_xlim([0,4])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return p1, p2, s1, s2, spr
        
    def animate(frame):

        d1 = amp1 * np.sin(2*np.pi*omega*frame + phase1)
        d2 = amp2 * np.sin(2*np.pi*omega*frame + phase2)
        p1_y = 1.25 + 0.3 * d1**2
        p2_y = 1.25 + 0.3 * d2**2
        p1_x = 1 + d1
        p2_x = 3 + d2

        p1.set_data([p1_x], [p1_y])
        s1.set_data([1, p1_x], [3, p1_y])

        p2.set_data([p2_x], [p2_y])
        s2.set_data([3, p2_x], [3, p2_y])

        spr.set_data([p1_x, p2_x], [p1_y, p2_y])


        x = np.linspace(0,1,50)
        
        tr1.set_data(1 + amp1 * np.sin(2*np.pi*omega*(frame - x) + phase1), 1-x)
        tr2.set_data(3 + amp2 * np.sin(2*np.pi*omega*(frame - x) + phase2), 1-x)
        

        return p1, p2, s1, s2, spr, tr1, tr2
    
    actor = ani.FuncAnimation(fig, animate, init_func=init, frames=np.linspace(0,1, 51)[:-1],
                              blit=True, interval=1000/24, repeat=False)
    actor.save("Pendula.mp4", writer=ani.FFMpegWriter(fps=24), dpi=200)
    
plt.show() 

