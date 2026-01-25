"""
ABB_MD.py
    CODE VERSION : v.3.2.1

Author: Nathan Silvano
Date: [2026-Jan]
Description: This is a molecular dynamics simulation of interacting particles.
        The number of particles is not conserved due to birth and death processes
        The particles have a position and a velocity vector (Active Brownian Particles Dynamics)

    .
"""
# -*- coding: utf-8 -*-

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import matplotlib.colors as mcolors
from tqdm import tqdm


'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
from matplotlib import rc
# Global Plot Font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''

'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
SIMULATION FUNCTIONS
'''
def plot_particles_with_orientation(xpos, ypos, theta, ax=None, s=10):
    """
    Scatter plot of particles where color encodes orientation angle using HSV.

    Parameters
    ----------
    xpos, ypos : array-like
        Particle positions
    theta : array-like
        Particle orientations (radians)
    ax : matplotlib axis, optional
        Axis to plot on
    s : float
        Marker size
    """
    if ax is None:
        fig, ax = plt.subplots()

    # fig.patch.set_facecolor('xkcd:black')
    ax.set_facecolor('xkcd:black')
    # Map angles to [0, 1] for HSV
    theta_mod = np.mod(theta, 2*np.pi)
    norm = mcolors.Normalize(vmin=0, vmax=2*np.pi)

    sc = ax.scatter(
        xpos,
        ypos,
        c=theta_mod,
        cmap='hsv',
        norm=norm,
        s=s,
        edgecolors='none',
    )

    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$',fontsize = 14)
    ax.set_ylabel(r'$y$' ,fontsize = 14)

    # Colorbar to show orientation mapping
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r'Orientation $\theta$' ,fontsize = 14 ,rotation= 270)
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

    return ax
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
print('INITIALIZE THE SYSTEM\n')
seed = 3
nmax = 7000  # maximum number of bugs
xcells = 1  # system size in x (periodic box width)
ycells = 1  # system size in y
###########################################
ncycles = 10000  # number of cycles
nsteps = 500  # steps per cycle
dt = 0.001
tfinal = ncycles * nsteps * dt
###########################################
# demographic rates
dd = 0.15
bb = 0.85
d = dd * dt
b = bb * dt
###########################################
probsome = d + b
if probsome > 1.0:
    raise ValueError("Probabilities should be smaller than one")
print(f"time for one death of a particular bug: {1. / dd:.3f}. Time for one birth: {1. / bb:.3f}")
print(f"demographic timescale: {1. / (bb + dd):.3f} \n")
###########################################
_range = 0.1  # neighbor counting range
range2 = _range ** 2
###########################################
# diffusion and active speed
diff = float(sys.argv[1])#0.356e-5
difr = 2e-3
jump = np.sqrt(2.0 * diff * dt)
jumpr = np.sqrt(2.0 * difr * dt)

v0 = float(sys.argv[2]) #3.557e-3
v0dt = v0 * dt
CARRYING_CAPACITY = 50
###########################################
nbugs = 1500  # initial number of bugs

colorfrominit = 0
concentrated = 0
initpolar = 0


# display / output
save_gif_frames = True  # set True to save frames and assemble GIF at end (requires imageio)
gif_outdir = f"gif_frames_V0{v0:.4e}_DT{diff:.4e}/"
figsize = (5, 5)
summary_plot_name = f"polarization_{v0}_{diff}.png"
if save_gif_frames:
    os.makedirs(gif_outdir, exist_ok=True)


# -----------------------
# Initialization
# -----------------------
rng = np.random.default_rng(seed)

# arrays
xpos = np.zeros(nmax, dtype=float)
ypos = np.zeros(nmax, dtype=float)
theta = np.zeros(nmax, dtype=float)


###########################################
# initial placement
if concentrated == 0:
    xpos[:nbugs] = xcells * rng.random(nbugs)
    ypos[:nbugs] = ycells * rng.random(nbugs)
elif concentrated == 1:
    xpos[:nbugs] = xcells / 2.0
    ypos[:nbugs] = ycells / 2.0
elif concentrated == 2:
    xpos[:nbugs] = (xcells / 2.0) * rng.random(nbugs)
    ypos[:nbugs] = ycells * rng.random(nbugs)
else:
    raise ValueError('inconsistent initial distribution')
###########################################
# initial polarizations
if initpolar == 0:
    theta[:nbugs] = 2.0 * np.pi * rng.random(nbugs)
elif initpolar == 1:
    theta[:nbugs] = np.pi / 2.0
elif initpolar == 2:
    theta[:nbugs] = np.pi
else:
    raise ValueError('inconsistent initial polarization')
###########################################
# plot_particles_with_orientation(xpos, ypos, theta)
# plt.show()
# plt.close()
# print(f'Number of bugs initialized: {nbugs}')
###########################################
# history arrays
history = np.zeros(ncycles + 1, dtype=int)
# hlin = np.zeros(ncycles + 1, dtype=int)
hpolar = np.zeros(ncycles + 1, dtype=float)


# initial stats and plots
# lineages = len(np.unique(colorin[:nbugs]))
polarization = np.sqrt(np.sum(np.cos(theta[:nbugs])) ** 2 + np.sum(np.sin(theta[:nbugs])) ** 2) / nbugs
# print(f"time=0. (of {tfinal}). {nbugs} bugs. Polarization={polarization:.4f}")
history[0] = nbugs
# hlin[0] = lineages
hpolar[0] = polarization


# -----------------------
# main time loop
# -----------------------
for icycle in tqdm(range(1, ncycles + 1)):
    for istep in range(1, nsteps + 1):

        itime = (icycle - 1) * nsteps + istep
        time = itime * dt

        # Store current number of bugs at the beginning of the step
        current_nbugs = nbugs

        # Lists to track births and deaths
        to_die = []
        new_bugs = []  # Store (parent_idx, x, y, theta)

        # Compute pairwise distances ONLY if we have bugs
        if current_nbugs > 1:
            dx = xpos[:current_nbugs, None] - xpos[:current_nbugs]
            dx = dx - xcells * np.round(dx / xcells)
            dy = ypos[:current_nbugs, None] - ypos[:current_nbugs]
            dy = dy - ycells * np.round(dy / ycells)
            dist2 = dx ** 2 + dy ** 2
        else:
            dist2 = np.zeros((max(current_nbugs, 1), max(current_nbugs, 1)))

        # First pass: determine which bugs die and which give birth
        for ibug in range(current_nbugs):
            ruletarusa = rng.random()

            if ruletarusa < probsome:
                if ruletarusa < d:
                    # Mark for death
                    to_die.append(ibug)
                else:
                    # Check for birth
                    if current_nbugs > 1:
                        # Count neighbors within range (excluding self)
                        neigh = np.count_nonzero(dist2[ibug, :] < range2) - 1
                    else:
                        neigh = 0

                    prob = 1.0 - neigh / CARRYING_CAPACITY
                    prob = max(prob, 0.0)

                    if rng.random() < prob:
                        # Mark for birth - store parent's CURRENT state (no indices!)
                        new_bugs.append((
                            xpos[ibug],
                            ypos[ibug],
                            theta[ibug]
                        ))

        # Process deaths: remove marked bugs
        if to_die:
            # Create a mask of alive bugs
            alive_mask = np.ones(current_nbugs, dtype=bool)
            alive_mask[to_die] = False

            # Compact the arrays
            n_alive = np.sum(alive_mask)
            if n_alive > 0:
                xpos[:n_alive] = xpos[:current_nbugs][alive_mask]
                ypos[:n_alive] = ypos[:current_nbugs][alive_mask]
                theta[:n_alive] = theta[:current_nbugs][alive_mask]

                nbugs = n_alive
            else:
                nbugs = 0
        else:
            # No deaths, but we need to compact if we had deaths earlier in the loop
            nbugs = current_nbugs

        # Process births: add new bugs if there's space
        if new_bugs and nbugs < nmax:
            available_space = nmax - nbugs
            births_to_add = min(len(new_bugs), available_space)

            for i in range(births_to_add):
                x, y, th = new_bugs[i]
                xpos[nbugs] = x
                ypos[nbugs] = y
                theta[nbugs] = th
                nbugs += 1

        # Motion: random (translational) + active motion
        if nbugs > 0:
            xpos[:nbugs] = (xpos[:nbugs] + v0dt * np.cos(theta[:nbugs]) + jump * rng.standard_normal(nbugs) + xcells) % xcells
            ypos[:nbugs] = (ypos[:nbugs] + v0dt * np.sin(theta[:nbugs]) + jump * rng.standard_normal(nbugs) + ycells) % ycells

            theta[:nbugs] = theta[:nbugs] + jumpr * rng.standard_normal(nbugs)


    # print(f'Number of bugs alive at cycle {icycle}, step {istep}: {nbugs}')

        # if istep == 10:
        #     break

    # end of steps loop
#-----------------------------------------------------------------------------------------------------------------------#
    # compute stats
    if nbugs > 0:
        polarization = np.sqrt(np.sum(np.cos(theta[:nbugs])) ** 2 + np.sum(np.sin(theta[:nbugs])) ** 2) / nbugs
    else:
        polarization = 0.0

    # print(f"time={time} (of {tfinal}). {nbugs} bugs. Polarization={polarization:.4f}")
    # history[icycle] = nbugs
    hpolar[icycle] = polarization
    # --- Polarization Plot ---
    plot_particles_with_orientation(xpos, ypos, theta)
    plt.title(f'(Cycle {icycle}); Polarization={polarization:.4f})')
    # plt.show()
    # Save Polarization plot frame (optional, not used in GIF assembly below)
    fpath_pol = os.path.join(gif_outdir, f"pol_frame_{icycle:05d}.png")
    plt.savefig(fpath_pol, dpi=300, bbox_inches="tight")
    plt.close()
#
# # end of time loop
#
# # final polarization plot
# # plt.ioff() # Removed
# fig_summary, ax_summary = plt.subplots()
# ax_summary.plot(hpolar)
# ax_summary.set_xlabel('t (cycle)')
# ax_summary.set_ylabel('polarization')
# ax_summary.set_title('Polarization over time')
#
# # --- Save final summary plot ---
# fig_summary.savefig(summary_plot_name)
# print(f"Saved summary plot to {summary_plot_name}")
# plt.close(fig_summary)

