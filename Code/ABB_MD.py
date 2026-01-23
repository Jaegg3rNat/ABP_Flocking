"""
ABB_MD.py
    CODE VERSION : v.3.2

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
from matplotlib import cm
import imageio
import os


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
KEY SIMULATION FUNCTIONS

'''


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
ncycles = 1000  # number of cycles
nsteps = 100  # steps per cycle
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
diff = 0.5e-5
difr = 2e-3
jump = np.sqrt(2.0 * diff * dt)
jumpr = np.sqrt(2.0 * difr * dt)

v0 = 0.356e-3
v0dt = v0 * dt
CARRYING_CAPACITY = 50
###########################################
nbugs = 1500  # initial number of bugs

colorfrominit = 0
concentrated = 0
initpolar = 0


# display / output
save_gif_frames = True  # set True to save frames and assemble GIF at end (requires imageio)
gif_outdir = "gif_frames"
figsize = (5, 5)
summary_plot_name = "polarization_summary.png"

# -----------------------
# Initialization
# -----------------------
rng = np.random.default_rng(seed)

# arrays
xpos = np.zeros(nmax, dtype=float)
ypos = np.zeros(nmax, dtype=float)
theta = np.zeros(nmax, dtype=float)

colorin = np.zeros(nmax, dtype=float)
colortheta = np.zeros(nmax, dtype=float)
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
plt.scatter(xpos, ypos)
plt.show()
plt.close()
print(f'Number of bugs initialized: {nbugs}')
###########################################
factorc = 256.0 / (2.0 * np.pi)
colortheta[:nbugs] = factorc * theta[:nbugs]

if colorfrominit == 1:
    colorin[:nbugs] = 256.0 * ypos[:nbugs] / ycells
elif colorfrominit == 2:
    colorin[:] = 255.0
    colorin[nbugs:] = 0.0
# history arrays
history = np.zeros(ncycles + 1, dtype=int)
hlin = np.zeros(ncycles + 1, dtype=int)
hpolar = np.zeros(ncycles + 1, dtype=float)

fig_pos, ax_pos = plt.subplots(figsize=figsize)
fig_pol, ax_pol = plt.subplots(figsize=figsize)
# helper to map colorin values to a colormap
cmap_colors = cm.get_cmap('gnuplot')


def map_colors(vals, vmin=None, vmax=None, cmap=cmap_colors):
    """Map numeric values to RGBA using a colormap; zero values -> gray (dead)."""
    if vmin is None:
        vmin = np.min(vals) if vals.size else 0.0
    if vmax is None:
        vmax = np.max(vals) if vals.size else 1.0
    # avoid div by zero
    if vmax == vmin:
        vmax = vmin + 1.0
    norm = (vals - vmin) / (vmax - vmin)
    cols = cmap(norm)
    # deaths (colorin==0) -> dark gray
    dead_mask = (vals == 0)
    cols[dead_mask] = np.array([0.2, 0.2, 0.2, 1.0])
    return cols


# optional gif frame directory
if save_gif_frames:
    os.makedirs(gif_outdir, exist_ok=True)
    frames_paths = []


# initial stats and plots
lineages = len(np.unique(colorin[:nbugs]))
polarization = np.sqrt(np.sum(np.cos(theta[:nbugs])) ** 2 + np.sum(np.sin(theta[:nbugs])) ** 2) / nbugs
print(f"time=0. (of {tfinal}). {nbugs} bugs of {lineages} different lineages. Polarization={polarization:.4f}")
history[0] = nbugs
hlin[0] = lineages
hpolar[0] = polarization


# -----------------------
# main time loop
# -----------------------
for icycle in range(1, ncycles + 1):
    for istep in range(1, nsteps + 1):

        itime = (icycle - 1) * nsteps + istep
        time = itime * dt

        # Store current number of bugs at the beginning of the step
        current_nbugs = nbugs

        # Lists to track births and deaths
        to_die = []
        new_bugs = []  # Store (parent_idx, x, y, colorin, theta, colortheta)

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
                        # Mark for birth - store parent's data
                        new_bugs.append((
                            ibug,
                            xpos[ibug],
                            ypos[ibug],
                            colorin[ibug],
                            theta[ibug],
                            colortheta[ibug]
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
                colorin[:n_alive] = colorin[:current_nbugs][alive_mask]
                colortheta[:n_alive] = colortheta[:current_nbugs][alive_mask]
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
                parent_idx, x, y, col, th, cth = new_bugs[i]
                xpos[nbugs] = x
                ypos[nbugs] = y
                colorin[nbugs] = col
                theta[nbugs] = th
                colortheta[nbugs] = cth
                nbugs += 1

        # Clear the rest of the arrays if needed
        if nbugs < nmax:
            colorin[nbugs:nmax] = 0.0
            colortheta[nbugs:nmax] = 0.0

        # Motion: random (translational) + active motion
        if nbugs > 0:
            xpos[:nbugs] = (xpos[:nbugs] + jump * rng.standard_normal(nbugs) + xcells) % xcells
            ypos[:nbugs] = (ypos[:nbugs] + jump * rng.standard_normal(nbugs) + ycells) % ycells

            theta[:nbugs] = theta[:nbugs] + jumpr * rng.standard_normal(nbugs)
            colortheta[:nbugs] = factorc * theta[:nbugs]

            xpos[:nbugs] = (xpos[:nbugs] + v0dt * np.cos(theta[:nbugs]) + xcells) % xcells
            ypos[:nbugs] = (ypos[:nbugs] + v0dt * np.sin(theta[:nbugs]) + ycells) % ycells

    print(f'Number of bugs alive at cycle {icycle}, step {istep}: {nbugs}')

        # if istep == 10:
        #     break

    # end of steps loop

    # plotting
    # # --- Positions Plot ---
    # ax_pos.clear()
    # ax_pos.set_xlim(0, xcells)
    # ax_pos.set_ylim(0, ycells)
    # ax_pos.set_title(f'Positions (Cycle {icycle})')
    # if nbugs > 0:
    #     cols = map_colors(colorin[:nbugs], vmin=np.min(colorin[:nbugs]), vmax=np.max(colorin[:nbugs]))
    #     ax_pos.scatter(xpos[:nbugs], ypos[:nbugs], s=2.5, c=cols)
    # plt.pause(0.001) # Removed

    # --- Polarization Plot ---
    ax_pol.clear()
    ax_pol.set_xlim(0, xcells)
    ax_pol.set_ylim(0, ycells)
    ax_pol.set_title(f'Polarization (Cycle {icycle})')
    if nbugs > 0:
        cols_theta = map_colors(colortheta[:nbugs], vmin=np.min(colortheta[:nbugs]), vmax=np.max(colortheta[:nbugs]))
        ax_pol.scatter(xpos[:nbugs], ypos[:nbugs], s=2.5, c=cols_theta)
    # plt.pause(0.001) # Removed

    if save_gif_frames:
        # Save Positions plot frame
        # fpath_pos = os.path.join(gif_outdir, f"pos_frame_{icycle:05d}.png")
        # fig_pos.savefig(fpath_pos)
        # frames_paths.append(fpath_pos)
        # Note: You were only saving one figure previously (fig_pos),
        # I'm keeping the original logic of only appending the positions plot to the GIF list for simplicity,
        # but you might want to save and process fig_pol too.

        # Save Polarization plot frame (optional, not used in GIF assembly below)
        fpath_pol = os.path.join(gif_outdir, f"pol_frame_{icycle:05d}.png")
        fig_pol.savefig(fpath_pol)

        # print(f"Saved frame {fpath_pos}")  # Added print for logging progress

    # compute stats
    if nbugs > 0:
        lineages = len(np.unique(colorin[:nbugs]))
        polarization = np.sqrt(np.sum(np.cos(theta[:nbugs])) ** 2 + np.sum(np.sin(theta[:nbugs])) ** 2) / nbugs
    else:
        lineages = 0
        polarization = 0.0

    print(f"time={time} (of {tfinal}). {nbugs} bugs of {lineages} different lineages. Polarization={polarization:.4f}")
    history[icycle] = nbugs
    hlin[icycle] = lineages
    hpolar[icycle] = polarization

# end of time loop

# final polarization plot
# plt.ioff() # Removed
fig_summary, ax_summary = plt.subplots()
ax_summary.plot(hpolar)
ax_summary.set_xlabel('t (cycle)')
ax_summary.set_ylabel('polarization')
ax_summary.set_title('Polarization over time')

# --- Save final summary plot ---
fig_summary.savefig(summary_plot_name)
print(f"Saved summary plot to {summary_plot_name}")
plt.close(fig_summary)
plt.close(fig_pos)
plt.close(fig_pol)

# optionally assemble GIF
if save_gif_frames and frames_paths:
    gif_name = 'Dpolarization.gif'
    print(f"Assembling GIF: {gif_name} from {len(frames_paths)} frames...")
    try:
        with imageio.get_writer(gif_name, mode='I', duration=0.05) as writer:
            for p in frames_paths:
                image = imageio.v2.imread(p)
                writer.append_data(image)
        print(f"Saved GIF to {gif_name}")
    except Exception as e:
        print(f"Error assembling GIF: {e}")
        print("Ensure 'imageio' and 'imageio-ffmpeg' (if using video formats) are installed.")

# end of script