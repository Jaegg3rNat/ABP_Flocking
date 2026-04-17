import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from matplotlib import rc
import matplotlib.ticker as tkr
import matplotlib.colors as mcolors
import os

'''
2D snapshots configuration
'''

def rho_x(mu, pe):
    f = h5py.File(f'../Data/2d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5', 'r')

    # f = h5py.File(f'/data/workspaces/nathan/Logistic/Sine/sinusoidal_mu{mu:.2f}_Pe{pe:.1f}_w{2 * np.pi:.2f}/dat.h5',
    #               'r')

    grp2 = f['FIELD_DATA']
    rhodata = grp2['RHO_F'][:]


    return rhodata

def polseries(mu, pe):

    file_path = f'../Data/2d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5'
    with h5py.File(file_path, 'r') as f:
        grp1 = f['time_series']
        op = grp1['POLARIZATION'][:].flatten()

    return  op
def rgb(mu,pe):
    file_path = f'../Data/2d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5'
    with h5py.File(file_path, 'r') as f:

        grp2 = f['FIELD_DATA']

        Px = grp2['PX_F'][:]
        Py = grp2['Py_F'][:]
        rho = grp2['RHO_F'][:]

    # --- Polarization magnitude and angle ---
    mag = np.sqrt(Px ** 2 + Py ** 2)
    mag_norm = mag / (mag.max() if mag.max() > 0 else 1.0)
    angle = (np.arctan2(Py, Px) + np.pi) / (2 * np.pi)
    # --- HSV image ---
    hsv = np.zeros((*mag.shape, 3))
    hsv[..., 0] = angle  # hue
    hsv[..., 1] = 1.0  # saturation
    hsv[..., 2] = mag_norm  # value

    rgb = mcolors.hsv_to_rgb(hsv)

    #
    return rgb



###############################################
###############################################
fig = plt.figure(dpi=600, figsize=(8, 4))

# Create division
subfigs = fig.subfigures(2, 1, hspace=0.1, wspace=-1* 0.01, height_ratios=[1,0.42])
#
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


ax1 = subfigs[0].subplots(1,2)#,sharex=False, gridspec_kw={'height_ratios': [1.,  0.5], 'hspace': 0.5})
ax2 = subfigs[1].subplots(1,2)#, sharex = False, gridspec_kw={'height_ratios': [1.,  0.5], 'hspace': 0.5})
# ax3 = subfigs[2].subplots(2,1, sharex = False,gridspec_kw={'height_ratios': [1.,0.5], 'hspace': 0.3})


bounds = np.array([-0.5, 0.5])  # Domain bounds
L = bounds[1] - bounds[0]  # Length of the domain

# Numerical grid properties
nx = 128  # Number of grid points in x
dx = L / nx  # Grid spacing in x
x = np.linspace(*bounds, nx + 1)[:-1]  # Periodic in x (exclude 0 for periodicity)

ny = nx  # Number of grid points in y
dy = L / ny  # Grid spacing in y
y = np.linspace(*bounds, ny + 1)[:-1]  # Periodic in y
# # Mesh grid of the numerical space (physical domain)
x_, y_ = np.meshgrid(x, y, indexing="ij")
# --- Extent ---
x_min, x_max = bounds[0], bounds[1]
y_min, y_max = bounds[0], bounds[1]
extent = [x_min, x_max, y_min, y_max]
import matplotlib.colors as colors


rgb3 = rgb(350, 3.5)
rgb2 = rgb(350, 1)

pcm1 = ax1[0].imshow(rgb2 , extent=extent,cmap='hsv',vmin=-np.pi,
        vmax=np.pi )

pcm =ax1[1].imshow(rgb3,  extent=extent,cmap='hsv',vmin=-np.pi,
        vmax=np.pi)
ax1[0].set_xticklabels([-5,-2.5,0,2.5,5])
ax1[0].set_yticklabels([-5,-4,-2,0,2,4])
ax1[1].set_xticklabels([-5,-2.5,0,2.5,5])
ax1[1].set_yticklabels([-5,-4,-2,0,2,4])
########################################
cbar = plt.colorbar(
    pcm,
    ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
)
cbar.ax.set_yticklabels([
    r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"
],fontsize = 15)
cbar.set_label(
    r"Normalized orientation angle, $\theta$",
    rotation=270,      # angle in degrees (0 = horizontal, 90 = vertical)
    labelpad=15 ,fontsize = 12     # distance from colorbar
)
#########################################
cbar = plt.colorbar(
    pcm1,
    ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
)
cbar.ax.set_yticklabels([
    r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"
],fontsize = 15)
# cbar.set_label(
#     r"Normalized orientation angle, $\theta$",
#     rotation=270,      # angle in degrees (0 = horizontal, 90 = vertical)
#     labelpad=15      # distance from colorbar
# )
############################################
ax1[0].set_ylabel(r'$y/R$', fontsize=15)
# ax1[1].set_ylabel(r'$y$', fontsize=12)
ax1[0].set_xlabel(r'$x/R$', fontsize=15)
ax1[1].set_xlabel(r'$x/R$', fontsize=15)

ax2[0].axhline(0, color='gray', lw=1, ls='--')
ax2[1].axhline(0, color='gray', lw=1, ls='--')
op3 = polseries(350,3.5)

ax2[1].plot(op3, '-', color = 'darkblue', lw = 2)


op2 = polseries(350,1)
ax2[0].plot(op2, '-', color = 'darkblue', lw = 2)

ax2[0].set_ylabel(r'Order Parameter, $\psi$', fontsize=12)
ax2[0].set_xlabel(r'Simulation time, $t$', fontsize=12)
ax2[1].set_xlabel(r'Simulation time, $t$', fontsize=12)


text_labels = [r'(\textbf{a})', r'(\textbf{b})', r'(\textbf{c})', r'(\textbf{d})']

ax1[0].text(0.02, 0.98, text_labels[0], transform=ax1[0].transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.25",
                  fc="white", ec="black", lw=0.6, alpha=1))
ax1[1].text(0.02, 0.98, text_labels[1], transform=ax1[1].transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.25",
                  fc="white", ec="black", lw=0.6, alpha=1))
ax2[0].text(0.02, 1, text_labels[2], transform=ax2[0].transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.23",
                  fc="white", ec="black", lw=0.6, alpha=1))
ax2[1].text(0.02, 1, text_labels[3], transform=ax2[1].transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.23",
                  fc="white", ec="black", lw=0.6, alpha=1))



plt.savefig('../Draft/V2/Fig_4.pdf', bbox_inches="tight")  #