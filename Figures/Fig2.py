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
1D snapshots configuration
'''


# ------------------------------------------
# ------------------------------------------
def check_corrupt_files(mu_values, pe_values):
    """Check for corrupted or missing .h5 files in the directory structure."""

    base_folder = f"../Data/1d_v1"
    # base_folder = f"/data/workspaces/nathan/Logistic/Sine"

    corrupt_files = []

    # Loop through mu and Pe values
    for i, mu in enumerate(mu_values):
        for j, pe in enumerate(pe_values):

            file_path = f"{base_folder}/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5"

            if not os.path.exists(file_path):  # Check if file is missing
                print(f"Missing file: {file_path}")
                corrupt_files.append(file_path)
                continue

            # Try to open the .h5 file
            try:
                with h5py.File(file_path, "r") as f:
                    _ = f.keys()  # Try reading the file structure
            except Exception as e:
                print(f"Corrupted file at mu={mu}, Pe={pe}: {file_path} - Error: {e}")
                corrupt_files.append(file_path)

    print("\nList of corrupted or missing files:", corrupt_files)
    return corrupt_files


#


def rho_x(mu, pe):
    f = h5py.File(f'../Data/1d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5', 'r')

    # f = h5py.File(f'/data/workspaces/nathan/Logistic/Sine/sinusoidal_mu{mu:.2f}_Pe{pe:.1f}_w{2 * np.pi:.2f}/dat.h5',
    #               'r')

    meta = f['FIELD_DATA']
    rho = meta['RHO_F'][:].flatten()

    return rho



def p_x(mu, pe):
    f = h5py.File(f'../Data/1d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5', 'r')

    # f = h5py.File(f'/data/workspaces/nathan/Logistic/Sine/sinusoidal_mu{mu:.2f}_Pe{pe:.1f}_w{2 * np.pi:.2f}/dat.h5',
    #               'r')

    meta = f['FIELD_DATA']
    px = meta['PX_F'][:].flatten()


    return px

def op(mu, pe, x):
    f = h5py.File(f'../Data/1d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5', 'r')

    # f = h5py.File(f'/data/workspaces/nathan/Logistic/Sine/sinusoidal_mu{mu:.2f}_Pe{pe:.1f}_w{2 * np.pi:.2f}/dat.h5',
    #               'r')

    meta = f['FIELD_DATA']
    px = meta['PX_F'][:].flatten()
    # op = np.sum(px) /512
    op = np.sum(px) / np.sum(np.abs(px))

    return op


# ------------------------------------------
# ------------------------------------------
# Domain bounds and system properties
R = 0.1
bounds = np.array([-0.5/R, 0.5/R])  # Domain bounds (float32)
L = (bounds[1] - bounds[0])  # Length of the domain (float32)
print(L)

# Numerical grid properties
nx = 512  # Number of grid points in x (keep as int)
dx = (L / nx)  # Grid spacing in x (float32)
x = np.linspace(*bounds, nx + 1)[:-1]  # Periodic in x (exclude 0 for periodicity), float32
x_min, x_max = bounds[0], bounds[1]



fig = plt.figure(dpi=600, figsize=(10, 4))

# Create division
subfigs = fig.subfigures(1, 3, hspace=0, wspace=-4 * 0.01, width_ratios=[1, 1,1])
#
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


ax1 = subfigs[0].subplots(2,1,sharex=True, gridspec_kw={'height_ratios': [1.,  0.8], 'hspace': 0.3})
ax2 = subfigs[1].subplots(2,1, sharex = True, gridspec_kw={'height_ratios': [1.,  0.8], 'hspace': 0.3})
ax3 = subfigs[2].subplots(2,1, sharex = True,gridspec_kw={'height_ratios': [1.,0.8], 'hspace': 0.3})


ax1[0].axhline(y= 1, color='k', linestyle='--',lw = 0.4)
ax2[0].axhline(y= 0, color='k', linestyle='--',lw = 0.4)
ax3[0].axhline(y= 0, color='k', linestyle='--',lw = 0.4)

ax1[1].axhline(y= 0, color='k', linestyle='--',lw = 0.4)
ax2[1].axhline(y= 0, color='k', linestyle='--',lw = 0.4)
ax3[1].axhline(y= 0, color='k', linestyle='--',lw = 0.4)


mu1 ,pe1 = 80,7.5
rho1 = rho_x(80,7.5)
px1 = p_x(80,7.5)
ax1[0].plot(x,rho1, '.-', color = 'darkgreen', lw = 1.5)
ax1[1].plot(x,px1, '.-', color = 'darkblue', lw = 1.5)

mu2 ,pe2 = 200,1.5
rho2 = rho_x(mu2,pe2)
px2 = p_x(mu2,pe2)
ax2[0].plot(x,rho2, '.-', color = 'darkgreen', lw = 1.5)
ax2[1].plot(x,px2, '.-', color = 'darkblue', lw = 1.5)

mu3,pe3 = 200,4.0
rho3 = rho_x(mu3,pe3)
px3 = p_x(mu3,pe3)
ax3[0].plot(x,rho3, '.-', color = 'darkgreen', lw = 1.5)
ax3[1].plot(x,px3, '.-', color = 'darkblue', lw = 1.5)

ax1[0].set_title(r'$\mu = 80 ; \quad \mathrm{Pe}= 7.5$', fontsize=16)
ax2[0].set_title(r'$\mu = 200 ; \quad \mathrm{Pe}= 1.5$', fontsize=16)
ax3[0].set_title(r'$\mu = 200 ; \quad \mathrm{Pe}= 4.0$', fontsize=16)

ax1[0].set_ylabel(r'$\rho(u,t)$', fontsize=16,labelpad=16,)
ax1[1].set_ylabel(r"$p(u,t)$",labelpad = 2, fontsize=14)
ax1[1].set_xlabel(r"System length, $u$",labelpad = 2, fontsize=14)
ax2[1].set_xlabel(r"System length, $u$",labelpad = 2, fontsize=14)
ax3[1].set_xlabel(r"System length, $u$",labelpad = 2, fontsize=14)


op1 = op(mu1,pe1,x)
op2 = op(mu2,pe2,x)
op3 = op(mu3,pe3,x)
ax1[1].set_title(r'$\psi = $ ' + f'{op1:.3f}', fontsize=14)
ax2[1].set_title(r'$\psi = $ ' + f'{op2:.3f}', fontsize=14)
ax3[1].set_title(r'$\psi = $ ' + f'{op3:.3f}', fontsize=14)


ax1[1].set_xlim([x_min, x_max])

ax1[0].set_ylim([0.9, 1.1])
ax1[1].set_ylim([-0.15, 0.15])
#

#
#
list = [r'(\textbf{a})', r'(\textbf{b})', r'(\textbf{c})', r'(\textbf{d})',r'(\textbf{e})', r'(\textbf{f})', r'(\textbf{g})', r'(\textbf{h})',f'(i) ',f'(j) ',f'(k) ',f'(l) ']
#

ax2[0].text(-19.4, 3.58,list[0], fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.23",
                  fc="white", ec="black", lw=0.6, alpha=1))
ax2[0].text(-5.6, 3.58,list[1], fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.23",
                  fc="white", ec="black", lw=0.6, alpha=1))
ax3[0].text(-5.6, 3.58,list[2], fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.23",
                  fc="white", ec="black", lw=0.6, alpha=1))

ax2[1].text(-19.4, 0.21,list[3], fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.23",
                  fc="white", ec="black", lw=0.6, alpha=1))
ax2[1].text(-5.6, 0.21,list[4], fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.23",
                  fc="white", ec="black", lw=0.6, alpha=1))
ax3[1].text(-5.6,2.25,list[5], fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.23",
                  fc="white", ec="black", lw=0.6, alpha=1))



plt.savefig('../Draft/V2/Fig_2.pdf', bbox_inches="tight")  #
