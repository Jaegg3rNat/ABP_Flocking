import h5py
import os, sys, re
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rc

# Choose the font for the figure
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

### HDF5 File Inspection
def print_name(name, obj):
    """Function to print the name of groups and datasets in an HDF5 file."""
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
def print_hdf5_contents(file_path):
    """Function to open an HDF5 file and print its contents."""
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_name)


def readValues(file_path):
    """Function to read a specific dataset from an HDF5 file."""

    # file_path = f'../Data/1d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5'
    # file_path = f'../Data/turing_crossing/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5'

    with h5py.File(file_path, 'r') as f:
        print_hdf5_contents(file_path)


        meta = f['metadata']
        rhodata = meta['DENSITY_SERIES'][:].flatten()
        pol = meta['POLARIZATION_INTEGRAL_SERIES'][:].flatten()



        return rhodata, pol

ro,po, = readValues(f'../Data/1d_v1/_PE{7.5:.2f}_MU{200:.2f}/DATA.h5')
ro1,po1, = readValues(f'../Data/1d_v1/_PE{4.5:.2f}_MU{200:.2f}/DATA.h5')
fig, ax1 = plt.subplots(dpi = 900, figsize = (8,4))
time = np.arange(len(ro))
# Left y-axis (first scale)
l1, =ax1.plot(time,ro[::], '.', label=r'$\mathrm{Pe}=7.5$', color='tab:blue')
l2, =ax1.plot(time ,ro1[::],'--', color='tab:cyan', label=r'$\mathrm{Pe}=4.5$')
ax1.set_ylabel(r'$\langle \rho \rangle$ scale', color='tab:blue',fontsize = 16)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create second y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Right y-axis (second scale)
l3, =ax2.plot(time ,po[::]/(po[-1]), '^',markersize =4, color='tab:red', label=r'$\mathrm{Pe}=7.5$')
l4, =ax2.plot(time ,po1[::]/po1[-1], '-.', color='tab:orange', label=r'$\mathrm{Pe}=4.5$')

ax2.set_ylabel(r'$\psi$ scale',rotation= 270 ,color='tab:red',fontsize = 16, labelpad=18)
ax2.tick_params(axis='y', labelcolor='tab:red')

# Optional: legends
ax1.legend(loc='upper left',fontsize= 12)
ax2.legend(loc='upper right',fontsize = 12)
# Create grouped legends
# ax1.legend([l1, l3], [l1.get_label(), l3.get_label()],
#            loc='upper right', fontsize=12)
#
# ax2.legend([l2, l4], [l2.get_label(), l4.get_label()],
#            loc='upper right', fontsize=12)


plt.xlim(10,480)
ax1.set_xlabel('Simulation Time',fontsize=16)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
plt.savefig('../Draft/V2/Fig_6.pdf', bbox_inches="tight")
