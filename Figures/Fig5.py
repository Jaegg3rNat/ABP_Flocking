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
def integral_px(Px):
    integral = np.sum(Px) * dx
    return integral
def rho(u):
    '''
    compute the integral of the density field rho(x) at the input time
    '''
    rho = u
    # print(len(rho))
    return np.sum(rho) * dx
def readValues(file_path):
    """Function to read a specific dataset from an HDF5 file."""

    # file_path = f'../Data/1d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5'
    # file_path = f'../Data/turing_crossing/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5'

    with h5py.File(file_path, 'r') as f:
        # print_hdf5_contents(file_path)


        meta = f['FIELD_DATA']
        rhodata = meta['RHO_F'][:]
        pol = meta['PX_F'][:]

        k, Pk = compute_power_spectrum_1d(rhodata, 512)
        k_char, power_char = find_max_characteristic_frequency_1d(k, Pk)

        return rho(rhodata), integral_px(pol), power_char
def compute_power_spectrum_1d(field, bins):
    """Compute the 1D power spectrum of the field and return the frequencies and power spectrum."""

    L = 1
    nx = bins
    dx = L / nx

    fft_vals = np.fft.fft(field)
    power_spectrum = np.abs(fft_vals) ** 2 / nx  # Normalize

    # Wavenumbers
    k = np.fft.fftfreq(nx, d=dx) * 2 * np.pi

    # Keep only positive frequencies
    mask = k >= 0
    k = k[mask]
    power_spectrum = power_spectrum[mask]

    return k, power_spectrum
def find_max_characteristic_frequency_1d(k, power_spectrum):
    """Find the characteristic wavenumber with the highest power after k=0."""

    mask = k > 0  # Ignore k = 0
    k_filtered = k[mask]
    power_filtered = power_spectrum[mask]

    max_index = np.argmax(power_filtered)

    return k_filtered[max_index], power_filtered[max_index]
dx = 1/512
mu_vals = np.arange(60,220,2)
mu_vals2 = np.arange(70,149,1)
mu_vals3 = np.arange(90,185,1)
pe_list = np.arange(0,10,0.1)
pe_list2 = np.arange(4.8,10,0.1)
pe=0
# print(mu_vals)
y_list=[]
p_list=[]
s_list = []
y_list2=[]
p_list2=[]
s_list2 = []

y_list3=[]
p_list3=[]
s_list3 = []

y_list4=[]
p_list4=[]
s_list4 = []

y_list5=[]
p_list5=[]
s_list5 = []
for mu in mu_vals:
    ro,po ,smax = readValues(f'../Data/turing_crossing/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5')
    y_list.append(ro)
    p_list.append(po)
    s_list.append(smax)
pe = 4.5
for mu in mu_vals:
    ro,po,smax = readValues(f'../Data/turing_crossing/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5')
    y_list2.append(ro)
    p_list2.append(po)
    s_list2.append(smax)
#
for mu in mu_vals2:
    ro,po,smax = readValues(f'../Data/turing_backward/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5')
    y_list3.append(ro)
    p_list3.append(po)
    s_list3.append(smax)
#
pe = 7.5
for mu in mu_vals:
    ro,po,smax = readValues(f'../Data/turing_crossing/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5')
    y_list4.append(ro)
    p_list4.append(po)
    s_list4.append(smax)

#
for mu in mu_vals3:
    ro,po,smax = readValues(f'../Data/turing_backward/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5')
    y_list5.append(ro)
    p_list5.append(po)
    s_list5.append(smax)



#

fig = plt.figure(dpi=900, figsize=(7., 4))


plt.axhline(1, color='k',linestyle = '--', lw =0.6)
line4 = plt.axvline(84.2, color='k', label = r'$\mu_c^{turing}$')
line5 = plt.axvline(132.1, linestyle = ':',color='k', label = r'$\mu_c^{4.5}$')
line6 = plt.axvline(175.484, linestyle = '--',color='k', label = r'$\mu_c^{hopf}$')

line1, = plt.plot(mu_vals, y_list, '^-',color='indianred', label = r'$\mathrm{Pe}=0$', markersize = 4.5)
#
line2, = plt.plot(mu_vals, y_list2,'o-',color='darkviolet', label = r'$\mathrm{Pe}=4.5$', markersize = 5.5)
plt.plot(mu_vals2, y_list3, 'o-',color='darkviolet', mfc = 'none', markersize =5.5)
# # #
# # #
line3, = plt.plot(mu_vals, y_list4, '.-',color='gold', label = r'$\mathrm{Pe}=7.5$')
plt.plot(mu_vals3, y_list5, '.-',color='gold', mfc = 'none')
# # plt.plot(mu_vals, s_list)
# plt.plot(mu_vals, p_list, label = r'$\mathrm{Pe}=0$')
# #
# plt.plot(mu_vals, s_list2)
# #

# plt.plot(mu_vals, p_list2)


# for p in pe_list:
#     ro,po ,smax = readValues(f'../Data/turing_crossing/_PE{p:.2f}_MU{150:.2f}/DATA.h5')
#     y_list.append(ro)
#     p_list.append(po)
#     s_list.append(smax)
# for p in pe_list2:
#     ro,po ,smax = readValues(f'../Data/turing_backward/_PE{p:.2f}_MU{150:.2f}/DATA.h5')
#     y_list2.append(ro)
#     p_list2.append(po)
#     s_list2.append(smax)
# #
# plt.plot(pe_list, s_list,'-',color='r', label = r'$\mu=150$')
# plt.plot(pe_list2, s_list2,'o-',color='r',mfc ='none')
# plt.axvline(5.28, linestyle = ':',color='k')

# plt.text(mu_vals[0]-1,y_list[-1]+10, r'(\textbf{a})', fontsize=16+3,
#         verticalalignment='top',
#         bbox=dict(boxstyle="round,pad=0.25",
#                   fc="white", ec="black", lw=0.6, alpha=1))
# plt.legend(fontsize = 16)
# After creating your plots...
fig.legend(handles =[line1, line4 ,line2,line5,line3,line6],
          loc='upper center',
          bbox_to_anchor=(0.5, 0.),
          frameon=True,
          fancybox=True,
          shadow=False,
          ncol=3,
           fontsize = 12)


plt.xlabel(r'Net growth rate, $\mu$',fontsize=14)
plt.ylabel(r'Normalized density, $\langle \rho\rangle$',fontsize=14)
plt.savefig('../Draft/V2/Fig_5.pdf', bbox_inches="tight")  #