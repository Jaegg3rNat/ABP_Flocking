import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from matplotlib import rc
import matplotlib.ticker as tkr
import matplotlib.colors as mcolors
import os


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
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
def rho(mu, pe):
    f = h5py.File(f'../Data/1d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5', 'r')

    # f = h5py.File(f'/data/workspaces/nathan/Logistic/Sine/sinusoidal_mu{mu:.2f}_Pe{pe:.1f}_w{2 * np.pi:.2f}/dat.h5',
    #               'r')

    meta = f['FIELD_DATA']
    density = meta['DENSITY_SERIES'][:].flatten()
    return np.mean(density[-100:])
def px(mu, pe):
    f = h5py.File(f'../Data/1d_v1/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5', 'r')

    # f = h5py.File(f'/data/workspaces/nathan/Logistic/Sine/sinusoidal_mu{mu:.2f}_Pe{pe:.1f}_w{2 * np.pi:.2f}/dat.h5',
    #               'r')

    meta = f['FIELD_DATA']
    px = meta['POLARIZATION_INTEGRAL_SERIES'][:].flatten()

    return np.mean(px[-100:])
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''

fig = plt.figure(dpi=900, figsize=(10, 4))

# Create division
subfigs = fig.subfigures(1, 2, hspace=0, wspace=-4 * 0.01, width_ratios=[1, 1])
#
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Configure Arial for text (no LaTeX)
# rc('font', **{
#     'family': 'sans-serif',
#     'sans-serif': ['Arial'],  # Use Arial for all text
#                  Optional: set default font size
# })
# rc('text', usetex=True)      # Disable LaTeX for text (required for Arial)
# Enable LaTeX *only for math* (to keep default symbols)
# rc('mathtext', fontset='cm')  # 'cm' = Computer Modern (default LaTeX math font)
#
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
axL = subfigs[0].subplots(1, 1)

# mu = [80 + 1 * i for i in range(1,101)]
mu = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 280]

Pe = [0 + 0.1 * i for i in range(100)]
Pe.sort()
print(Pe)
print(mu)
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
corrupt_files = check_corrupt_files(mu_values=mu, pe_values=Pe)
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
nx = len(mu)
ny = len(Pe)
bounds = np.array([mu[0], mu[-1]])
bounds2 = np.array([Pe[0], Pe[-1]])

rho_matrix = np.zeros((nx, ny))
px_matrix = np.zeros((nx, ny))

for j in tqdm(range(nx)):
    for i in (range(ny)):
        rho_matrix[j, i] = rho(mu[j], Pe[i])
        px_matrix[j, i] = px(mu[j], Pe[i])

dZdx = np.gradient(rho_matrix, axis=1)  # Gradient along x
dZdy = np.gradient(rho_matrix, axis=0)  # Gradient along y
gradient_magnitude = np.sqrt(dZdx ** 2 + dZdy ** 2)
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
line1 = np.loadtxt('results.txt')
line2 = np.loadtxt('resultsIm.txt')

# Define threshold
threshold = 1.0001

# Define colormap and set color for values under threshold
cmap = plt.get_cmap('Greens').copy()
cmap.set_under('white')  # all values below vmin will be black

# Define the colors for the gradient
colors = ["#000000", "#800080", "#8A2BE2", "#FF0000", "#FF4500"]

pm = plt.imshow(rho_matrix,
                cmap=cmap,
                vmin=threshold,
                vmax=rho_matrix.max(),
                extent=np.concatenate((bounds2, bounds)),
                origin='lower',
                aspect='auto')
cbar = fig.colorbar(pm, ax=axL)
cbar.ax.tick_params(labelsize=13)
axL.tick_params(axis='both', labelsize=15)
cbar.set_label('Normalized population abundance',
               rotation=270, fontsize=15, labelpad=22)

plt.plot(line1[:, 0], line1[:, 1], ls='-', color='black', alpha=0.8)
plt.axhline(168, 0, 10, color='black')

# Details
axL.set_ylabel('Net growth rate, ' r'$\mu $', fontsize=18)
axL.set_xlabel('Characteristic Péclet, ' r'$\mathrm{Pe}$', fontsize=18)
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
# Right PANEL:  CARRYING CAPACITY
axR = subfigs[1].subplots(1, 1)


pm = plt.imshow(px_matrix,
                cmap='Blues',
                vmin=0, vmax=1,
                extent=np.concatenate((bounds2, bounds)),
                origin='lower',
                aspect='auto')
cbar = fig.colorbar(pm, ax=axR)
cbar.ax.tick_params(labelsize=13)
axR.tick_params(axis='both', labelsize=15)
cbar.set_label('Order Parameter', rotation=270, fontsize=15, labelpad=22)

# Details
# axR.set_ylabel('Diffusive Damköhler, ' r'$\mathrm{Da} $', fontsize=13)
axR.set_xlabel('Characteristic Péclet, ' r'$\mathrm{Pe}$', fontsize=18)

# # --------------------------------------------------
# ==================================================
# ==================================================
plt.savefig('Fig_1.png', bbox_inches="tight")  #
# plt.savefig('Fig_HeatM.pdf', bbox_inches="tight")  #
