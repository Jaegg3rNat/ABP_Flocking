"""
pseudospec_2d.py
    VERSION - 2.12.0

Author: Nathan Silvano
Date: [2025-Oct]
Description:
    .
"""
# -*- coding: utf-8 -*-


#Libraries
import numpy as np  # For array operations and mathematical functions
from matplotlib import colors, cm  # For handling colors and colormaps in plots
from matplotlib.cbook import strip_math
from scipy.fftpack import fft2,ifft2,fftfreq,fftshift,ifftshift  # For Fast Fourier Transform operations
import matplotlib.pyplot as plt  # For creating plots
from tqdm import tqdm  # For progress bars during iterations
import os  # For operating system interface, e.g., file paths
import h5py  # For handling HDF5 file format
import sys  # For accessing command-line arguments
import matplotlib.colors as mcolors
from math import atan,atan2
import matplotlib as mpl

# Espera-se
if len(sys.argv) < 3:
    print("FAILED WARNING: WRONG INPUT USAGE \n  Correct Usage:")
    print("  python runfourier.py <MU> <PE> <TIME>")
    print("Description:")
    print("  <MU>   : Admensional Growth Rate; float")
    print("  <PE> : Peclet Number; float")
    print("  <TIME>   : Simulation time; integer")
    sys.exit(1)

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
def tophat_kernel(dx,dy,nx,ny,radius):
    # Calculate the radius in grid points corresponding to the competition radius
    r_int = int(radius / dx)  # Number of grid points within the competition radius

    # Initialize the kernel matrix
    num_points_inside = 0
    m = np.zeros((1 + 2 * r_int, 1 + 2 * r_int))  # Kernel matrix of appropriate size
    # m_norm = np.pi * radius ** 2  # Normalization constant for the 2D kernel

    # Populate the kernel matrix
    for i in range(-r_int, r_int + 1):
        for j in range(-r_int, r_int + 1):
            if i ** 2 + j ** 2 <= r_int ** 2:  # Inside the competition radius
                m[i + r_int, j + r_int] = 1.0  # / m_norm
                num_points_inside += 1

    # Calculate the area of the discretized kernel
    area_discretized_kernel = num_points_inside * dx ** 2
    m /= area_discretized_kernel
    # Initialize the domain-wide kernel matrix
    m2 = np.zeros((nx, ny))  # Kernel matrix in domain space

    # Place the kernel matrix at the center of the domain
    m2[nx // 2 - r_int:nx // 2 + r_int + 1, ny // 2 - r_int:ny // 2 + r_int + 1] = m

    # Normalize the kernel with grid spacing
    m2 *= dx * dy  # Ensure the kernel accounts for the area of each grid cell
    return m2
def rk4_pseudospectral(u, Px, Py, D, f_hat, dt, n_step, kx_, ky_, v0, r, gamma, D_R):
    for i in range(n_step):
        # Compute Fourier transforms of all fields
        u_hat = fft2(u)
        Px_hat = fft2(Px)
        Py_hat = fft2(Py)

        # Perform Runge-Kutta updates for all fields
        k1_u, k1_Px, k1_Py = update_step(u_hat, Px_hat, Py_hat, kx_, ky_, D, f_hat, v0,r, gamma, D_R)

        k2_u, k2_Px, k2_Py = update_step(
            u_hat + dt / 2 * k1_u,
            Px_hat + dt / 2 * k1_Px,
            Py_hat + dt / 2 * k1_Py,
            kx_, ky_, D, f_hat, v0, r, gamma, D_R
        )

        k3_u, k3_Px, k3_Py = update_step(
            u_hat + dt / 2 * k2_u,
            Px_hat + dt / 2 * k2_Px,
            Py_hat + dt / 2 * k2_Py,
            kx_, ky_, D, f_hat, v0, r, gamma, D_R
        )

        k4_u, k4_Px, k4_Py = update_step(
            u_hat + dt * k3_u,
            Px_hat + dt * k3_Px,
            Py_hat + dt * k3_Py,
            kx_, ky_, D, f_hat, v0, r, gamma, D_R
        )

        # Combine all RK4 steps to update the solutions in Fourier space
        u_hat += dt / 6 * (k1_u + 2 * (k2_u + k3_u) + k4_u)
        Px_hat += dt / 6 * (k1_Px + 2 * (k2_Px + k3_Px) + k4_Px)
        Py_hat += dt / 6 * (k1_Py + 2 * (k2_Py + k3_Py) + k4_Py)

        # Transform back to physical space
        u = ifft2(u_hat).real
        Px = ifft2(Px_hat).real
        Py = ifft2(Py_hat).real

    return u, Px, Py
def rk2_pseudospectral(u, Px, Py, D, f_hat, dt, n_step, kx_, ky_, v0, r, gamma, D_R):
    for i in range(n_step):
        # Fourier transforms
        u_hat  = fft2(u).astype(np.complex64)
        Px_hat = fft2(Px).astype(np.complex64)
        Py_hat = fft2(Py).astype(np.complex64)

        # First stage
        k1_u, k1_Px, k1_Py = update_step(
            u_hat, Px_hat, Py_hat,
            kx_, ky_, D, f_hat, v0, r, gamma, D_R
        )

        # Midpoint fields in Fourier space
        u_hat_mid  = u_hat  + dt / 2 * k1_u
        Px_hat_mid = Px_hat + dt / 2 * k1_Px
        Py_hat_mid = Py_hat + dt / 2 * k1_Py

        # Second stage
        k2_u, k2_Px, k2_Py = update_step(
            u_hat_mid, Px_hat_mid, Py_hat_mid,
            kx_, ky_, D, f_hat, v0, r, gamma, D_R
        )

        # Update solution
        u_hat  += dt * k2_u
        Px_hat += dt * k2_Px
        Py_hat += dt * k2_Py

        # Back to physical space
        u  = ifft2(u_hat).real
        Px = ifft2(Px_hat).real
        Py = ifft2(Py_hat).real

    return u, Px, Py
def update_step(u_hat, Px_hat, Py_hat, kx_, ky_, D, f_hat, v0, r, gamma, D_R):

    # Transform back to physical space
    u = ifft2(u_hat).real
    Px = ifft2(Px_hat).real
    Py = ifft2(Py_hat).real

    # Compute convolution for interaction term
    S_rho0 = ifftshift(ifft2(u_hat * f_hat))

    # u field
    diffusion_term = - D * (ksq) * u_hat
    interaction_term = (r) * u_hat - gamma * fft2(u * S_rho0 )
    activity_term = -v0 *(1j * kx_ * Px_hat + 1j * ky_ * Py_hat)
    u_hat_updated = diffusion_term + interaction_term+ activity_term

    # Px field
    diffusion_polar_x = - D * (ksq) * Px_hat
    activity_px = -1j *( kx_) * u_hat * (v0 / 2)
    interaction_px = + (r - D_R) * Px_hat -gamma * fft2(Px * S_rho0)
    Px_hat_updated = diffusion_polar_x + activity_px  + interaction_px

    # #  Py field
    diffusion_polar_y = - D * (ksq) * Py_hat
    activity_py = -1j * ( ky_) * u_hat * (v0 / 2)
    interaction_py = + (r - D_R) * Py_hat - gamma * fft2(Py * S_rho0)
    Py_hat_updated =diffusion_polar_y + activity_py +interaction_py

    return u_hat_updated, Px_hat_updated, Py_hat_updated
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
ANALYSIS FUNCTIONS
'''
def global_order_parameter(Px, Py):
    """
    Global polarization order parameter for 2D hydrodynamic PDEs.
    """
    Px_sum = np.sum(Px)
    Py_sum = np.sum(Py)

    numerator = np.sqrt(Px_sum ** 2 + Py_sum ** 2)
    denominator = np.sum(np.sqrt(Px ** 2 + Py ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator
def global_density(u,dx,dy):
    rhomean = np.sum(u) * dx *dy
    return rhomean
############################################################################################
def compute_power_spectrum_2d(field, bins):
    """Compute the 2D power spectrum of the field and return the frequencies and power spectrum."""
    L = 1
    nx = bins
    fft_vals = np.fft.fft2(field)
    power_spectrum = np.abs(fft_vals) ** 2 / (nx * nx)  # Normalize by total number of points
    k_magnitude = np.sqrt(ksq)  # Compute radial wavenumber

    # Define radial bins
    k_bins = np.linspace(0, np.max(k_magnitude), nx // 2)
    radial_spectrum = np.zeros_like(k_bins)

    # Compute the radial power spectrum by averaging in annular bins
    for i in range(len(k_bins) - 1):
        mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i + 1])
        if np.any(mask):
            radial_spectrum[i] = np.mean(power_spectrum[mask])
    # plt.plot(radial_spectrum)

    return k_bins[:-1], radial_spectrum[:-1]  # Remove last bin to match lengths
def find_max_characteristic_frequency_2d(k_bins, radial_spectrum):
    """Find the characteristic radial wavenumber with the highest power after k=0."""

    mask = k_bins > 0  # Ignore k=0
    k_filtered = k_bins[mask]
    power_filtered = radial_spectrum[mask]

    max_index = np.argmax(power_filtered)
    return k_filtered[max_index], power_filtered[max_index]
def compute_smax(u, nx):
    k_bins, radial_spectrum = compute_power_spectrum_2d(u, nx)
    k_max, p_max = find_max_characteristic_frequency_2d(k_bins, radial_spectrum)
    return p_max
#############################################################################################
def plot_polarization_orientation(Px, Py,count,path):
    """
    Color-coded orientation plot for a 2D polarization field.
    """
    theta = np.arctan2(Py, Px)  # [-pi, pi)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(
        theta,
        origin="lower",
        cmap="hsv",
        vmin=-np.pi,
        vmax=np.pi
    )
    plt.colorbar(
        im,
        ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        label=r"$\theta$"
    )
    plt.axis("equal")
    plt.title("Polarization orientation (HSV)")
    plt.tight_layout()
    plt.savefig(f"{path}/fig{count:03d}.png", dpi=200, bbox_inches='tight')
    plt.close()
def plot_density(u,count,path):
    """
    Plot the density field using a colormap.
    """
    plt.figure(figsize=(6, 6))
    im = plt.imshow(
        u,
        origin="lower",
        cmap="viridis"
    )
    plt.colorbar(im, label="Density")
    plt.axis("equal")
    plt.title("Density Field")
    plt.tight_layout()
    plt.savefig(f"{path}/density_fig{count:03d}.png", dpi=200, bbox_inches='tight')
    plt.close()
def plot_pol_hue(Px,Py,count, path):
    fig, ax2 = plt.subplots(figsize=(10, 10))

    # --- Extent ---
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[0], bounds[1]
    extent = [x_min, x_max, y_min, y_max]

    # --- Polarization magnitude and angle ---
    mag = np.sqrt(Px ** 2 + Py ** 2)
    mag_norm = mag / (mag.max() if mag.max() > 0 else 1.0)

    angle = (np.angle(Px + 1j * Py) % (2 * np.pi)) / (2 * np.pi)  # [0,1)

    # --- HSV image ---
    hsv = np.zeros((*mag.shape, 3))
    hsv[..., 0] = angle  # hue
    hsv[..., 1] = 1.0  # saturation
    hsv[..., 2] = mag_norm  # value

    rgb = mcolors.hsv_to_rgb(hsv)

    ax2.imshow(rgb, origin="lower", extent=extent)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    # ax2.set_title("Polarization direction & magnitude")

    # --- Correct cyclic colorbar (orientation only) ---
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap="hsv", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
    # cbar.set_label("Orientation (degrees)")
    cbar.set_ticks(np.linspace(0, 1, 9))
    cbar.set_ticklabels(np.linspace(0, 360, 9).astype(int))

    # --- Save ---
    plt.savefig(f"{path}/fig_pol{count:03d}.png", dpi=200, bbox_inches="tight")
    plt.close()


'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
INITIALIZE THE SYSTEM
'''
# Seed for reproducibility of initial conditions
seed = 3
np.random.seed(seed)

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
################
DEATH_RATE = 0.15
DIFF_T = 1e-4
COMP_RADIUS = 0.1
DIFF_R = 0.01
TAU = COMP_RADIUS**2 / DIFF_T
MU = float(sys.argv[1])
PE = float(sys.argv[2])
VELOCITY = PE * DIFF_T /COMP_RADIUS
r = MU /TAU
COMPETITION_RATE = r
D_r = DIFF_R* TAU
##################
m2 = tophat_kernel(dx,dy,nx,ny, COMP_RADIUS)  # mc top hat kernel
f_hat = fft2(m2)
print(f'Kernel sum check (should be ~1.0): {np.sum(m2):.6f}')
##################
# Time setup
dt = 0.01
T = int(sys.argv[3])  # simulation duration
t = np.arange(0, T + dt, dt)
nt = len(t)
##################
# Create the arrays of frequencies that will be used in the simulation
kx = 2 * np.pi * fftfreq(nx, L / nx)
ky = 2 * np.pi * fftfreq(nx, L / ny)
kx_, ky_ = np.meshgrid(kx, ky, indexing="ij")


dmin = min(dx, dy)
kmax = np.max(np.sqrt(kx_**2 + ky_**2))
CFL = 0.3 *np.sqrt(2)/ (VELOCITY * kmax )
if dt < CFL:
    print(f'CFL Stable: {dt:.2e} < {CFL:.2e}')
else:
    print(f'🚨 CFL Unstable: {dt:.2e} >= {CFL:.2e}')

print(f'Difussion bound:{dx**2 / (4 * DIFF_T):.2e}, Time step:{dt:.2e}')
print(f'Advection bound:{(dx**2/ (4*VELOCITY)):.2e}, Time step:{dt:.2e}')

'''
# -------------------------------------------------------------------------------------
# Initial Conditions
# -------------------------------------------------------------------------------------
'''
def continue_run(pe,mu):
    f = h5py.File(f'Data/run_data_2d/_PE{pe:.2f}_MU{mu:.2f}/DATA.h5', 'r')
    meta = f['FIELD_DATA']
    Px = meta['PX_F'][:].flatten()
    Py = meta['Py_F'][:].flatten()
    u = meta['RHO_F'][:].flatten()
    f.close()
    return u,Px,Py
jojo = False
if jojo == True:
    u,Px,Py = continue_run(PE,MU)
    u = u.reshape((nx,ny))
    Px = Px.reshape((nx,ny))
    Py = Py.reshape((nx,ny))
else:
    # Initialize density field (float32 precision, values around 1)
    u0 = 1.0 + np.random.rand(nx, ny)
    u = np.copy(u0)
    # Define noise amplitude (explicitly float32)
    noise_amp = (0.00)
    bias = (0.00)  # small bias to nudge global direction
    # Initialize polarization fields (float32 precision)
    Px = (noise_amp * np.random.randn(nx, ny) + bias)
    Py = (noise_amp * np.random.randn(nx, ny))

'''
# -------------------------------------------------------------------------------------
# Directory Setup
# -------------------------------------------------------------------------------------
'''
# Define a directory to store results
main_directory = "run_data_2d"  # Main directory for results
if not os.path.exists(main_directory):
    os.makedirs(main_directory)  # Create directory if it doesn't exist

# Automatically create subdirectory based on initial configuration
lattice_size_dir = f"{main_directory}"  #
if not os.path.exists(lattice_size_dir):
    os.makedirs(lattice_size_dir)

# Create path for saving results based on N and D values
path = f"{lattice_size_dir}/_PE{PE:.2f}_MU{MU:.2f}"  # Path for specific simulation
if not os.path.exists(path):
    os.makedirs(path)

# -----------------------
# Display Simulation Parameters
# -----------------------
print("Simulation Parameters:")
print(f'Birth Rate (mu): {MU}')
print(f'Péclet Number (Pe): {PE}')
print(f"Competition Radius (R): {COMP_RADIUS}")
print("\nDomain Parameters:")
print(f"Domain Length (L): {L}")
print(f'Time Step (dt): {dt}')
print(f'Simulation Time: {T}')

print(f'Diffusion Coefficient (D): {DIFF_T}')
print(f'Polarization Diffusion Coefficient (D_R): {DIFF_R}')
print(f'Velocity (v0): {VELOCITY}')
print(f'Death Rate (d): {DEATH_RATE}')
print(f'Growth Rate (r): {r}')
print(f'Interaction Strength (gamma): {COMPETITION_RATE}')
##################
D = (DIFF_T)
dt = (dt)
v0 = (VELOCITY)
r =(r)
d = (DEATH_RATE)
gamma = (COMPETITION_RATE)
D_R = (DIFF_R)
kx_ = kx_
ky_ = ky_
f_hat = f_hat
ksq = kx_**2 + ky_**2





# def run_simulation_and_animate(u, Px, Py, t, nt, nx, bounds, path,
#                                D, f_hat, dt, v0, r, gamma, D_R,
#                                TOLERANCE_BREAK, TOLERANCE_WARNING,
#                                frame_interval=500, save_interval=10000):
#     # Make sure output directory exists
#     os.makedirs(path, exist_ok=True)
#
#     ##################
#     density_list = []
#     pol_list = []
#     pol2_list = []
#     smaxlist = []
#     # density_list.append(global_density(u, dx, dy))
#     pol_list.append(global_order_parameter(Px, Py))
#     # pol2_list.append(global_polarization_unweighted(Px,Py))
#     # smaxlist.append(compute_smax(u, nx))
#     ##################
#
#     count = 0
#     h5file = h5py.File(f"{path}/DATA.h5", "w")
#     grp1 = h5file.create_group("FIELD_DATA")
#     grp1.create_dataset(f"RHO_0", data=u)
#     grp1.create_dataset(f"PX_0", data=Px)
#     grp1.create_dataset(f"Py_0", data=Py)
#
#     # Create initial metadata
#     if "metadata" in h5file:
#         del h5file["metadata"]
#     grp = h5file.create_group("metadata")
#
#     metadata = {
#         "NET_GROWTH_RATE": MU,
#         "PECLET": PE,
#         "TIME_STEP": dt,
#         "TOTAL_TIME": T,
#         "DEATH_RATE": DEATH_RATE,
#         "DIFF_T": DIFF_T,
#         "DIFF_R": DIFF_R,
#         "RADIUS_KERNEL": COMP_RADIUS,
#         "ADMENSIONAL_DR": D_r,
#         "L": L,
#         "VELOCITY": VELOCITY,
#         "TAU": TAU,
#         "DENSITY_SERIES": density_list,
#         "POLARIZATION_DENSITY_SERIES": pol_list,
#         "ORIENTATIONAL_ORDER_SERIES": pol2_list,
#         "S_MAX_SERIES": smaxlist,
#         "LAST_SAVED_ITERATION": 0,
#     }
#
#     # Save initial metadata
#     for key, value in metadata.items():
#         grp.create_dataset(key, data=np.array([value]))
#
#     # Save initial final state
#     for dataset_name in ["RHO_F", "PX_F", "Py_F"]:
#         if dataset_name in grp1:
#             del grp1[dataset_name]
#     grp1.create_dataset(f"RHO_F", data=u)
#     grp1.create_dataset(f"PX_F", data=Px)
#     grp1.create_dataset(f"Py_F", data=Py)
#
#     # Flush to ensure data is written
#     h5file.flush()
#
#     for n in tqdm(range(1, nt)):
#         u, Px, Py = rk4_pseudospectral(u, Px, Py, D, f_hat, dt, 1,
#                                        kx_, ky_, v0, r, gamma, D_R)
#         ############################################
#         min_density = np.min(u)
#         if min_density < TOLERANCE_BREAK:
#             print(f"🚨 BREAK: Significant negative density {min_density:.2e} at iteration {n}")
#             break
#         elif min_density < TOLERANCE_WARNING:
#             print(f"⚠️ WARNING: Small negative density {min_density:.2e} at iteration {n}")
#             u = np.maximum(u, 0.0)
#         ###########################################
#
#         # Update time series data
#         if n % frame_interval == 0:
#             # density_list.append(global_density(u, dx, dy))
#             pol_list.append(global_order_parameter(Px, Py))
#             # pol2_list.append(global_polarization_unweighted(Px,Py))
#             # smaxlist.append(compute_smax(u, nx))
#
#             # --- Generate figure ---
#             plot_polarization_orientation(Px, Py,count,path)
#             plot_density(u,count,path)
#
#         # Update metadata with current time series data
#         metadata["DENSITY_SERIES"] = density_list
#         metadata["POLARIZATION_SERIES"] = pol_list
#         metadata["ORIENTATIONAL_ORDER_SERIES"] = pol2_list
#         metadata["S_MAX_SERIES"] = smaxlist
#
#         # PERIODIC STATE SAVING (every save_interval steps)
#         if n % save_interval == 0:
#             print(f"💾 Auto-saving state at iteration {n}")
#
#             # Update metadata with current iteration
#             metadata["LAST_SAVED_ITERATION"] = n
#
#             # Close and reopen file to ensure proper saving
#             h5file.close()
#             h5file = h5py.File(f"{path}/DATA.h5", "r+")
#
#             # Get reference to FIELD_DATA group
#             if "FIELD_DATA" in h5file:
#                 grp1 = h5file["FIELD_DATA"]
#             else:
#                 grp1 = h5file.create_group("FIELD_DATA")
#
#             # Remove existing metadata group
#             if "metadata" in h5file:
#                 del h5file["metadata"]
#
#             # Create new metadata group with updated data
#             grp = h5file.create_group("metadata")
#             for key, value in metadata.items():
#                 grp.create_dataset(key, data=np.array([value]))
#
#             # Remove existing final state datasets
#             for dataset_name in ["RHO_F", "PX_F", "Py_F"]:
#                 if dataset_name in grp1:
#                     del grp1[dataset_name]
#
#             # Save current state as final state
#             grp1.create_dataset(f"RHO_F", data=u)
#             grp1.create_dataset(f"PX_F", data=Px)
#             grp1.create_dataset(f"Py_F", data=Py)
#
#             # Flush to ensure data is written
#             h5file.flush()
#
#
#
#     # Final save after loop completes or breaks
#     print(f"💾 Final save at iteration {n}")
#     metadata["LAST_SAVED_ITERATION"] = n
#
#     # Ensure file is open for final save
#     if not h5file.id.valid:
#         h5file = h5py.File(f"{path}/DATA.h5", "r+")
#
#     # Get reference to FIELD_DATA group
#     if "FIELD_DATA" in h5file:
#         grp1 = h5file["FIELD_DATA"]
#     else:
#         grp1 = h5file.create_group("FIELD_DATA")
#
#     # Remove existing metadata group
#     if "metadata" in h5file:
#         del h5file["metadata"]
#
#     # Create new metadata with final data
#     grp = h5file.create_group("metadata")
#     for key, value in metadata.items():
#         grp.create_dataset(key, data=np.array([value]))
#
#     # Remove existing final state datasets
#     for dataset_name in ["RHO_F", "PX_F", "Py_F"]:
#         if dataset_name in grp1:
#             del grp1[dataset_name]
#
#     # Save final state
#     grp1.create_dataset(f"RHO_F", data=u)
#     grp1.create_dataset(f"PX_F", data=Px)
#     grp1.create_dataset(f"Py_F", data=Py)
#
#     # Flush and close
#     h5file.flush()
#     h5file.close()
#
#


def run_simulation_and_animate(
    u, Px, Py, t, nt, nx, bounds, path,
    D, f_hat, dt, v0, r, gamma, D_R,
    TOLERANCE_BREAK, TOLERANCE_WARNING,
    frame_interval=500, save_interval=10000
):


    os.makedirs(path, exist_ok=True)

    # ===============================
    # Open HDF5 file ONCE
    # ===============================
    h5file = h5py.File(f"{path}/DATA.h5", "w")

    # -------------------------------
    # FIELD DATA
    # -------------------------------
    grp_field = h5file.create_group("FIELD_DATA")
    grp_field.create_dataset("RHO_0", data=u)
    grp_field.create_dataset("PX_0", data=Px)
    grp_field.create_dataset("Py_0", data=Py)

    # Final-state placeholders
    grp_field.create_dataset("RHO_F", data=u)
    grp_field.create_dataset("PX_F", data=Px)
    grp_field.create_dataset("Py_F", data=Py)

    # -------------------------------
    # METADATA (scalar parameters)
    # -------------------------------
    grp_meta = h5file.create_group("metadata")

    scalar_metadata = {
        "NET_GROWTH_RATE": MU,
        "PECLET": PE,
        "TIME_STEP": dt,
        "TOTAL_TIME": T,
        "DEATH_RATE": DEATH_RATE,
        "DIFF_T": DIFF_T,
        "DIFF_R": DIFF_R,
        "RADIUS_KERNEL": COMP_RADIUS,
        "ADMENSIONAL_DR": D_r,
        "L": L,
        "SIZE STEP": dx,
        "VELOCITY": VELOCITY,
        "TAU": TAU,
        "LAST_SAVED_ITERATION": 0,
    }

    for key, val in scalar_metadata.items():
        grp_meta.create_dataset(key, data=val)

    # -------------------------------
    # TIME SERIES DATASETS (appendable)
    # -------------------------------
    count = 0
    grp_ts = h5file.create_group("time_series")

    ds_pol = grp_ts.create_dataset(
        "POLARIZATION",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float64,
        chunks=True,
    )
    grp_ds = h5file.create_group("density_series")

    ds_density = grp_ds.create_dataset(
        "DENSITY",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float64,
        chunks=True,
    )
    ds_smax = grp_ds.create_dataset(
        "S_MAX",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float64,
        chunks=True,
    )

    # ===============================
    # MAIN TIME LOOP
    # ===============================
    for n in tqdm(range(1, nt)):

        u, Px, Py = rk4_pseudospectral(
            u, Px, Py, D, f_hat, dt, 1,
            kx_, ky_, v0, r, gamma, D_R
        )

        # ---------------------------
        # Stability checks
        # ---------------------------
        min_density = u.min()
        # if min_density < TOLERANCE_BREAK:
        #     print(f"🚨 BREAK: density {min_density:.2e} at step {n}")
        #     idx = np.unravel_index(np.argmin(u), u.shape)
        #     print(f'Breaking in position: {idx}')
        #     break

        if min_density < TOLERANCE_WARNING:
            # print(f"⚠️ WARNING: density {min_density:.2e} at step {n}")
            np.maximum(u, 0.0, out=u)

        # ---------------------------
        # Append time series
        # ---------------------------
        if n % frame_interval == 0:
            # ------------------------------
            pol = global_order_parameter(Px, Py)

            ds_pol.resize(ds_pol.shape[0] + 1, axis=0)
            ds_pol[-1] = pol
            # -----------------------------------
            density = global_density(u,dx,dy)

            ds_density.resize(ds_density.shape[0] + 1, axis=0)
            ds_density[-1] = density
            # -----------------------------------
            s_max = compute_smax(u,nx)

            ds_smax.resize(ds_smax.shape[0] + 1, axis=0)
            ds_smax[-1] = s_max
            # --- plotting ---
            plot_polarization_orientation(Px, Py, count, path)
            plot_density(u, count, path)
            plot_pol_hue(Px,Py,count,path)
            plt.close("all")  # CRITICAL

        # ---------------------------
        # Periodic checkpoint
        # ---------------------------
        if n % save_interval == 0:
            print(f"💾 Auto-saving at iteration {n}")

            grp_meta["LAST_SAVED_ITERATION"][...] = n

            grp_field["RHO_F"][...] = u
            grp_field["PX_F"][...] = Px
            grp_field["Py_F"][...] = Py

            h5file.flush()

    # ===============================
    # FINAL SAVE
    # ===============================
    print(f"💾 Final save at iteration {n}")

    grp_meta["LAST_SAVED_ITERATION"][...] = n
    grp_field["RHO_F"][...] = u
    grp_field["PX_F"][...] = Px
    grp_field["Py_F"][...] = Py

    h5file.flush()
    h5file.close()

run_simulation_and_animate(
    u, Px, Py, t, nt, nx, bounds, path,
    D, f_hat, dt, v0, r, gamma, D_R,
    TOLERANCE_BREAK=1e-5,
    TOLERANCE_WARNING=1e-8,
    frame_interval=500
)