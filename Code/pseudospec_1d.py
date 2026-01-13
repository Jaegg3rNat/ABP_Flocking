"""
pseudospec_1d.py
    CODE VERSION : v.3.2

Author: Nathan Silvano
Date: [2026-Jan]
Description:

    .
"""
# -*- coding: utf-8 -*-

#Libraries
import numpy as np
from scipy.fftpack import fft,ifft,fftfreq,fftshift,ifftshift
import sys, os, h5py
import matplotlib.pyplot as plt
import cv2 # This is only for video-making (optional)
from tqdm import tqdm

# Espera-se
if len(sys.argv) < 3:
    print("WARNING ERROR || Correct Usage:")
    print("  python pseudospec_1d.py <MU> <PE> <T>")
    print("Description:")
    print("  <MU>   : Admensional Growth Rate; float")
    print("  <PE> : Peclet Number; float")
    print("  <T>   : Final Time; integer")

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
def tophat_kernel_1d(dx, nx, radius):
    """
    Construct a 1D top-hat kernel embedded in a domain of length nx.

    Parameters
    ----------
    dx : float
        Grid spacing.
    nx : int
        Number of grid points in the domain.
    radius : float
        Physical radius of the top-hat kernel.

    Returns
    -------
    m2 : np.ndarray
        1D top-hat kernel array (length nx), normalized so that its integral = 1.
    """
    # Convert physical radius to number of grid points
    r_int = int(radius / dx)

    # Initialize the small kernel (local)
    m = np.zeros(1 + 2 * r_int)

    # Fill the kernel within the radius
    for i in range(-r_int, r_int + 1):
        if abs(i) <= r_int:
            m[i + r_int] = 1.0

    # Normalize by the discretized integral
    area_discretized_kernel = np.sum(m) * dx
    m /= area_discretized_kernel

    # Embed the small kernel into the full domain
    m2 = np.zeros(nx)
    center = nx // 2
    m2[center - r_int:center + r_int + 1] = m

    # Multiply by dx to account for discretization in convolutions
    m2 *= dx

    return m2
def rk4_pseudospectral_1d(u, Px, D, f_hat, dt, n_step, kx_, v0, r, gamma, D_R):
    """
    Perform 1D pseudospectral RK4 integration for fields u and Px.

    Parameters
    ----------
    u : ndarray
        Scalar field in physical space.
    Px : ndarray
        Polarization field in physical space.
    D : float
        Diffusion coefficient.
    f_hat : ndarray
        Fourier transform of the kernel (for nonlocal term).
    dt : float
        Time step.
    n_step : int
        Number of time steps to perform.
    kx_ : ndarray
        Wavenumber array (1D).
    v0, b, d, gamma, D_R : floats
        Model parameters.

    Returns
    -------
    u, Px : ndarray
        Updated fields after n_step RK4 iterations.
    """
    for i in range(n_step):
        # Fourier transform of fields
        u_hat = fft(u)
        Px_hat = fft(Px)

        # Runge–Kutta 4 integration
        k1_u, k1_Px = update_step_1d(u_hat, Px_hat, kx_, D, f_hat, v0, r, gamma, D_R)

        k2_u, k2_Px = update_step_1d(
            u_hat + dt / 2 * k1_u,
            Px_hat + dt / 2 * k1_Px,
            kx_, D, f_hat, v0, r, gamma, D_R
        )

        k3_u, k3_Px = update_step_1d(
            u_hat + dt / 2 * k2_u,
            Px_hat + dt / 2 * k2_Px,
            kx_, D, f_hat, v0, r, gamma, D_R
        )

        k4_u, k4_Px = update_step_1d(
            u_hat + dt * k3_u,
            Px_hat + dt * k3_Px,
            kx_, D, f_hat, v0,r, gamma, D_R
        )

        # Combine all RK4 stages
        u_hat += dt / 6 * (k1_u + 2 * (k2_u + k3_u) + k4_u)
        Px_hat += dt / 6 * (k1_Px + 2 * (k2_Px + k3_Px) + k4_Px)

        # Transform back to physical space
        u = ifft(u_hat).real
        Px = ifft(Px_hat).real

    return u, Px
def update_step_1d(u_hat, Px_hat, kx_, D, f_hat, v0, r, gamma, D_R):
    """
    Compute one RHS update (in Fourier space) for the 1D pseudospectral RK4 scheme.

    Parameters
    ----------
    u_hat : ndarray
        Fourier transform of the scalar field u.
    Px_hat : ndarray
        Fourier transform of the polarization field Px.
    kx_ : ndarray
        1D wavenumber array.
    D : float
        Diffusion coefficient.
    f_hat : ndarray
        Fourier transform of the interaction kernel.
    v0, b, d, gamma, D_R : floats
        Model parameters.

    Returns
    -------
    u_hat_updated, Px_hat_updated : tuple of ndarrays
        Time derivatives (du_hat/dt, dPx_hat/dt) in Fourier space.
    """

    # Transform back to physical space
    u = ifft(u_hat).real
    Px = ifft(Px_hat).real

    # Compute convolution for interaction term
    S_rho0 = ifftshift(ifft(u_hat * f_hat))  # convolution with kernel

    # ----- u field -----
    diffusion_term_u = -D * (kx_ ** 2) * u_hat
    interaction_term_u = r * u_hat - gamma * fft(u * S_rho0)
    activity_term_u = -v0 * (1j * kx_ * Px_hat)
    u_hat_updated = diffusion_term_u + interaction_term_u + activity_term_u

    # ----- Px field -----
    diffusion_term_Px = -D * (kx_ ** 2) * Px_hat
    activity_term_Px = -1j * kx_ * u_hat * v0 / 2
    interaction_term_Px = fft(( r - D_R) * Px) - gamma * fft(Px * S_rho0)
    Px_hat_updated = diffusion_term_Px + activity_term_Px + interaction_term_Px

    return u_hat_updated, Px_hat_updated
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
ANALYSIS FUNCTIONS
'''
def rho(u):
    '''
    compute the integral of the density field rho(x) at the input time
    '''
    rho = u
    dx = 1 / 256
    return np.sum(rho) * dx
def rho_x(u):

    rho = u

    return rho
def px(Px):

    px = Px

    # --- Compute order parameter (1D version) ---
    mask = np.abs(px) > 1e-7
    n_included = np.count_nonzero(mask)

    if n_included > 0:
        sum_p = np.sum(px[mask])
        sum_mag = np.sum(np.abs(px[mask]))
        P = np.abs(sum_p) / sum_mag if sum_mag > 0 else 0.0
    else:
        P = 0.0

    return P
def integral_px(Px):
    integral = np.sum(Px) * dx
    return integral
def p_x(Px):
    px = Px
    return px
def op(Px,x):
    Px = Px
    # Normalize magnitude for arrow size and color intensity
    Px_max = np.max(np.abs(Px)) if np.max(np.abs(Px)) > 0 else 1.0
    Px_norm = Px / Px_max

    # Choose arrow positions (sample to avoid clutter)
    step = max(1, nx // 40)
    x_sample = x[::step]
    Px_sample = Px_norm[::step]

    return Px_sample, x_sample

def create_polarization_plot_1d(u, Px, t, n, bounds, path, count):
    # ------------------------------------------
    # ------------------------------------------
    # Domain bounds and system properties
    bounds = np.array([-0.5, 0.5])  # Domain bounds (float32)
    L = (bounds[1] - bounds[0])  # Length of the domain (float32)

    # Numerical grid properties
    nx = 512  # Number of grid points in x (keep as int)
    dx = (L / nx)  # Grid spacing in x (float32)
    x = np.linspace(*bounds, nx + 1)[:-1]  # Periodic in x (exclude 0 for periodicity), float32
    x_min, x_max = bounds[0], bounds[1]



    fig = plt.figure(dpi=300, figsize=(8, 4))

    # === Figure layout: 3 rows ===

    # Overall Details
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


    ax1,ax2 =fig.subplots(2,1,sharex=True, gridspec_kw={'height_ratios': [1.,  0.5], 'hspace': 5*0.05})



    ax1.axhline(y= 0, color='k', linestyle='--')
    ax2.axhline(y= 0, color='k', linestyle='--',lw = 0.4)
    rho1 = rho_x(u)

    ax1.plot(x,rho1, '-', color = 'darkgreen', lw = 2)


    # Px_sample, x_sample = op(Px,x)
    # # Draw arrows: right = red, left = blue
    # cmap = plt.get_cmap("seismic")  # red→positive, blue→negative
    # colors = cmap((Px_sample + 1) / 2)  # map [-1,1] → [0,1]
    #
    # for xi, pxi, c in zip(x_sample, Px_sample, colors):
    #     ax2.arrow(xi, 0, 0, 0.6 * np.sign(pxi), head_width=0.05 * (x_max - x_min) / nx,head_length=0.1, fc=c, ec=c, lw=0.5, alpha=1)
    #
    # # Add colorbar for direction
    # norm = mcolors.Normalize(vmin=-1, vmax=1)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', pad=0.2)
    # cbar.set_label("" r"($\leftarrow$ negative $|$ $\rightarrow$ positive)", fontsize = 12)
    # # Adjust position (y-shift)
    # cbar_ax = cbar.ax
    # pos = cbar_ax.get_position()
    # cbar_ax.set_position([pos.x0, pos.y0 - 0.1, pos.width, pos.height])  # shift up by 0.05


    ax1.set_ylabel(r'$\rho(x)$', fontsize=16,labelpad=16,)
    ax1.set_title(f'Simulation Time: {t[n]:.2f}', fontsize=14)
    P = px(Px)
    ax2.plot(x, Px, '-', color='darkblue', lw=2)
    ax2.set_title('Order Parameter 'r'$|{\bf P}|:'f' {P:.2f}$', fontsize=14)


    # for i in range(2):
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # # === (3) Polarization representation (arrows + color map) ===
    # ax3[1].set_xlim([x_min, x_max])
    # ax3[1].set_ylim([-1, 1])
    # ax3[1].set_yticks([])
    # ax3[1].set_xlabel(r"$x$",fontsize = 17)
    # ax2.set_ylabel("Polarization direction\n  and magnitude",labelpad = 20, fontsize=12)
    # ax2.yaxis.set_label_coords(-0.05, -0.15)  # (x, y) in axes fraction



    # === (3) Polarization representation (arrows + color map) ===
    ax2.set_xlim([x_min, x_max])
    # ax2.set_ylim([-1, 1])
    # ax1.set_ylim([0., 1.05])

    # ax2.set_yticks([])
    # ax2.set_xlabel(r"$x$",fontsize = 17)


    plt.savefig(f"{path}/fig{count:03d}.png", bbox_inches="tight")  #
    plt.close()
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
INITIALIZE THE SYSTEM
'''
# Seed for reproducibility of initial conditions
seed = 3
np.random.seed(seed)

# Domain bounds and system properties
bounds = np.array([-0.5, 0.5], dtype=np.float32)  # Domain bounds (float32)
L = (bounds[1] - bounds[0]).astype(np.float32)  # Length of the domain (float32)

# Numerical grid properties
nx = 512  # Number of grid points in x (keep as int)
dx = np.float32(L / nx)  # Grid spacing in x (float32)
x = np.linspace(*bounds, nx + 1, dtype=np.float32)[:-1]  # Periodic in x (exclude 0 for periodicity), float32



DEATH_RATE = np.float16(0.15)
COMP_RADIUS = np.float16(0.1)
DIFF_R = np.float16(1e-3)



# Command-line inputs
MU = np.float16(float(sys.argv[1]))
PE = np.float16(float(sys.argv[2]))

DIFF_T = np.float16(1.5e-5)
TAU = np.float32(COMP_RADIUS**2) / DIFF_T
r =  np.float32(MU / TAU)
VELOCITY = np.float16(PE * DIFF_T / COMP_RADIUS)
COMPETITION_RATE = np.float32(r)  # BIRTH_RATE/(2*np.pi) - DEATH_RATE  (kept as float32)
D_r = np.float16(DIFF_R * TAU)

# Kernel (make sure m2 is float32 before FFT)
m2 = tophat_kernel_1d(dx, nx, COMP_RADIUS).astype(np.float32)  # mc top hat kernel (float32)
print(f'Check Kernel Normalization: {np.sum(m2)}')
f_hat = fft(m2).astype(np.complex64)

# Time setup (dt as float32)
dt = np.float32(0.01)
T = np.float32(float(sys.argv[3]))  # simulation duration (can remain int)
t = np.arange(0, T + float(dt), float(dt), dtype=np.float32)
nt = len(t)

# Create the arrays of frequencies that will be used in the simulation
kx_ = (2 * np.pi * fftfreq(nx, L / nx)).astype(np.float32)

# (optional) squared wavenumber if you need it later
ksq = (kx_ ** 2).astype(np.float32)
print(f'Advection bound: {(dx / VELOCITY):.2e}, Time step: {dt:.2e}')


# -------------------------------------------------------------------------------------
# Directory Setup (unchanged)
# -------------------------------------------------------------------------------------
main_directory = "data_1d"
if not os.path.exists(main_directory):
    os.makedirs(main_directory)

lattice_size_dir = f"{main_directory}"
if not os.path.exists(lattice_size_dir):
    os.makedirs(lattice_size_dir)

path = f"{lattice_size_dir}/_PE{PE:.2f}_MU{MU:.2f}"
if not os.path.exists(path):
    os.makedirs(path)
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
# -------------------------------------------------------------------------------------
# Initial Conditions (arrays as float32)
# -------------------------------------------------------------------------------------
u0 = (1+ 0.1 * np.random.rand(nx)).astype(np.float16)
u = np.copy(u0)
noise_amp = np.float16(0.01)
bias = np.float16(0.0)  # small bias to nudge global direction
Px = (noise_amp * np.random.randn(nx) + bias).astype(np.float16)
# -----------------------
# Display Simulation Parameters
# -----------------------
print("Simulation Parameters:")
print(f"Competition Radius (R): {COMP_RADIUS}")
print(f'Diffusion Coefficient (D): {DIFF_T}')
print(f'Polarization Diffusion Coefficient (D_R): {DIFF_R}')
print(f'Velocity (v0): {VELOCITY}')
print(f'Interaction Strength (gamma): {COMPETITION_RATE}')
print(f'Net Growth Rate (mu): {MU}')
print(f'Peclet Number (Pe): {PE}')
print("\nDomain Parameters:")
print(f"Domain Length (L): {L}")
print(f'Time Step (dt): {dt}')
print(f'Simulation Time: {T}')
'''
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
'''
# -------------------------------------------------------------------------------------
# Run Simulation
# -------------------------------------------------------------------------------------
count = 0
TOLERANCE_WARNING = -1e-10
TOLERANCE_BREAK = -1e-6

#### Create data series array
density_list = []
pol_list = []
integral_pol_list = []
##################################
density_list.append(rho(u))
pol_list.append(px(Px))
integral_pol_list.append(integral_px(Px))
##################################
create_polarization_plot_1d(u, Px, t, 0, bounds, path, count) #(create initial plot)
count+=1


h5file = h5py.File(f"{path}/DATA.h5", "w")
grp1 = h5file.create_group("FIELD_DATA")
grp1.create_dataset(f"RHO_0", data=u)
grp1.create_dataset(f"PX_0", data=Px)


for n in tqdm(range(1, nt)):
    u, Px = rk4_pseudospectral_1d(
        u, Px,
        DIFF_T, f_hat, dt, 1,
        kx_, VELOCITY, r, COMPETITION_RATE, DIFF_R
    )


   ############################################
   #CHECK FOR NEGATIVE DENSITIES
    min_density = np.min(u)
    #
    if min_density < TOLERANCE_BREAK:
        print(f"🚨 BREAK: Significant negative density {min_density:.2e} at iteration {n}")
        break
    elif min_density < TOLERANCE_WARNING:
        print(f"⚠️  WARNING: Small negative density {min_density:.2e} at iteration {n}")
        # Continue but clip small negatives
        u = np.maximum(u, 0.0)

    ###########################################
    #GENERATE AND UPDATE FIG
    if n % 500 == 0:
        # create_polarization_plot_1d(u, Px, t, 0, bounds, path, count) #(optional create figures for animation)
    #     count+=1
    ###########################################
    # Time series data
        density_list.append(rho(u))
        pol_list.append(px(Px))
        integral_pol_list.append(integral_px(Px))
    # if check_stabilization(density_list, pol_list, n):
    #     print(f"🛑 Simulation stabilized at iteration {n}. Breaking loop.")
#########################################
metadata = {
        "NET_GROWTH_RATE": MU,
        "BIRTH_RATE": r,
        "PECLET": PE,
        "TIME_STEP": dt,
        "SIUMULATION_TIME": T,
        "DIFF_T": DIFF_T,
        "DIFF_R": DIFF_R,
        "RADIUS_KERNEL": COMP_RADIUS,
        "ADMENSIONAL_DR": D_r,
        "L": L,
        "VELOCITY": VELOCITY,
        "TAU": TAU,
        "DENSITY_SERIES": density_list,
        "POLARIZATION_SERIES": pol_list,
        "POLARIZATION_INTEGRAL_SERIES": integral_pol_list,
    }
# Save results to HDF5 file
with h5file:
    grp = h5file.create_group("metadata")  # create group "metadata"
    for key, value in metadata.items():
        # ensure the value is array-like (h5py requires this)
        grp.create_dataset(key, data=np.array([value]))
    grp1.create_dataset(f"RHO_F", data=u)
    grp1.create_dataset(f"PX_F", data=Px)
h5file.close()  # Close the HDF5 file after writing