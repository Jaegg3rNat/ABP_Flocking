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
# import cv2 # This is only for video-making (optional)
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter

from scipy.signal import find_peaks

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
def instantaneous_drift_velocity(rho_prev, rho, dx, dt):
    """
    Compute instantaneous drift velocity between two consecutive states.

    Assumes periodic boundary conditions.
    """
    a = rho_prev - np.mean(rho_prev)
    b = rho - np.mean(rho)

    corr = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).real

    shift_index = np.argmax(corr)
    N = len(rho)

    if shift_index > N // 2:
        shift_index -= N

    shift = shift_index * dx
    return shift / dt
def INITIAL_CONDITIONS(initialize, pe0=None, mu0=None):
    if initialize == "random":
        u0 = (1+ 0.1 * np.random.rand(nx)).astype(np.float64)
        u = np.copy(u0)
        noise_amp = np.float32(0.01)
        bias = np.float32(0.0)  # small bias to nudge global direction
        Px = (noise_amp * np.random.randn(nx) + bias).astype(np.float64)
    else:
        folder = f'C:../Data/turing_crossing/_PE{pe0:.2f}_MU{mu0:.2f}'
        # folder = f'C:../Data/1d_v1/_PE{pe0:.2f}_MU{mu0:.2f}'
        f = h5py.File(f'{folder}/DATA.h5', 'r')
        meta = f['FIELD_DATA']
        u = meta['RHO_F'][:].flatten()
        Px = meta['PX_F'][:].flatten()
        f.close()
    return u.astype(np.float64), Px.astype(np.float64)
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
    px = Px
    op = np.sum(px) / np.sum(np.abs(px))

    return op

def create_polarization_plot_1d(u, Px, t, n, path, count):
    # ------------------------------------------
    # ------------------------------------------
    # Domain bounds and system properties
    R = 0.1
    bounds = np.array([-0.5 / R, 0.5 / R])  # Domain bounds (float32)
    L = (bounds[1] - bounds[0])  # Length of the domain (float32)

    # Numerical grid properties
    nx = 512  # Number of grid points in x (keep as int)
    # dx = (L / nx)  # Grid spacing in x (float32)
    x = np.linspace(*bounds, nx + 1)[:-1]  # Periodic in x (exclude 0 for periodicity), float32
    x_min, x_max = bounds[0], bounds[1]


    fig = plt.figure(dpi=400, figsize=(8, 4))

    # === Figure layout: 3 rows ===

    # Overall Details
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


    ax1,ax2 =fig.subplots(2,1,sharex=True, gridspec_kw={'height_ratios': [1.,  0.5], 'hspace': 5*0.05})



    ax1.axhline(y= 0, color='k', linestyle='--')
    ax2.axhline(y= 0, color='k', linestyle='--',lw = 0.4)
    rho1 = rho_x(u)
    px = p_x(Px)

    ax1.plot(x,rho1, '.-', color = 'darkgreen', lw = 1.5)
    ax2.plot(x, px, '.-', color='darkblue', lw=1.5)


    # ax1.set_ylabel(r'$\rho(u,t)$', fontsize=16,labelpad=16,)
    ax1.set_title(f'Simulation Time: {t[n]:.1f}', fontsize=14)
    psi = op(px,x)
    ax2.set_title('Order Parameter 'rf'$\psi = {psi:.3f}$', fontsize=14)


    # for i in range(2):
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # # === (3) Polarization representation (arrows + color map) ===
    # ax3[1].set_xlim([x_min, x_max])
    # ax3[1].set_ylim([-1, 1])
    # ax3[1].set_yticks([])
    # ax3[1].set_xlabel(r"$x$",fontsize = 17)

    # ax2.set_ylabel(fr"$p(u,t)$",labelpad = 3, fontsize=14)

    ax1.set_ylabel(r'$\rho(u,t)$', fontsize=16)
    ax2.set_ylabel(r'$p(u,t)$', fontsize=14)

    # Fix label positions in axes coordinates
    ax1.yaxis.set_label_coords(-0.10, 0.5)
    ax2.yaxis.set_label_coords(-0.1, 0.5)



    # === (3) Polarization representation (arrows + color map) ===
    ax2.set_xlim([x_min, x_max])
    # ax1.set_ylim([0.5, 1.5])
    ax2.set_ylim([-0.2, 0.2])

    # ax2.set_yticks([])
    ax2.set_xlabel(r"System length, $u$",fontsize = 17)

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.savefig(f"{path}/fig{count:04d}.png", bbox_inches="tight")  #
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
bounds = np.array([-0.5, 0.5], dtype=np.float64)  # Domain bounds (float32)
L = (bounds[1] - bounds[0]).astype(np.float64)  # Length of the domain (float32)

# Numerical grid properties
nx = 512  # Number of grid points in x (keep as int)
dx = np.float64(L / nx)  # Grid spacing in x (float32)
x = np.linspace(*bounds, nx + 1, dtype=np.float64)[:-1]  # Periodic in x (exclude 0 for periodicity), float32
# Biological parameters
DEATH_RATE = np.float64(0.15)
COMP_RADIUS = np.float64(0.1)
DIFF_R = np.float64(1e-3)

# Command-line inputs
MU = np.float64(float(sys.argv[1]))
PE = np.float64(float(sys.argv[2]))

DIFF_T = np.float64(1.5e-5)
TAU = np.float64(COMP_RADIUS**2) / DIFF_T
r =  np.float64(MU / TAU)
VELOCITY = np.float64(PE * DIFF_T / COMP_RADIUS)
COMPETITION_RATE = np.float64(r)  # BIRTH_RATE/(2*np.pi) - DEATH_RATE  (kept as float32)
D_r = np.float64(DIFF_R * TAU)


# Kernel (make sure m2 is float32 before FFT)
m2 = tophat_kernel_1d(dx, nx, COMP_RADIUS).astype(np.float64)  # mc top hat kernel (float32)
print(f'Check Kernel Normalization: {np.sum(m2)}')
f_hat = fft(m2).astype(np.complex128)

# Time setup (dt as float32)
dt = np.float64(0.01)
T = np.float32(float(sys.argv[3]))  # simulation duration (can remain int)
t = np.arange(0, T + float(dt), float(dt), dtype=np.float32)
nt = len(t)

# Create the arrays of frequencies that will be used in the simulation
kx_ = (2 * np.pi * fftfreq(nx, L / nx)).astype(np.float64)

# (optional) squared wavenumber if you need it later
ksq = (kx_ ** 2).astype(np.float64)
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
u, Px = INITIAL_CONDITIONS("random")
# u,Px = INITIAL_CONDITIONS('', pe0 = 4, mu0=150 )
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
##################################   INITIAL CONFIGURATION
density_list.append(rho(u))
pol_list.append(px(Px))
integral_pol_list.append(integral_px(Px))
##################################
create_polarization_plot_1d(u, Px, t, 0,  path,count) #(create initial plot)
count+=1


h5file = h5py.File(f"{path}/DATA.h5", "w")
grp1 = h5file.create_group("FIELD_DATA")
grp1.create_dataset(f"RHO_0", data=u)
grp1.create_dataset(f"PX_0", data=Px)

rho_prev = Px.copy()


# create = [int(20000 + ii*10000) for ii in range(10)]  #Create list for density snapshots
# print(create)
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
    if n % 200 == 0:
    # if n == nt -1:
    #     # v = instantaneous_drift_velocity(rho_prev, Px, dx, 1000*dt)
    #
    #     # store or average v if you want
    #     # print(v)
    #
    #     # rho_prev[:] = Px
        create_polarization_plot_1d(u, Px, t, n, path, count) #(optional create figures for animation)
        count+=1
    # ###########################################
    # Time series data
        density_list.append(rho(u))
        pol_list.append(px(Px))
        integral_pol_list.append(integral_px(Px))
#########################################
#THIS SEGMENT IS TO SAVE THE INSTANTANEOUS DENSITY AS SNAPSHOT TXT FILES (OPTIONAL)
    # if n in create:
    #     np.savetxt(f"rho{count}.txt", u)
    #     count += 1
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