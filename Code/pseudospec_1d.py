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