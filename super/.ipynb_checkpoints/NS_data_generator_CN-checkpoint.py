import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt

def de_alias(f_hat, N):
    """
    Applies 2/3-rule de-aliasing on the Fourier coefficients.
    """
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k)
    cutoff = N // 3
    mask = (np.abs(KX) < cutoff) & (np.abs(KY) < cutoff)
    return f_hat * mask

def spectral_filter(f_hat, k_cutoff):
    """
    Filters the Fourier coefficients to only include modes with |kx|, |ky| <= k_cutoff.
    """
    N = f_hat.shape[0]
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k)
    mask = (np.abs(KX) <= k_cutoff) & (np.abs(KY) <= k_cutoff)
    return f_hat * mask

def compute_streamfunction(omega_hat, ksq, eps=1e-10):
    """
    Computes the streamfunction in Fourier space:
       psi_hat = -omega_hat / ksq
    with zero-frequency handled by a small cutoff.
    """
    psi_hat = np.zeros_like(omega_hat, dtype=complex)
    psi_hat[ksq > eps] = -omega_hat[ksq > eps] / ksq[ksq > eps]
    return psi_hat

def navier_stokes_solver_CN(omega0, nu, dt, nsteps, N, k_cutoff=None):
    """
    Solves the 2D Navier–Stokes equations (vorticity formulation) using a pseudo-spectral method
    with a Crank–Nicolson time-stepping scheme for the linear (viscous) term and explicit treatment
    for the nonlinear term and forcing.
    
    Method overview:
      - Compute the streamfunction: psi_hat = -omega_hat/ksq.
      - Compute velocity components via spectral differentiation:
          u = dpsi/dy,  v = -dpsi/dx.
      - Differentiate omega spectrally and compute the nonlinear term in physical space:
          nonlinear = u*(domega/dx) + v*(domega/dy).
      - De-alias the nonlinear term in Fourier space.
      - Advance in time in Fourier space using:
           (1 + nu*dt/2 * ksq)*omega_hat^(n+1) =
           (1 - nu*dt/2 * ksq)*omega_hat^(n) - dt*nonlinear_hat + dt*forcing_hat.
    """
    L = 1.0
    dx = L / N

    # Create wavenumber grid (with 2*pi scaling)
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(k, k)
    ksq = KX**2 + KY**2
    ksq[0, 0] = 1e-10  # avoid division by zero

    # Compute forcing (time independent)
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    forcing = 0.1 * (np.sin(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X + Y)))
    forcing_hat = np.fft.fft2(forcing)
    
    # Initialize vorticity
    omega = omega0.copy()
    series = [omega.copy()]
    
    for step in trange(nsteps):
        # Transform current vorticity to Fourier space
        omega_hat = np.fft.fft2(omega)
        
        # Solve for streamfunction in Fourier space: psi_hat = -omega_hat/ksq
        psi_hat = compute_streamfunction(omega_hat, ksq)
        
        # Compute velocity components in physical space
        psi = np.fft.ifft2(psi_hat)
        u = np.fft.ifft2(1j * KY * psi_hat)
        v = -np.fft.ifft2(1j * KX * psi_hat)
        
        # Compute derivatives of omega (for nonlinear term) in physical space
        domega_dx = np.fft.ifft2(1j * KX * omega_hat)
        domega_dy = np.fft.ifft2(1j * KY * omega_hat)
        
        # Compute nonlinear term in physical space:
        nonlinear = np.real(u) * np.real(domega_dx) + np.real(v) * np.real(domega_dy)
        
        # Transform nonlinear term to Fourier space and de-alias it
        nonlinear_hat = np.fft.fft2(nonlinear)
        nonlinear_hat = de_alias(nonlinear_hat, N)
        
        # Crank–Nicolson update (linear viscous term implicit, nonlinear & forcing explicit)
        omega_hat_new = ((1 - nu * dt / 2 * ksq) * omega_hat - dt * nonlinear_hat + dt * forcing_hat) / (1 + nu * dt / 2 * ksq)
        
        # Optionally apply filtering to the updated vorticity if a cutoff is provided
        if k_cutoff is not None:
            omega_hat_new = spectral_filter(omega_hat_new, k_cutoff)
        else:
            omega_hat_new = de_alias(omega_hat_new, N)
        
        # Inverse transform to obtain the new vorticity in physical space
        omega = np.real(np.fft.ifft2(omega_hat_new))
        series.append(omega.copy())
    
    return np.array(series)

def generate_initial_condition_paper(N, L=1.0, r=1.0, seed=None):
    """
    Generate a random field omega0 ~ N(0, r^(3/2)*(Delta + 49I)^(-2.5))
    with periodic boundary conditions on [0,L]^2.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Wavenumber grids
    kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY = np.meshgrid(kx, ky)
    ksq = KX**2 + KY**2
    
    # Power spectrum = r^(3/2) * (49 + k^2)^(-2.5)
    spectrum = r**1.5 * (49.0 + ksq)**(-2.5)
    
    # Random complex field with independent real and imaginary parts
    random_complex = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
    
    # Multiply by the square-root of the power spectrum
    field_hat = random_complex * np.sqrt(spectrum)
    
    # Inverse FFT to get the real-space field
    omega0 = np.fft.ifft2(field_hat)
    return np.real(omega0)

def project_to_coarse(omega_fine, N_coarse):
    """
    Projects a field from a fine grid to a coarse grid via Fourier truncation.
    """
    N_fine = omega_fine.shape[0]
    omega_hat_fine = np.fft.fft2(omega_fine)
    omega_hat_fine_shifted = np.fft.fftshift(omega_hat_fine)
    start = (N_fine - N_coarse) // 2
    end = start + N_coarse
    omega_hat_coarse_shifted = omega_hat_fine_shifted[start:end, start:end]
    omega_hat_coarse = np.fft.ifftshift(omega_hat_coarse_shifted)
    omega_coarse = np.real(np.fft.ifft2(omega_hat_coarse, s=(N_coarse, N_coarse)))
    return omega_coarse

if __name__ == '__main__':
    # -------------------------------
    # Simulation parameters
    # -------------------------------
    N_fine = 128         # Fine grid resolution (128 x 128)
    N_coarse = 64        # Coarse grid resolution (64 x 64)
    nsteps = 200000         # Number of time steps
    dt = 1e-4            # Time step size
    nu = 1e-4            # Viscosity coefficient
    nsamples = 1         # Number of samples in each dataset

    # Desired Fourier mode cutoff for de-aliasing
    k_cutoff_fine = None   # Fine simulation cutoff
    k_cutoff_coarse = None  # Coarse simulation cutoff

    # Lists to hold datasets for the fine and coarse grids
    dataset_fine = []
    dataset_coarse = []

    for i in trange(nsamples):
        # Generate the initial condition on the fine grid using the paper's covariance
        omega0_fine = generate_initial_condition_paper(N_fine, L=1.0, r=7.0, seed=None)
        # Project the fine initial condition to the coarse grid via Fourier truncation
        omega0_coarse = project_to_coarse(omega0_fine, N_coarse)
        
        # Evolve the vorticity field using the Crank–Nicolson scheme
        series_fine = navier_stokes_solver_CN(omega0_fine, nu, dt, nsteps, N_fine, k_cutoff=k_cutoff_fine)
        series_coarse = navier_stokes_solver_CN(omega0_coarse, nu, dt, nsteps, N_coarse, k_cutoff=k_cutoff_coarse)
        
        dataset_fine.append(series_fine)
        dataset_coarse.append(series_coarse)
        
        if (i + 1) % 100 == 0:
            print(f'Generated sample {i + 1}/{nsamples}')

    dataset_fine = np.array(dataset_fine)      # shape: (nsamples, nsteps+1, 128, 128)
    dataset_coarse = np.array(dataset_coarse)    # shape: (nsamples, nsteps+1, 64, 64)

    # -------------------------------
    # Save the datasets
    # -------------------------------
    output_path_fine = f'/pscratch/sd/h/hbassi/NavierStokes_fine_128_nu={nu}_kcutoff={k_cutoff_fine}_CN.npy'
    output_path_coarse = f'/pscratch/sd/h/hbassi/NavierStokes_coarse_64_nu={nu}_kcutoff={k_cutoff_coarse}_CN.npy'

    for path in [output_path_fine, output_path_coarse]:
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # with open(output_path_fine, 'wb') as f:
    #     np.save(f, dataset_fine)
    # with open(output_path_coarse, 'wb') as f:
    #     np.save(f, dataset_coarse)
    
    # print(f"Fine dataset saved to '{output_path_fine}'")
    # print(f"Coarse dataset saved to '{output_path_coarse}'")
    
    # ----------------------------------------------------------
    # Plot sample dynamics from one simulation for both grids
    # ----------------------------------------------------------
    sample_index = 0
    time_indices = 1000 * np.array([0, 50, 75, 100, 125, 150, 200])
    
    sample_fine = dataset_fine[sample_index]      # shape: (nsteps+1, 128, 128)
    sample_coarse = dataset_coarse[sample_index]    # shape: (nsteps+1, 64, 64)
    
    num_plots = len(time_indices)
    fig, axes = plt.subplots(2, num_plots, figsize=(3*num_plots, 6))
    for row, sample, grid_label in zip(axes, [sample_fine, sample_coarse], ['Fine (128x128)', 'Coarse (64x64)']):
        for ax, t in zip(row, time_indices):
            im = ax.imshow(sample[t], cmap='jet', origin='lower', extent=[0, 1, 0, 1])
            ax.set_title(f'{grid_label}\nTime {t}')
            ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    plt.savefig(f'./figures/NS_example_dynamics_sample_{sample_index}_fine128_coarse64_nu={nu}_kcutoff_CN.png')
    plt.show()
