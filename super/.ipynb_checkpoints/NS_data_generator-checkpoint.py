import numpy as np
import os, gc
from tqdm import trange
from numpy.lib.format import open_memmap

# -----------------------------------------------------------------------------
# 1. Spectral‑method helper functions (unchanged except for dtype conversions)
# -----------------------------------------------------------------------------
def de_alias(f_hat, N):
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing="ij")
    cutoff = N // 3
    mask = (np.abs(KX) < cutoff) & (np.abs(KY) < cutoff)
    return f_hat * mask

def spectral_filter(f_hat, k_cutoff):
    N = f_hat.shape[0]
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing="ij")
    mask = (np.abs(KX) <= k_cutoff) & (np.abs(KY) <= k_cutoff)
    return f_hat * mask

def compute_streamfunction(omega_hat, ksq, eps=1e-10):
    psi_hat = np.zeros_like(omega_hat, dtype=omega_hat.dtype)
    psi_hat[ksq > eps] = -omega_hat[ksq > eps] / ksq[ksq > eps]
    return psi_hat

def navier_stokes_solver(omega0, nu, dt, nsteps, N, k_cutoff=None):
    """Pseudo‑spectral vorticity solver; returns (nsteps+1, N, N) float32."""
    L = 1.0
    dx = L / N
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    ksq = KX**2 + KY**2
    ksq[0, 0] = 1e-10  # avoid division by zero

    # forcing term f(x) = 0.025*(sin2π(x+y) + cos2π(x+y))
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    forcing = 0.025 * (np.sin(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X + Y))).astype(
        np.float32
    )

    omega = omega0.astype(np.float32)
    out = np.empty((nsteps + 1, N, N), dtype=np.float32)
    out[0] = omega
    for step in range(nsteps):
        omega_hat = np.fft.fft2(omega).astype(np.complex64)

        psi_hat = compute_streamfunction(omega_hat, ksq)
        u = np.real(np.fft.ifft2(1j * KY * psi_hat)).astype(np.float32)
        v = -np.real(np.fft.ifft2(1j * KX * psi_hat)).astype(np.float32)

        domega_dx = np.real(np.fft.ifft2(1j * KX * omega_hat)).astype(np.float32)
        domega_dy = np.real(np.fft.ifft2(1j * KY * omega_hat)).astype(np.float32)
        nonlinear = u * domega_dx + v * domega_dy

        lap_omega = np.real(np.fft.ifft2(-ksq * omega_hat)).astype(np.float32)
        omega_new = omega + dt * (-nonlinear + nu * lap_omega + forcing)

        # Fourier filtering
        omega_new_hat = np.fft.fft2(omega_new).astype(np.complex64)
        omega_new_hat = (
            spectral_filter(omega_new_hat, k_cutoff)
            if k_cutoff is not None
            else de_alias(omega_new_hat, N)
        )
        omega = np.real(np.fft.ifft2(omega_new_hat)).astype(np.float32)
        out[step + 1] = omega
    return out

def generate_random_initial_condition(N, seed=None, mode_threshold=10.0):
    if seed is not None:
        np.random.seed(seed)
    noise_hat = np.fft.fft2(np.random.randn(N, N)).astype(np.complex64)
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing="ij")
    mask = (np.abs(KX) < mode_threshold) & (np.abs(KY) < mode_threshold)
    return np.real(np.fft.ifft2(noise_hat * mask)).astype(np.float32)

def project_to_coarse(omega_fine, N_coarse):
    N_fine = omega_fine.shape[0]
    start = (N_fine - N_coarse) // 2
    end = start + N_coarse
    omega_hat_fine_shifted = np.fft.fftshift(np.fft.fft2(omega_fine))
    omega_hat_coarse_shifted = omega_hat_fine_shifted[start:end, start:end]
    omega_hat_coarse = np.fft.ifftshift(omega_hat_coarse_shifted)
    return np.real(
        np.fft.ifft2(omega_hat_coarse, s=(N_coarse, N_coarse))
    ).astype(np.float32)

# -----------------------------------------------------------------------------
# 2. User‑configurable parameters
# -----------------------------------------------------------------------------
N_fine, N_coarse = 128, 32
nsteps = 1000
dt, nu = 0.01, 1e-4
nsamples = 20
k_cutoff_fine = 7.5
k_cutoff_coarse = 7.5
# output paths
out_fine = f"/pscratch/sd/h/{os.environ['USER']}/NavierStokes_fine_{N_fine}_nu{nu}_k{k_cutoff_fine}_test_data.npy"
out_coarse = f"/pscratch/sd/h/{os.environ['USER']}/NavierStokes_coarse_{N_coarse}_nu{nu}_k{k_cutoff_coarse}_test_data.npy"
os.makedirs(os.path.dirname(out_fine), exist_ok=True)

# -----------------------------------------------------------------------------
# 3. Allocate memmaps once
# -----------------------------------------------------------------------------
fine_mm = open_memmap(
    out_fine,
    mode="w+",
    dtype="float32",
    shape=(nsamples, nsteps + 1, N_fine, N_fine),
)
coarse_mm = open_memmap(
    out_coarse,
    mode="w+",
    dtype="float32",
    shape=(nsamples, nsteps + 1, N_coarse, N_coarse),
)

# -----------------------------------------------------------------------------
# 4. Main loop – one trajectory at a time
# -----------------------------------------------------------------------------
for i in trange(nsamples , desc="Generating trajectories"):
    omega0_f = generate_random_initial_condition(
        N_fine, seed=i + 1001, mode_threshold=k_cutoff_fine
    )
    omega0_c = project_to_coarse(omega0_f, N_coarse)

    fine_mm[i] = navier_stokes_solver(
        omega0_f, nu, dt, nsteps, N_fine, k_cutoff=k_cutoff_fine
    )
    coarse_mm[i] = navier_stokes_solver(
        omega0_c, nu, dt, nsteps, N_coarse, k_cutoff=k_cutoff_coarse
    )

    del omega0_f, omega0_c
    gc.collect()  # immediately return freed memory

print("\nDatasets saved to:")
print(f"  {out_fine}")
print(f"  {out_coarse}")
