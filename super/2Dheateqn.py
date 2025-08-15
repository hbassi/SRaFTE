import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange

def generate_forced_heat_equation_data(
    Nf=128,
    Nc=32,
    nu=0.05,
    dt=0.01,
    t_final=1.0,
    seed=123,
    sigma=0.1,
    amp_range=(0.5, 1.0)
):

    np.random.seed(seed)
    # Compute number of timesteps, including one extra beyond t_final:
    T_steps = int(np.round((t_final + dt) / dt))  # e.g. if t_final=1.0, dt=0.01 → T_steps=101

    # Preallocate storage for fine and coarse data
    u_f_data = np.zeros((T_steps+1, Nf, Nf), dtype=np.float64)
    u_c_data = np.zeros((T_steps+1, Nc, Nc), dtype=np.float64)
    times    = np.linspace(0, T_steps*dt, T_steps+1)

    # --------------------------------------------------
    # 1) Random Gaussian initial condition on fine grid
    # --------------------------------------------------
    # 1a) Choose random center (x0, y0) ∈ [0,1]×[0,1] and amplitude A
    x0 = np.random.uniform(0.0, 1.0)
    y0 = np.random.uniform(0.0, 1.0)
    A  = np.random.uniform(amp_range[0], amp_range[1])

    # 1b) Build coordinate arrays in [0,1] for fine grid
    x_f = (np.arange(Nf) + 0.5) / Nf
    y_f = (np.arange(Nf) + 0.5) / Nf
    X_f, Y_f = np.meshgrid(x_f, y_f, indexing='ij')

    # 1c) Compute Gaussian: u0_f = A * exp(−((X_f−x0)^2 + (Y_f−y0)^2)/(2*sigma^2))
    dist2 = (X_f - x0)**2 + (Y_f - y0)**2
    u0_f = A * np.exp(-dist2 / (2.0 * sigma**2))

    u_f_data[0] = u0_f.copy()

    # --------------------------------------------------
    # 2) Coarse initial by downsampling once (every 4th point)
    # --------------------------------------------------
    u0_c = u0_f[::4, ::4]
    u_c_data[0] = u0_c.copy()

    # --------------------------------------------------
    # 3) Build forcing f(x,y) = sin(2πx) sin(2πy) on [0,1]^2 for fine grid
    # --------------------------------------------------
    f_xy_f = np.sin(2*np.pi*X_f) * np.sin(2*np.pi*Y_f)

    # --------------------------------------------------
    # 4) Build forcing f(x,y) = sin(2πx) sin(2πy) on [0,1]^2 for coarse grid
    # --------------------------------------------------
    x_c = (np.arange(Nc) + 0.5) / Nc
    y_c = (np.arange(Nc) + 0.5) / Nc
    X_c, Y_c = np.meshgrid(x_c, y_c, indexing='ij')
    f_xy_c = np.sin(2*np.pi*X_c) * np.sin(2*np.pi*Y_c)

    # --------------------------------------------------
    # 5) Precompute Fourier wavenumbers on [0,1]^2 for fine grid
    # --------------------------------------------------
    k1d_f = np.fft.fftfreq(Nf, d=1.0/Nf) * (2.0 * np.pi)
    kx_f, ky_f = np.meshgrid(k1d_f, k1d_f, indexing='ij')
    k2_f = kx_f**2 + ky_f**2

    # --------------------------------------------------
    # 6) Precompute Fourier wavenumbers on [0,1]^2 for coarse grid
    # --------------------------------------------------
    k1d_c = np.fft.fftfreq(Nc, d=1.0/Nc) * (2.0 * np.pi)
    kx_c, ky_c = np.meshgrid(k1d_c, k1d_c, indexing='ij')
    k2_c = kx_c**2 + ky_c**2

    # --------------------------------------------------
    # 7) Fourier‐transform of the forcing on fine and coarse grids
    # --------------------------------------------------
    f_hat_f = np.fft.fft2(f_xy_f)
    f_hat_c = np.fft.fft2(f_xy_c)

    # --------------------------------------------------
    # 8) Precompute integrating‐factors for fine and coarse grids
    # --------------------------------------------------
    exp_f = np.exp(-nu * k2_f * dt)
    exp_c = np.exp(-nu * k2_c * dt)

    # --------------------------------------------------
    # 9) Initialize u_hat (spectral coefficients) on fine and coarse
    # --------------------------------------------------
    u_hat_f = np.fft.fft2(u0_f)
    u_hat_c = np.fft.fft2(u0_c)

    # --------------------------------------------------
    # 10) Time‐stepping loop (n = 1 … T_steps)
    # --------------------------------------------------
    for n in range(1, T_steps+1):
        mask_f = (k2_f > 1e-14)
        mask_c = (k2_c > 1e-14)

        # --- Fine grid update ---
        # Homogeneous decay
        u_hat_f = u_hat_f * exp_f
        # Forcing correction on nonzero modes
        u_hat_f[mask_f] += 0.01 * (
            f_hat_f[mask_f] * (1.0 - exp_f[mask_f]) / (nu * k2_f[mask_f])
        )
        # Inverse FFT → physical space
        u_f = np.real(np.fft.ifft2(u_hat_f))
        u_f_data[n] = u_f

        # --- Coarse grid update ---
        # Homogeneous decay
        u_hat_c = u_hat_c * exp_c
        # Forcing correction on nonzero modes
        u_hat_c[mask_c] += 0.01 * (
            f_hat_c[mask_c] * (1.0 - exp_c[mask_c]) / (nu * k2_c[mask_c])
        )
        # Inverse FFT → physical space
        u_c = np.real(np.fft.ifft2(u_hat_c))
        u_c_data[n] = u_c

    return u_f_data, u_c_data, times


# === Generate dataset of num_samples samples, storing all in one big array ===
num_samples = 20 
Nf = 128
Nc = 32
nu = 0.1    # as specified
dt = 0.01
t_final = 10.0

# Compute number of timesteps (including the extra one)
T_steps = int(np.round((t_final + dt) / dt))  # e.g. 101
# So each trajectory has length (T_steps+1) = 102

# Preallocate two big arrays:
#   u_f_dataset: shape (num_samples, T_steps+1, Nf, Nf)
#   u_c_dataset: shape (num_samples, T_steps+1, Nc, Nc)
u_f_dataset = np.zeros((num_samples, T_steps+1, Nf, Nf), dtype=np.float64)
u_c_dataset = np.zeros((num_samples, T_steps+1, Nc, Nc), dtype=np.float64)
times_dataset = np.zeros((num_samples, T_steps+1), dtype=np.float64)

# Create directories (if you still want to save individual samples)
os.makedirs("/pscratch/sd/h/hbassi/dataset/fine", exist_ok=True)
os.makedirs("/pscratch/sd/h/hbassi/dataset/coarse", exist_ok=True)

for i in trange(num_samples, desc="Generating samples"):
    seed_i = i
    # Pass `sigma` and amplitude‐range if desired; here we keep defaults
    u_f_data, u_c_data, times = generate_forced_heat_equation_data(
        Nf=Nf,
        Nc=Nc,
        nu=nu,
        dt=dt,
        t_final=t_final,
        seed=seed_i
    )

    # Store into the big arrays
    u_f_dataset[i] = u_f_data        # shape (T_steps+1, Nf, Nf)
    u_c_dataset[i] = u_c_data        # shape (T_steps+1, Nc, Nc)
    times_dataset[i] = times         # shape (T_steps+1,)

# === Done generating all samples ===
print("Shape of u_f_dataset:", u_f_dataset.shape)
print("Shape of u_c_dataset:", u_c_dataset.shape)
print("Shape of times_dataset:", times_dataset.shape)

# Save the combined arrays as single .npy files:
np.save("/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_fine_all_smooth_gauss_1k_test2A.npy", u_f_dataset)
np.save("/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_coarse_all_smooth_gauss_1k_test2A.npy", u_c_dataset)
np.save("/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_times_test_gauss.npy", times_dataset)

# (Optional) Reload to verify
u_f_dataset = np.load("/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_fine_all_smooth_gauss_1k_test2A.npy")
u_c_dataset = np.load("/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_coarse_all_smooth_gauss_1k_test2A.npy")
times_dataset = np.load("/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_times_test_gauss.npy")

# Choose a random trajectory index
num_samples = u_f_dataset.shape[0]
import random
idx = random.randint(0, num_samples - 1)

# Retrieve the number of timesteps and the time vector for this trajectory
num_timesteps = u_f_dataset.shape[1]
time_vector = times_dataset[idx]

# Plot every 10th time step
for t in range(0, num_timesteps, 10):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Coarse‐grid plot on the left
    im0 = axes[0].imshow(
        u_c_dataset[idx, t],
        cmap="viridis",
        origin="lower",
        interpolation="nearest"
    )
    axes[0].set_title(f"Coarse (t = {time_vector[t]:.2f})")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Fine‐grid plot on the right
    im1 = axes[1].imshow(
        u_f_dataset[idx, t],
        cmap="viridis",
        origin="lower",
        interpolation="nearest"
    )
    axes[1].set_title(f"Fine (t = {time_vector[t]:.2f})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'./figures/heat_gauss_ic_trajectory_idx={idx}_t={t}_f=0.1_test_traj.pdf')
    # plt.show()
