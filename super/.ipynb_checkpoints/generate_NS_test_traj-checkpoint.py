import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from NS_data_generator import *
from tqdm import trange


nu = 1e-4
k_cutoff_coarse = 7.5
k_cutoff_fine = 7.5
# --- Step 1: Load the dataset ---
# Replace with your actual file path
dataset_fine = np.load(f'/pscratch/sd/h/hbassi/NavierStokes_coarse_32_nu={nu}_kcutoff={k_cutoff_coarse}_with_forcing_no_norm_test_trajs.npy')#np.load(f'/pscratch/sd/h/hbassi/NavierStokes_coarse_32_nu={1e-3}_kcutoff={7}.npy')

# --- Step 2: Select a particular trajectory and extract its initial condition ---
sample_index = 72  # Choose the trajectory you want to simulate
trajectory = dataset_fine[sample_index]  # shape: (nsteps+1, 128, 128)
omega0 = trajectory[0]  # initial condition from the trajectory

# --- Step 3: Set simulation parameters for the full simulation ---
nsteps = 1000      # number of time steps to simulate         # viscosity coefficient
dt = 0.01          # time step size
N_fine = 32        # grid resolution (should match the dataset)

# --- Step 4: Run the simulation using the initial condition ---
extended_trajectory = navier_stokes_solver(omega0, nu, dt, nsteps, N_fine, k_cutoff=k_cutoff_fine)
test_output_path = f'/pscratch/sd/h/hbassi/NavierStokes_test_traj_coarse_nu={nu}_mode={k_cutoff_fine}_no_dealias_32to128_without_forcing_new.npy'
np.save(test_output_path, extended_trajectory)

# --- Step 5: (Optional) Visualize selected static time steps ---
time_indices = list(range(0, nsteps, 100))
fig, axes = plt.subplots(1, len(time_indices), figsize=(4 * len(time_indices), 4))
for ax, t in zip(axes, time_indices):
    im_static = ax.imshow(extended_trajectory[t], cmap='jet', origin='lower', extent=[0, 1, 0, 1])
    ax.set_title(f"Time step {t}")
    ax.axis('off')
fig.colorbar(im_static, ax=axes.ravel().tolist(), shrink=0.8)
plt.tight_layout()
plt.savefig(f'./figures/NS_test_sample_longtime_nsteps={nsteps}_sample_{sample_index}_coarse_32_no_forcing.png')
plt.show()

# # --- Step 6: Create frames for the GIF starting from index 100 ---
# os.makedirs('frames', exist_ok=True)
# filenames = []

# # We loop from index 100 to nsteps - this yields frames starting from index 100.
# for i in trange(100, nsteps):
#     plt.figure(figsize=(5, 5))
#     time_value = 1 + 0.01 * (10 + i)
#     time_str = f"{time_value:.3f}"
    
#     # Plot the current simulation data with a colorbar
#     im = plt.imshow(extended_trajectory[i], cmap='jet', origin='lower', extent=[0, 1, 0, 1])
#     plt.title(f"Coarse U at time {time_str}")
#     plt.colorbar(im)
#     plt.axis('off')
    
#     # Save the figure as a PNG file
#     filename = f"frames/frame_{i:03d}.png"
#     plt.savefig(filename, bbox_inches='tight')
#     filenames.append(filename)
#     plt.close()

# # --- Step 7: Create a GIF from the saved images using imageio ---
# gif_filename = 'NS_dynamics.gif'
# # The duration parameter (in seconds) controls the time each frame is shown.
# with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)

# print(f"GIF saved as {gif_filename}")

# # --- (Optional) Clean up the temporary frame files ---
# for filename in filenames:
#     os.remove(filename)
