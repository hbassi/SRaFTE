import numpy as np

def compute_ideal_l1_error(u_f_dataset, Nc=32):
    """
    Given u_f_dataset of shape (num_samples, T,  Nf, Nf),
    compute the ℓ₁ error between u_f and its low‐pass projection
    that keeps only |k| ≤ π·Nc.

    Returns:
      errors  of shape (num_samples, T), where
        errors[i, t] = sum_{x,y} | u_f[i,t,x,y] - u_low[i,t,x,y] |
      mean_error = average of errors over all i,t snapshots.
    """
    num_samples, T_steps, Nf, _ = u_f_dataset.shape

    # 1) Build the 2D wavenumber grid once:
    k1d = np.fft.fftfreq(Nf, d=1.0/Nf) * (2*np.pi)  # size Nf array of [–π,π)
    kx, ky = np.meshgrid(k1d, k1d, indexing='ij')
    k_abs = np.sqrt(kx**2 + ky**2)

    # 2) Define cutoff = π·Nc
    k_cut = np.pi * Nc

    # 3) Precompute a boolean mask for |k| ≤ k_cut
    keep_mask = (k_abs <= k_cut)  # shape (Nf, Nf)

    # 4) Allocate output
    errors = np.zeros((num_samples, T_steps), dtype=np.float64)

    # 5) Loop over every (sample, time) and compute the band‐limited projection
    for i in range(num_samples):
        for t in range(T_steps):
            u_f = u_f_dataset[i, t]  # shape (Nf, Nf)

            # 5a) FFT
            u_hat = np.fft.fft2(u_f)

            # 5b) Zero out high‐freqs
            u_hat_low = np.zeros_like(u_hat)
            u_hat_low[keep_mask] = u_hat[keep_mask]

            # 5c) Inverse FFT → real
            u_low = np.real(np.fft.ifft2(u_hat_low))

            # 5d) Compute ℓ₁‐error
            errors[i, t] = np.sum(np.abs(u_f - u_low))
    per_pixel = errors / (u_f_dataset.shape[-1] * u_f_dataset.shape[-2])
    mean_per_pixel = per_pixel.mean()
    return errors, mean_per_pixel

# Example usage (after loading your dataset):
u_f_dataset = np.load("/pscratch/sd/h/hbassi/dataset/2d_heat_eqn_fine_all_smooth_gauss_1k.npy")
ideal_errors, ideal_mean = compute_ideal_l1_error(u_f_dataset, Nc=32)
print(f"Ideal ℓ₁‐error per snapshot (averaged): {ideal_mean:.6e}")
