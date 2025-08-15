import numpy as np

def compute_ideal_l1_error_ns(u_f_dataset, Nc=32):
    """
    Given u_f_dataset of shape (num_samples, T, Nf, Nf),
    compute the ℓ₁ error between u_f and its projection onto the
    same square‐wavenumber truncation used in the NS data generation.

    In the NS setup, “coarse” initialization is obtained by:
      1) FFT2 of the Nf×Nf field,
      2) fftshift to center zero‐frequency,
      3) cropping the central Nc×Nc block of coefficients,
      4) ifftshift and inverse FFT (with output size Nc×Nc).

    To mirror that, we keep exactly those coefficients with array indices
    in [start:end)×[start:end), where start = (Nf−Nc)//2, end = start+Nc.

    Returns:
      errors           of shape (num_samples, T), where
        errors[i, t] = sum_{x,y}|u_f[i,t,x,y] − u_low[i,t,x,y]|
      mean_per_pixel   = average of (errors / Nf^2) over all i,t frames
    """
    num_samples, T_steps, Nf, _ = u_f_dataset.shape
    start = (Nf - Nc) // 2
    end   = start + Nc

    errors = np.zeros((num_samples, T_steps), dtype=np.float64)

    for i in range(num_samples):
        for t in range(T_steps):
            u_f = u_f_dataset[i, t]             # shape (Nf, Nf)
            u_hat = np.fft.fft2(u_f)            # (Nf, Nf), complex
            u_hat_shift = np.fft.fftshift(u_hat)

            # zero out everything except the central Nc×Nc block
            u_hat_low_shift = np.zeros_like(u_hat_shift)
            u_hat_low_shift[start:end, start:end] = u_hat_shift[start:end, start:end]

            u_hat_low = np.fft.ifftshift(u_hat_low_shift)
            u_low = np.real(np.fft.ifft2(u_hat_low))  # shape (Nf, Nf)

            errors[i, t] = np.sum(np.abs(u_f - u_low))

    # Convert to per-pixel MAE
    per_pixel = errors / (Nf * Nf)
    mean_per_pixel = per_pixel.mean()
    return errors, mean_per_pixel

# === Example usage ===
if __name__ == "__main__":
    # Replace this path with the actual NS fine‐grid .npy file:
    nu = 1e-4
    k_cutoff_coarse = 7.5
    k_cutoff_fine = 7.5
    u_f_dataset = np.load(f'/pscratch/sd/h/hbassi/NavierStokes_fine_128_nu={nu}_kcutoff={k_cutoff_coarse}_with_forcing_no_norm_test_trajs.npy')[:, :100, : , :]
    ideal_errors, ideal_mean = compute_ideal_l1_error_ns(u_f_dataset, Nc=32)
    print(f"Ideal ℓ₁‐error per snapshot (averaged): {ideal_mean:.6e}")
