import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# -------------------------------------------------------
# Load histories and plot metrics, saving to disk
# -------------------------------------------------------
def load_histories(log_dir):
    files = {
        'loss': ('train', '2d_vlasov_mx=8my=8_FUnet_train_PS_FT_32to128_1k_t=101_loss.npy',
                 'val',   '2d_vlasov_FUnet_mx=8my=8_val_PS_FT_32to128_1k_t=101_loss.npy'),
        'mom0': ('train','2d_vlasov_FUnet_mx=8my=8_train_PS_FT_32to128_1k_t=101_mom0.npy',
                 'val',   '2d_vlasov_FUnet_mx=8my=8_val_PS_FT_32to128_1k_t=101_mom0.npy'),
        'mom1': ('train','2d_vlasov_FUnet_mx=8my=8_train_PS_FT_32to128_1k_t=101_mom1.npy',
                 'val',   '2d_vlasov_FUnet_mx=8my=8_val_PS_FT_32to128_1k_t=101_mom1.npy'),
        'mom2': ('train','2d_vlasov_FUnet_mx=8my=8_train_PS_FT_32to128_1k_t=101_mom2.npy',
                 'val',   '2d_vlasov_FUnet_mx=8my=8_val_PS_FT_32to128_1k_t=101_mom2.npy'),
    }
    data = {}
    for key, (t_label, t_file, v_label, v_file) in files.items():
        train_arr = np.load(os.path.join(log_dir, t_file))
        val_arr   = np.load(os.path.join(log_dir, v_file))
        data[key] = {'train': train_arr, 'val': val_arr}
    return data

def plot_and_save(key, train, val, ylabel, title, run_name, out_dir):
    epochs = np.arange(len(train)) * 100  # assume logged every 100 epochs
    plt.figure(figsize=(7, 4))
    plt.semilogy(epochs[:20], train[:20], label='train')
    plt.semilogy(epochs[:20], val[:20],   label='val')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # ensure output folder exists
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{key}"
    if run_name:
        fname += f"_{run_name}"
    fname += ".pdf"
    path = os.path.join(out_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved {key} plot to {path}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot Vlasov FUNet training/validation metrics"
    )
    parser.add_argument(
        "--run_name", type=str, default="",
        help="string to append to each figure filename"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs",
        help="directory containing the .npy histories"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./figures/new_vlasov/funet",
        help="where to save the generated plots"
    )
    args = parser.parse_args()

    data = load_histories(args.log_dir)

    # plot loss
    plot_and_save(
        key="loss",
        train=data['loss']['train'],
        val=data['loss']['val'],
        ylabel="L1 Loss",
        title="Training vs Validation Loss",
        run_name=args.run_name,
        out_dir=args.out_dir
    )

    # plot 0th moment error
    plot_and_save(
        key="mom0",
        train=data['mom0']['train'],
        val=data['mom0']['val'],
        ylabel="Max abs error in ρ",
        title="Training vs Validation 0th Moment Error (ρ)",
        run_name=args.run_name,
        out_dir=args.out_dir
    )

    # plot 1st moment error
    plot_and_save(
        key="mom1",
        train=data['mom1']['train'],
        val=data['mom1']['val'],
        ylabel="Max abs error in J",
        title="Training vs Validation 1st Moment Error (J)",
        run_name=args.run_name,
        out_dir=args.out_dir
    )

    # plot 2nd moment error
    plot_and_save(
        key="mom2",
        train=data['mom2']['train'],
        val=data['mom2']['val'],
        ylabel="Max abs error in M2",
        title="Training vs Validation 2nd Moment Error (M2)",
        run_name=args.run_name,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()
