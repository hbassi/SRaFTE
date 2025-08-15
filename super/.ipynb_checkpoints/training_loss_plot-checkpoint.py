import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load training losses
ending = [8, 4, 2]
colors = ['C0', 'C1', 'C2']
#training_losses = np.load('./logs/2d_wave_FUnet_train_PS_FT_32to128_v1_low_sf=4_epochs=10000.npy')
#validation_losses = np.load('./logs/2d_wave_FUnet_val_PS_FT_32to128_v1_low_sf=4_epochs=10000.npy')
#plt.semilogy(training_losses[:] )
#plt.semilogy(validation_losses[:])
#plt.semilogy([0, 14], [2.1846e-05, 2.1846e-05], 'k:')
for i, end in enumerate(ending):
    training_losses = np.load(f'./logs/2d_wave_FUnet_train_PS_FT_32to128_v1_high_sf={end}.npy')
    validation_losses = np.load(f'./logs/2d_wave_FUnet_val_PS_FT_32to128_v1_high_sf={end}.npy')
    print(training_losses)
    plt.semilogy(training_losses[:], color = colors[i])
    plt.semilogy(validation_losses[:], '--', color = colors[i])

plt.legend([r'training ($8\times$)', r'validation ($8\times$)', r'training ($4\times$)', r'validation ($4\times$)',r'training ($2\times$)', r'validation ($2\times$)' ])
#plt.legend(['training', 'validation'])
# Customize x-axis labels to be multiplied by 100
def multiply_by_100(x, _):
    return f"{int(x * 100)}"

plt.gca().xaxis.set_major_formatter(FuncFormatter(multiply_by_100))

# Add labels and save the plot
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.savefig('./figures/funet_2d_wave_scaling_test_multi_traj_training_validation_losses_high_total.pdf')
plt.show()