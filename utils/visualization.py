import matplotlib.pyplot as plt
from utils.get_config import config

def visualize(train_losses, val_losses, plot_test):
    #plot_test = config['VISUALIZATION']['plot_test']
    if plot_test:
        n_panels = 3 #4
    else:
        n_panels = 4 #3
    fig, ax = plt.subplots(1, n_panels, figsize=(9/4 * n_panels, 8/3), layout='constrained')
    ax[0].semilogy(train_losses[0], 'tab:blue', label="train")
    ax[0].semilogy(val_losses[0], 'tab:orange', label="val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss (MSE)")
    ax[0].legend()
    ax[0].annotate(f"train: {train_losses[0][-1]:0.2e}" + "\n" +
                       f"val: {val_losses[0][-1]:0.2e}",
                       xy=(0.2, .75),
                       xycoords='axes fraction')
    ax[1].semilogy(train_losses[1], 'tab:blue', label="train")
    ax[1].semilogy(val_losses[1], 'tab:orange', label="val")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("prediction loss (MSE)")
    ax[1].legend()
    ax[1].annotate(f"train: {train_losses[1][-1]:0.2e}" + "\n" +
                       f"val: {val_losses[1][-1]:0.2e}",
                       xy=(0.2, .75),
                       xycoords='axes fraction')
    ax[2].semilogy(train_losses[2], 'tab:blue', label="train")
    ax[2].semilogy(val_losses[2], 'tab:orange', label="val")
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("concept loss (MSE)")
    ax[2].legend()
    ax[2].annotate(f"train: {train_losses[2][-1]:0.2e}" + "\n" +
                       f"val: {val_losses[2][-1]:0.2e}",
                       xy=(0.2, .75),
                       xycoords='axes fraction')
    output = config['OUTPUT']['dir']
    fig.savefig(f'{output}/viz.png', dpi=400)