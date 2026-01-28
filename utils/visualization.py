import matplotlib.pyplot as plt
from utils.get_config import config

def visualize(train_losses, val_losses, plot_test):
    #plot_test = config['VISUALIZATION']['plot_test']
    output = config['OUTPUT']['dir']
    if plot_test:
        n_panels = 2 #4
    else:
        n_panels = 2 #3
    fig, ax = plt.subplots(1, n_panels, figsize=(9/4 * n_panels, 8/3), layout='constrained')
    ax[0].semilogy(train_losses, 'tab:blue', label="train")
    ax[0].semilogy(val_losses, 'tab:orange', label="val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss (MSE)")
    ax[0].legend()
    ax[0].annotate(f"train: {train_losses[-1]:0.2e}" + "\n" +
                       f"val: {val_losses[-1]:0.2e}",
                       xy=(0.2, .75),
                       xycoords='axes fraction')
    fig.savefig(f'{output}/viz.png', dpi=400)