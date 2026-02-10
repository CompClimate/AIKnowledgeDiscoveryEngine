import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import os
import xarray as xr
import numpy as np

# 1. Import get_config
import utils.get_config as get_config

# 2. Define path and FORCE the read immediately
home_dir = os.path.expanduser("~")
config_path = os.path.join(home_dir, 'AIKnowledgeDiscoveryEngine/utils/config.ini')

if os.path.exists(config_path):
    get_config.config.read(config_path)
    print(f"Config loaded successfully from {config_path}", flush=True)
else:
    print(f"ERROR: Config not found at {config_path}", flush=True)

# 3. NOW import the dataset (it will now see the populated config)
from utils.load_data import EmulatorDataset
from torch.utils.data import Subset, DataLoader


def visualize(train_losses, val_losses, plot_test):
    #plot_test = config['VISUALIZATION']['plot_test']
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
    output = get_config.config['OUTPUT']['dir']
    fig.savefig(f'{output}/viz.png', dpi=400)


def plot_detailed_losses(losses_path='/quobyte/maikesgrp/sanah/models/detailed_losses.pt', output_dir='/quobyte/maikesgrp/sanah/models'):
    data = torch.load(losses_path, weights_only=False)
    train_pred = data['train_pred']
    val_pred = data['val_pred']
    train_per_concept = data['train_per_concept']
    val_per_concept = data['val_per_concept']
    concept_names = list(train_per_concept.keys())

    # Panel 1: prediction loss, Panel 2: per-concept losses
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')

    # Prediction loss
    ax[0].semilogy(train_pred, label='train')
    ax[0].semilogy(val_pred, label='val')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('BCEWithLogitsLoss')
    ax[0].set_title('Prediction Loss')
    ax[0].legend()

    # Per-concept losses (same colour per concept, solid=train, dashed=val)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, name in enumerate(concept_names):
        c = colors[i % len(colors)]
        ax[1].semilogy(train_per_concept[name], color=c, linestyle='-', label=f'{name} (train)')
        ax[1].semilogy(val_per_concept[name], color=c, linestyle='--', label=f'{name} (val)')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MSELoss')
    ax[1].set_title('Per-Concept Loss')
    ax[1].legend(fontsize=7, ncol=2)

    fig.savefig(f'{output_dir}/detailed_losses.png', dpi=300)
    plt.close(fig)
    print(f'Saved {output_dir}/detailed_losses.png')


def pairity_for_target():
    home_dir = os.path.expanduser("~")
    print('Starting parity visualization...', flush=True)
    
    # 1. Load normalization stats
    stats_path = os.path.join(home_dir, 'normalization_stats.pt')
    norms = torch.load(stats_path, weights_only=False)
    input_norm = norms['input']

    # 2. Setup Dataset and Loader
    dataset = EmulatorDataset()
    n = len(dataset)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)
    val_idx = list(range(train_end, val_end))
    val_set = Subset(dataset, val_idx)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_idx = list(range(val_end, n))
    test_set = Subset(dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=1)
    epochs_to_check = [0, 5, 10, 15]
    steps_mapping = [1, 3, 6]

    # Loop through lead times (time_step 0, 1, 2)
    for time_step in range(3):
        # MOVE subplots inside here so each file (1, 3, 6) starts with a clean figure
        fig, axes = plt.subplots(1, len(epochs_to_check), figsize=(20, 5))
        
        for i, epoch in enumerate(epochs_to_check):
            # Load specific epoch model
            model = get_config.get_model()
            model_path = f'/home/kkringel/temp/20260202_testingMLP/bs_1/PointwiseCBM_epoch{epoch}.pt'
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            all_logits = [] 
            all_trues = []

            with torch.no_grad():
                for data, concept_true, target_true in test_loader:
                    # Model returns (target_logits, concept_logits)
                    pred_logits, _ = model(data) 
                    
                    # Squeeze and slice for specific lead time
                    # Shape becomes [302, 400]
                    logits_2d = pred_logits[0, 0, time_step, :, :].squeeze()
                    true_2d = target_true[0, 0, time_step, :, :].squeeze()

                    # Masking NaNs based on ground truth
                    mask = ~torch.isnan(true_2d.flatten())

                    
                    all_logits.append(logits_2d.flatten()[mask])
                    all_trues.append(true_2d.flatten()[mask])

            # Concatenate all validation pixels for this epoch
            epoch_logits = torch.cat(all_logits).cpu().numpy()
            epoch_trues = torch.cat(all_trues).cpu().numpy()
            print('max logit: ', np.nanmax(epoch_logits))
            print('min logit: ', np.nanmin(epoch_logits))
            if time_step == 0:
                print(f"Epoch {epoch}: num points = {len(epoch_trues)}", flush=True)


            # --- Inside your loop after epoch_logits and epoch_trues are defined ---

            # 1. Hit Rate (Recall): "Of the real events, how many did we find?"
            actual_ones = (epoch_trues == 1)
            true_positives = (epoch_logits[actual_ones] > 0).sum()
            hit_rate = true_positives / actual_ones.sum()

            # 2. Precision: "Of our 'On' predictions, how many were actually real?"
            predicted_ones = (epoch_logits > 0)
            total_predicted_on = predicted_ones.sum()

            if total_predicted_on > 0:
                precision = (epoch_trues[predicted_ones] == 1).sum() / total_predicted_on
            else:
                precision = 0.0

            ax = axes[i]

            ax.set_title(f'Epoch {epoch}\nHit Rate: {hit_rate:.1%} | Precision: {precision:.1%}')

            # --- Plotting ---
            # Jitter helps visualize the density of points on the binary X-axis
            ax.hist(epoch_logits[epoch_trues == 0], bins=50, alpha=0.6, label='No Event (0)', color='blue')
            ax.hist(epoch_logits[epoch_trues == 1], bins=50, alpha=0.6, label='Event (1)', color='red')
            ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Threshold')
            ax.set_xlabel('Model Logit (Score)')
            ax.set_ylabel('Count')
            ax.legend()
            # jitter = np.random.normal(0, 0.05, size=epoch_trues.shape)
            # if i == 0:
            #     print('true shape: ', epoch_trues.shape)
            #     print('logit shape: ', epoch_logits.shape)
            # ax.scatter(
            #     epoch_trues, 
            #     epoch_logits, 
            #     s=1, 
            #     alpha=0.1, 
            #     color='teal'
            #     #rasterized=True # Recommended for cluster runs to keep file size small
            # )

            # ax.axhline(0, color='red', linestyle='--', label='Threshold')
            
            #ax.set_title(f'Epoch {epoch}\nHit Rate: {hit_rate:.1%}')
            #ax.set_ylabel('Model Logit (Score)')
            #ax.set_ylim(-1, 1) # Based on your previous screenshots
            #ax.set_xticks([0, 1])
            #ax.set_xticklabels(['No Event (0)', 'Event (1)'])

        # Final figure cleanup
        current_step = steps_mapping[time_step]
        fig.suptitle(f'Parity Plot: Time Step {current_step}', fontsize=16)
        fig.tight_layout()
        
        save_name = f'pairity_{current_step}_test.png'
        print(f'Saving figure: {save_name}', flush=True)
        fig.savefig(save_name, dpi=300)
        plt.close(fig) # Free up memory
    
def plot_val():
    # 1. Load the stats you are generating RIGHT NOW
    home_dir = os.path.expanduser("~")
    stats_path = os.path.join(home_dir, 'normalization_stats.pt')
    norms = torch.load(stats_path, weights_only=False)

    input_norm = norms['input']
    concept_norm = norms['concept']
    #output_norm = norms['output']

    # 2. Setup the Validation Set (No stats loop needed!)
    dataset = EmulatorDataset()
    n = len(dataset)
    # Same split logic to get the same Val set
    train_end = int(0.8 * n) # Adjust if your train_frac is different
    val_end = train_end + int(0.1 * n) # Adjust if your test_frac is different
    val_idx = list(range(train_end, val_end))

    val_set = Subset(dataset, val_idx)
    # shuffle=False is vital for visualization consistency
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    data, concept_true, target_true = next(iter(val_loader))

    epochs_to_check = [0, 5, 10, 15] 
    fig, axes = plt.subplots(1, len(epochs_to_check) + 1, figsize=(20, 5))
    cmap = ListedColormap(['#1f77b4', '#ff7f0e'])
    im = axes[0].imshow(target_true.squeeze()[0, :, :], vmin=0, vmax=1, cmap=cmap, aspect='equal')
    axes[0].set_title("Ground Truth")

    for i, epoch in enumerate(epochs_to_check):
        
        model = get_config.get_model()
        model_path = f'/quobyte/maikesgrp/sanah/model_test/unet_epoch{epoch}.pt'
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            norm_data = input_norm.normalize(data)
            output, concepts = model(norm_data)
            output_denorm = output
            
            binary_preds = torch.where(
                torch.isnan(output_denorm), 
                output_denorm,               # Keeps the NaN if it was a NaN
                (output_denorm >= 0).float()  # Otherwise returns 1.0 or 0.0
            )
            pred_2d = binary_preds[0, 0, 0, :, :].cpu().numpy()
            print('binary: ', pred_2d)
            im = axes[i+1].imshow(pred_2d, vmin=0, vmax=1, cmap=cmap, aspect='equal')
            axes[i+1].set_title(f"Epoch {epoch}")
            if i+1 == 4:
                cbar = plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04, ticks=[0, 1])
                cbar.ax.set_yticklabels(['No MLHC Event', 'MLHC Event']) 
    plt.tight_layout()
    plt.savefig('pred_test_binary_update.png', bbox_inches='tight')
    
def plot_concept_pred():
    # 1. Load the stats you are generating RIGHT NOW
    home_dir = os.path.expanduser("~")
    stats_path = os.path.join(home_dir, 'normalization_stats.pt')
    norms = torch.load(stats_path, weights_only=False)

    input_norm = norms['input']
    concept_norm = norms['concept']
    output_norm = norms['output']

    # 2. Setup the Validation Set (No stats loop needed!)
    dataset = EmulatorDataset()
    n = len(dataset)
    # Same split logic to get the same Val set
    train_end = int(0.8 * n) # Adjust if your train_frac is different
    val_end = train_end + int(0.1 * n) # Adjust if your test_frac is different
    val_idx = list(range(train_end, val_end))

    val_set = Subset(dataset, val_idx)
    # shuffle=False is vital for visualization consistency
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    concept_names = ['sowsc', 'voep', 'vori', 'von2', 'vos2', 'vohfe']
    for j in range(6):
        # concept_true: [1, 6, 3, 302, 400]
        data, concept_true, target_true = next(iter(val_loader))
        epochs_to_check = [0, 5, 10, 15] 
        fig, axes = plt.subplots(1, len(epochs_to_check) + 1, figsize=(20, 5))
        im = axes[0].imshow(concept_true.squeeze()[j, 0, :, :], cmap='viridis', aspect='equal')
        axes[0].set_title("Ground Truth")
        fig.suptitle(f'Concept: {concept_names[j]} at time step 1')

        for i, epoch in enumerate(epochs_to_check):
            
            model = get_config.get_model()
            model_path = f'/home/kkringel/temp/20260202_testingMLP/bs_1/PointwiseCBM_epoch{epoch}.pt'
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            with torch.no_grad():
                norm_data = input_norm.normalize(data)
                # concepts: [1, 6, 3, 302, 400]
                output, concepts = model(norm_data)
                concepts_denorm = concept_norm.denormalize(concepts)
                
                im = axes[i+1].imshow(concepts_denorm.squeeze()[j, 0, :, :], cmap='viridis', aspect='equal')
                axes[i+1].set_title(f"Epoch {epoch}")
                if i+1 == 4:
                    cbar = plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f'{concept_names[j]}_pred_1.png', bbox_inches='tight')
        print(f'saved {concept_names[j]}')

def plot_unet_pred():
    from models import UNetCBM
    from utils.get_config import config, try_cast
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    # Load land mask
    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    land_mask = (mask_2d == 0)

    # Setup validation set using preloaded data
    from utils.get_data import get_dataset_preload
    input_norm, concept_norm, _, _, val_loader, _ = get_dataset_preload()
    data, concept_true, target_true = next(iter(val_loader))
    # Take second sample for visualization
    val_sample_idx = 1
    data, concept_true, target_true = data[val_sample_idx:val_sample_idx+1], concept_true[val_sample_idx:val_sample_idx+1], target_true[val_sample_idx:val_sample_idx+1]

    # Compute target date for this sample
    window = config.getint('DATASET', 'context_window')
    offsets = try_cast(config['DATASET']['offset'])
    n_members = len(try_cast(config['DATASET']['members']))
    start_date = datetime.strptime(config['DATASET']['start'], "%Y%m")
    end_date = datetime.strptime(config['DATASET']['end'], "%Y%m")
    dates = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur.strftime("%Y%m"))
        cur += relativedelta(months=1)
    n_times = len(dates) - window - max(offsets) + 1
    train_time_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n_times)
    train_end = train_time_end * n_members
    original_idx = train_end + val_sample_idx
    member = original_idx % n_members
    time_idx = original_idx // n_members
    target_dates = {off: dates[time_idx + window - 1 + off] for off in offsets}
    print(f'Val sample: original idx={original_idx}, member=opa{member}, target dates={target_dates}')

    print('set up validation')

    # Build UNetCBM with same config as training
    model = UNetCBM(
        n_features=len(try_cast(config['DATASET']['features'])) * config.getint('DATASET', 'context_window'),
        n_concepts=len(try_cast(config['DATASET']['concepts'])),
        output_dim=len(try_cast(config['DATASET']['offset']))
    )

    print('initialized model')

    model_dir = '/quobyte/maikesgrp/sanah/model_test'
    epochs_to_check = list(range(0, 100, 5))  # [0, 5, 10, ..., 95]
    steps_mapping = [1, 3, 6]

    # One figure per lead time
    for time_step in range(3):
        print('starting time step')
        n_cols = len(epochs_to_check) + 1  # +1 for ground truth
        n_rows = 4
        cols_per_row = (n_cols + n_rows - 1) // n_rows  # ceil division
        fig, axes = plt.subplots(3, 7, figsize=(cols_per_row * 4, n_rows * 3.5), layout='constrained')
        axes = axes.flatten()

        # Hide unused axes
        for ax in axes:
            ax.axis('off')

        # Ground truth
        gt = target_true[0, 0, time_step, :, :].cpu().numpy()
        gt_masked = np.ma.masked_where(land_mask, gt)
        axes[0].set_facecolor('white')
        axes[0].imshow(gt_masked, vmin=0, vmax=1, cmap='RdYlBu_r', aspect='equal')
        axes[0].set_title("Ground Truth")
        axes[0].axis('on')

        for i, epoch in enumerate(epochs_to_check):
            checkpoint_path = f'{model_dir}/unet_epoch{epoch}.pt'
            if not os.path.exists(checkpoint_path):
                print(f'Checkpoint not found: {checkpoint_path}', flush=True)
                continue
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            with torch.no_grad():
                norm_data = input_norm.normalize(data)
                output, _ = model(torch.nan_to_num(norm_data, nan=0.0))
                pred_2d = torch.sigmoid(output[0, 0, time_step, :, :]).cpu().numpy()

            ax = axes[i + 1]
            ax.set_facecolor('white')
            ax.axis('on')
            pred_masked = np.ma.masked_where(land_mask, pred_2d)
            im = ax.imshow(pred_masked, vmin=0, vmax=1, cmap='RdYlBu_r', aspect='equal')
            ax.set_title(f"Epoch {epoch}")

        # Add shared colorbar
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cbar.set_label('P(MLHC Event)')

        current_step = steps_mapping[time_step]
        target_month = target_dates[current_step]
        fig.suptitle(f'UNet CBM Predictions: Lead Time {current_step} months (target: {target_month}, opa{member})', fontsize=16)
        save_name = f'./unet_pred_lead{current_step}.png'
        fig.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_name}', flush=True)

def plot_unet_concept():
    from models import UNetCBM
    from utils.get_config import config, try_cast
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    concepts = ['sowsc', 'voep', 'vori', 'von2', 'vos2', 'vohfe']

    # Load land mask
    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    land_mask = (mask_2d == 0)

    # Setup validation set using preloaded data
    from utils.get_data import get_dataset_preload
    input_norm, concept_norm, _, _, val_loader, _ = get_dataset_preload()
    data, concept_true, target_true = next(iter(val_loader))
    # Take second sample for visualization
    val_sample_idx = 1
    data, concept_true, target_true = data[val_sample_idx:val_sample_idx+1], concept_true[val_sample_idx:val_sample_idx+1], target_true[val_sample_idx:val_sample_idx+1]

    # Compute target date for this sample
    window = config.getint('DATASET', 'context_window')
    offsets = try_cast(config['DATASET']['offset'])
    n_members = len(try_cast(config['DATASET']['members']))
    start_date = datetime.strptime(config['DATASET']['start'], "%Y%m")
    end_date = datetime.strptime(config['DATASET']['end'], "%Y%m")
    dates = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur.strftime("%Y%m"))
        cur += relativedelta(months=1)
    n_times = len(dates) - window - max(offsets) + 1
    train_time_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n_times)
    train_end = train_time_end * n_members
    original_idx = train_end + val_sample_idx
    member = original_idx % n_members
    time_idx = original_idx // n_members
    target_dates = {off: dates[time_idx + window - 1 + off] for off in offsets}
    print(f'Val sample: original idx={original_idx}, member=opa{member}, target dates={target_dates}')

    print('set up validation')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build UNetCBM with same config as training
    model = UNetCBM(
        n_features=len(try_cast(config['DATASET']['features'])) * config.getint('DATASET', 'context_window'),
        n_concepts=len(try_cast(config['DATASET']['concepts'])),
        output_dim=len(try_cast(config['DATASET']['offset']))
    )
    model.to(DEVICE)

    print(f'initialized model on {DEVICE}')

    # Precompute: normalize + nan_to_num + move to GPU once
    data_gpu = torch.nan_to_num(input_norm.normalize(data), nan=0.0).to(DEVICE)
    # concept_true_denorm = concept_norm.denormalize(concept_true)

    model_dir = '/quobyte/maikesgrp/sanah/model_test'
    epochs_to_check = list(range(0, 100, 5))  # [0, 5, 10, ..., 95]
    steps_mapping = [1, 3, 6]

    # One figure per lead time per concept
    for ci in range(len(concepts)):
        for time_step in range(3):
            print(f'starting {concepts[ci]} time step {time_step}', flush=True)
            n_cols = len(epochs_to_check) + 1  # +1 for ground truth
            n_rows = 4
            cols_per_row = (n_cols + n_rows - 1) // n_rows  # ceil division
            fig, axes = plt.subplots(3, 7, figsize=(cols_per_row * 4, n_rows * 3.5), layout='constrained')
            axes = axes.flatten()

            # Hide unused axes
            for ax in axes:
                ax.axis('off')

            # Ground truth concept ci (already denormalized)
            gt = concept_true[0, ci, time_step, :, :].cpu().numpy()
            gt_masked = np.ma.masked_where(land_mask, gt)
            clean_arr = np.ma.filled(gt_masked.astype(float), np.nan)
            vmin, vmax = np.nanpercentile(clean_arr, [2, 98])
            print('vmin: ', vmin)
            print('vmax: ', vmax)
            #breakpoint()

            axes[0].set_facecolor('white')
            axes[0].imshow(gt_masked, cmap='RdYlBu_r', aspect='equal', vmin=vmin, vmax=vmax)
            axes[0].set_title("Ground Truth")
            axes[0].axis('on')

            for ei, epoch in enumerate(epochs_to_check):
                checkpoint_path = f'{model_dir}/unet_epoch{epoch}.pt'
                if not os.path.exists(checkpoint_path):
                    print(f'Checkpoint not found: {checkpoint_path}', flush=True)
                    continue
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                with torch.no_grad():
                    _, concept = model(data_gpu)
                    #breakpoint()
                    concept_denorm = concept_norm.denormalize(concept.cpu())
                    pred_2d = concept_denorm[0, ci, time_step, :, :].numpy()

                ax = axes[ei + 1]
                ax.set_facecolor('white')
                ax.axis('on')
                pred_masked = np.ma.masked_where(land_mask, pred_2d)
                im = ax.imshow(pred_masked, cmap='RdYlBu_r', aspect='equal', vmin=vmin, vmax=vmax)
                ax.set_title(f"Epoch {epoch}")

            # Add shared colorbar
            cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)

            current_step = steps_mapping[time_step]
            target_month = target_dates[current_step]
            fig.suptitle(f'UNet CBM {concepts[ci].upper()} Prediction: Lead Time {current_step} months (target: {target_month}, opa{member})', fontsize=16)
            save_name = f'/quobyte/maikesgrp/sanah/unet_{concepts[ci]}_lead{current_step}.png'
            fig.savefig(save_name, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_name}', flush=True)

if __name__ == "__main__":

    plot_unet_pred()
    plot_unet_concept()