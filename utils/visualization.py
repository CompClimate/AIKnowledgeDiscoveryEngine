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
import glob


def find_output_dir():
    """Find the latest output directory matching the current config."""
    config = get_config.config
    base = '/quobyte/maikesgrp/sanah/models'
    lam = config.getfloat('TRAINING', 'concept_lambda')
    ep = config.getint('TRAINING', 'epochs')
    lr = config.getfloat('OPTIMIZER.HYPERPARAMETERS', 'lr')
    bs = config.getint('DATASET', 'batch_size')
    loss = config['TRAINING']['out_loss_fn']
    name = f"lam{lam}_ep{ep}_lr{lr}_bs{bs}_{loss}"
    pattern = f"{base}/{name}*"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No output directory found matching {pattern}')
    # Return the latest (highest version)
    result = matches[-1]
    print(f'Using output directory: {result}', flush=True)
    return result


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


def plot_detailed_losses(losses_path=None, output_dir=None):
    if output_dir is None:
        output_dir = find_output_dir()
    if losses_path is None:
        losses_path = f'{output_dir}/detailed_losses.pt'
    data = torch.load(losses_path, weights_only=False)
    train_pred = data['train_pred']
    val_pred = data['val_pred']
    train_per_concept = data['train_per_concept']
    val_per_concept = data['val_per_concept']
    concept_names = list(train_per_concept.keys())

    # Compute combined loss: (1 - lambda) * pred + lambda * mean(concept losses)
    from utils.get_config import config
    concept_lambda = config.getfloat('TRAINING', 'concept_lambda')
    n_epochs = len(train_pred)
    train_combined = []
    val_combined = []
    for e in range(n_epochs):
        train_concept_mean = sum(train_per_concept[name][e] for name in concept_names) / len(concept_names)
        val_concept_mean = sum(val_per_concept[name][e] for name in concept_names) / len(concept_names)
        train_combined.append((1 - concept_lambda) * train_pred[e] + concept_lambda * train_concept_mean)
        val_combined.append((1 - concept_lambda) * val_pred[e] + concept_lambda * val_concept_mean)

    # Panel 1: combined loss, Panel 2: prediction loss, Panel 3: per-concept losses
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), layout='constrained')

    # Combined loss
    ax[0].semilogy(train_combined, label='train')
    ax[0].semilogy(val_combined, label='val')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title(f'Combined Loss (lambda={concept_lambda})')
    ax[0].legend()

    # Prediction loss
    ax[1].semilogy(train_pred, label='train')
    ax[1].semilogy(val_pred, label='val')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('BCEWithLogitsLoss')
    ax[1].set_title('Prediction Loss')
    ax[1].legend()

    # Per-concept losses (same colour per concept, solid=train, dashed=val)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, name in enumerate(concept_names):
        c = colors[i % len(colors)]
        ax[2].semilogy(train_per_concept[name], color=c, linestyle='-', label=f'{name} (train)')
        ax[2].semilogy(val_per_concept[name], color=c, linestyle='--', label=f'{name} (val)')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('MSELoss')
    ax[2].set_title('Per-Concept Loss')
    ax[2].legend(fontsize=7, ncol=2)

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
    im = axes[0].imshow(target_true.squeeze()[0, :, :], vmin=0, vmax=1, cmap=cmap, aspect='equal', origin='lower')
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
            im = axes[i+1].imshow(pred_2d, vmin=0, vmax=1, cmap=cmap, aspect='equal', origin='lower')
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
        im = axes[0].imshow(concept_true.squeeze()[j, 0, :, :], cmap='viridis', aspect='equal', origin='lower')
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
                
                im = axes[i+1].imshow(concepts_denorm.squeeze()[j, 0, :, :], cmap='viridis', aspect='equal', origin='lower')
                axes[i+1].set_title(f"Epoch {epoch}")
                if i+1 == 4:
                    cbar = plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f'{concept_names[j]}_pred_1.png', bbox_inches='tight')
        print(f'saved {concept_names[j]}')

def plot_unet_pred(model_dir=None):
    print('plotting prediction')
    from models import UNetCBM
    from utils.get_config import config, try_cast
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    if model_dir is None:
        model_dir = find_output_dir()

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

    n_epochs = config.getint('TRAINING', 'epochs')
    n_ckpt = config.getint('OUTPUT', 'n_epochs_between_checkpoints')
    epochs_to_check = list(range(0, n_epochs, n_ckpt))
    steps_mapping = [1, 3, 6]

    # One figure per lead time
    n_panels = len(epochs_to_check) + 1  # +1 for ground truth
    for time_step in range(3):
        print('starting time step')
        fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4, 3.5), layout='constrained')

        for ax in axes:
            ax.axis('off')

        # Ground truth
        gt = target_true[0, 0, time_step, :, :].cpu().numpy()
        gt_masked = np.ma.masked_where(land_mask, gt)
        axes[0].set_facecolor('white')
        axes[0].imshow(gt_masked, vmin=0, vmax=1, cmap='RdYlBu_r', aspect='equal', origin='lower')
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
            im = ax.imshow(pred_masked, vmin=0, vmax=1, cmap='RdYlBu_r', aspect='equal', origin='lower')
            ax.set_title(f"Epoch {epoch}")

        # Add shared colorbar
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cbar.set_label('P(MLHC Event)')

        current_step = steps_mapping[time_step]
        target_month = target_dates[current_step]
        fig.suptitle(f'UNet CBM Predictions: Lead Time {current_step} months (target: {target_month}, opa{member})', fontsize=14)
        save_name = f'{model_dir}/unet_pred_lead{current_step}.png'
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_name}', flush=True)

def plot_unet_concept(model_dir=None):
    from models import UNetCBM
    from utils.get_config import config, try_cast
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    if model_dir is None:
        model_dir = find_output_dir()
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

    n_epochs = config.getint('TRAINING', 'epochs')
    n_ckpt = config.getint('OUTPUT', 'n_epochs_between_checkpoints')
    epochs_to_check = list(range(0, n_epochs, n_ckpt))
    steps_mapping = [1, 3, 6]

    # One figure per lead time per concept
    n_panels = len(epochs_to_check) + 1  # +1 for ground truth
    for ci in range(len(concepts)):
        for time_step in range(3):
            print(f'starting {concepts[ci]} time step {time_step}', flush=True)
            fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4, 3.5), layout='constrained')

            for ax in axes:
                ax.axis('off')

            # Ground truth concept ci (already denormalized)
            gt = concept_true[0, ci, time_step, :, :].cpu().numpy()
            gt_masked = np.ma.masked_where(land_mask, gt)
            clean_arr = np.ma.filled(gt_masked.astype(float), np.nan)
            vmin, vmax = np.nanpercentile(clean_arr, [2, 98])

            axes[0].set_facecolor('white')
            axes[0].imshow(gt_masked, cmap='RdYlBu_r', aspect='equal', vmin=vmin, vmax=vmax, origin='lower')
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
                    concept_denorm = concept_norm.denormalize(concept.cpu())
                    pred_2d = concept_denorm[0, ci, time_step, :, :].numpy()

                ax = axes[ei + 1]
                ax.set_facecolor('white')
                ax.axis('on')
                pred_masked = np.ma.masked_where(land_mask, pred_2d)
                im = ax.imshow(pred_masked, cmap='RdYlBu_r', aspect='equal', vmin=vmin, vmax=vmax, origin='lower')
                ax.set_title(f"Epoch {epoch}")

            # Add shared colorbar
            cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)

            current_step = steps_mapping[time_step]
            target_month = target_dates[current_step]
            fig.suptitle(f'UNet CBM {concepts[ci].upper()} Prediction: Lead Time {current_step} months (target: {target_month}, opa{member})', fontsize=14)
            save_name = f'{model_dir}/unet_{concepts[ci]}_lead{current_step}.png'
            fig.savefig(save_name, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_name}', flush=True)

def eval_gt_concepts(model_dir=None):
    """Compare prediction using predicted concepts vs ground-truth concepts."""
    from models import UNetCBM
    from utils.get_config import config, try_cast
    from utils.get_data import get_dataset_preload
    if model_dir is None:
        model_dir = find_output_dir()

    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    mask_tensor = torch.tensor(mask_2d, dtype=torch.float32)[None, None, None, :, :]

    input_norm, concept_norm, _, _, val_loader, _ = get_dataset_preload()

    n_concepts = len(try_cast(config['DATASET']['concepts']))
    output_dim = len(try_cast(config['DATASET']['offset']))
    out_loss_fn = getattr(torch.nn, config['TRAINING']['out_loss_fn'])()

    model = UNetCBM(
        n_features=len(try_cast(config['DATASET']['features'])) * config.getint('DATASET', 'context_window'),
        n_concepts=n_concepts,
        output_dim=output_dim,
    )

    n_epochs = config.getint('TRAINING', 'epochs')
    n_ckpt = config.getint('OUTPUT', 'n_epochs_between_checkpoints')
    epochs_to_check = list(range(0, n_epochs, n_ckpt))
    pred_losses = []
    gt_losses = []

    for epoch in epochs_to_check:
        checkpoint_path = f'{model_dir}/unet_epoch{epoch}.pt'
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}', flush=True)
            continue
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        pred_loss_accum = 0
        gt_loss_accum = 0
        n_snaps = 0

        with torch.no_grad():
            for batch, concept_y, y in val_loader:
                batch = torch.nan_to_num(input_norm.normalize(batch), nan=0.0)
                concept_y = torch.nan_to_num(concept_norm.normalize(concept_y), nan=0.0)
                y = torch.nan_to_num(y, nan=0.0)

                # Normal forward pass (predicted concepts)
                pred, _ = model(batch)
                pred = pred * mask_tensor
                pred_loss_accum += out_loss_fn(pred, y).item()

                # Ground-truth concepts through output_head
                B, C, T, Y, X = concept_y.shape
                gt_flat = concept_y.view(B, C * T, Y, X)  # (B, n_concepts*output_dim, Y, X)
                gt_output = model.output_head(gt_flat)
                gt_output = gt_output.unsqueeze(1) * mask_tensor  # (B, 1, output_dim, Y, X)
                gt_loss_accum += out_loss_fn(gt_output, y).item()

                n_snaps += 1

        pred_losses.append(pred_loss_accum / n_snaps)
        gt_losses.append(gt_loss_accum / n_snaps)
        print(f'Epoch {epoch}: pred_concepts={pred_losses[-1]:.5f}, gt_concepts={gt_losses[-1]:.5f}')

    # Plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout='constrained')
    ax.plot(epochs_to_check, pred_losses, 'o-', label='Predicted concepts')
    ax.plot(epochs_to_check, gt_losses, 's--', label='Ground-truth concepts')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCEWithLogitsLoss')
    ax.set_title('Prediction Loss: Predicted vs Ground-Truth Concepts')
    ax.legend()
    save_name = f'{model_dir}/gt_concept_comparison.png'
    fig.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_name}')


def plot_concept_timeseries(output_dir=None, member='opa0'):
    if output_dir is None:
        output_dir = find_output_dir()
    """Plot time series of spatial mean/std and distributions for each concept."""
    from utils.get_config import config, try_cast
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    loc = config['DATASET']['location']
    concepts = try_cast(config['DATASET']['concepts'])
    start_date = datetime.strptime(config['DATASET']['start'], "%Y%m")
    end_date = datetime.strptime(config['DATASET']['end'], "%Y%m")

    # Build date axis
    dates = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur)
        cur += relativedelta(months=1)

    # Load land mask
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    ocean_mask = (mask_2d == 1)

    # Load each concept and compute stats
    concept_data = {}
    for concept in concepts:
        ds = xr.open_zarr(f'{loc}/{member}/{concept}_na.zarr')
        ds = ds.rename({'time_counter': 'time'})
        arr = ds.sel(y=slice(0, 302), x=slice(0, 400)).to_array().values.squeeze(0)  # (time, y, x)
        # Mask land
        arr[:, ~ocean_mask] = np.nan
        concept_data[concept] = arr
        print(f'Loaded {concept}: shape={arr.shape}, range=[{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]')

    # Apply transforms: log10 for vori, 98th percentile clip for vohfe, von2, vos2
    transform_labels = {}
    if 'vori' in concept_data:
        arr = concept_data['vori']
        arr = np.where(arr > 0, arr, np.nan)
        concept_data['vori'] = np.log10(arr)
        transform_labels['vori'] = 'log10(vori)'
        print(f'Transformed vori: log10, range=[{np.nanmin(concept_data["vori"]):.3g}, {np.nanmax(concept_data["vori"]):.3g}]')
    for name in ['vohfe', 'von2', 'vos2']:
        if name in concept_data:
            arr = concept_data[name]
            p2 = np.nanpercentile(arr, 2)
            p98 = np.nanpercentile(arr, 98)
            concept_data[name] = np.clip(arr, p2, p98)
            transform_labels[name] = f'{name} (clipped [{p2:.3g}, {p98:.3g}])'
            print(f'Clipped {name} at [2nd, 98th] pct=[{p2:.3g}, {p98:.3g}]')

    # Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    smooth_concepts = try_cast(config.get('DATASET', 'smooth_concepts', fallback='[]'))
    smooth_sigma = config.getfloat('DATASET', 'smooth_sigma', fallback=0)
    for name in smooth_concepts:
        if name in concept_data and smooth_sigma > 0:
            arr = concept_data[name]
            for t in range(arr.shape[0]):
                nan_mask = np.isnan(arr[t])
                arr[t][nan_mask] = 0.0
                arr[t] = gaussian_filter(arr[t], sigma=smooth_sigma)
                arr[t][nan_mask] = np.nan
            label = transform_labels.get(name, name)
            transform_labels[name] = f'{label} (smooth σ={smooth_sigma})'
            print(f'Smoothed {name} with sigma={smooth_sigma}')

    n_concepts = len(concepts)
    n_times = min(len(dates), next(iter(concept_data.values())).shape[0])
    dates = dates[:n_times]

    # --- Figure 1: Spatial mean time series (each concept gets its own y-axis scale) ---
    fig, axes = plt.subplots(n_concepts, 1, figsize=(14, 2.5 * n_concepts), sharex=True, layout='constrained')
    for i, concept in enumerate(concepts):
        arr = concept_data[concept][:n_times]
        spatial_mean = np.nanmean(arr, axis=(1, 2))
        spatial_p5 = np.nanpercentile(arr, 5, axis=(1, 2))
        spatial_p95 = np.nanpercentile(arr, 95, axis=(1, 2))
        axes[i].plot(dates, spatial_mean, linewidth=1, label='mean')
        axes[i].fill_between(dates, spatial_p5, spatial_p95, alpha=0.2, label='5th-95th pct')
        axes[i].set_ylabel(transform_labels.get(concept, concept))
        axes[i].legend(loc='upper right', fontsize=7)
    axes[-1].set_xlabel('Date')
    fig.suptitle(f'Concept Spatial Mean + Spread Over Time ({member})', fontsize=14)
    save_name = f'{output_dir}/concept_timeseries_{member}.png'
    fig.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_name}')

    # --- Figure 2: Spatial std time series (all on same plot for comparison) ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 5), layout='constrained')
    for concept in concepts:
        arr = concept_data[concept][:n_times]
        spatial_std = np.nanstd(arr, axis=(1, 2))
        ax.plot(dates, spatial_std, linewidth=1, label=concept)
    ax.set_xlabel('Date')
    ax.set_ylabel('Spatial Std Dev')
    ax.set_title(f'Concept Spatial Variability Over Time ({member})')
    ax.legend()
    save_name = f'{output_dir}/concept_spatial_std_{member}.png'
    fig.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_name}')

    # --- Figure 3: Value distributions (histograms) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    axes = axes.flatten()
    for i, concept in enumerate(concepts):
        arr = concept_data[concept][:n_times]
        vals = arr[~np.isnan(arr)].flatten()
        # Subsample if too many points for histogram
        if len(vals) > 500000:
            vals = np.random.choice(vals, 500000, replace=False)
        axes[i].hist(vals, bins=100, edgecolor='none', alpha=0.8)
        axes[i].set_title(transform_labels.get(concept, concept))
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
        axes[i].axvline(np.mean(vals), color='red', linestyle='--', linewidth=1, label=f'mean={np.mean(vals):.2g}')
        axes[i].legend(fontsize=7)
    fig.suptitle(f'Concept Value Distributions ({member})', fontsize=14)
    save_name = f'{output_dir}/concept_distributions_{member}.png'
    fig.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_name}')


if __name__ == "__main__":

    # eval_gt_concepts()
    plot_unet_pred()
    plot_unet_concept()
    #plot_detailed_losses()
    #plot_concept_timeseries()