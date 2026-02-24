import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import torch
import os
import xarray as xr
import numpy as np
import glob
from utils.get_config import config, try_cast, get_model
from utils.get_data import get_dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta

def find_output_dir():
    """Find the latest output directory matching the current config."""
    base = config['OUTPUT']['dir']
    lam = config.getfloat('TRAINING', 'concept_lambda')
    ep = config.getint('TRAINING', 'epochs')
    lr = config.getfloat('OPTIMIZER.HYPERPARAMETERS', 'lr')
    bs = config.getint('DATASET', 'batch_size')
    loss = config['TRAINING']['out_loss_fn']
    norm = config.get('TRAINING', 'norm_type', fallback='MinMax')
    model_type = config['MODEL']['type']
    name = f"{model_type}_lam{lam}_ep{ep}_lr{lr}_bs{bs}_{loss}_{norm}"
    pattern = f"{base}/{name}*"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No output directory found matching {pattern}')
    # Return the latest (highest version)
    result = matches[-1]
    print(f'Using output directory: {result}', flush=True)
    return result


def visualize():
    output_dir = find_output_dir()
    losses_path = f'{output_dir}/detailed_losses.pt'

    data = torch.load(losses_path, weights_only=False)
    train_loss = data['loss']
    val_loss = data['val_loss']
    train_pred = data['train_pred']
    val_pred = data['val_pred']
    train_per_concept = data['train_per_concept']
    val_per_concept = data['val_per_concept']
    concept_names = list(train_per_concept.keys())
    has_test = 'test_loss' in data

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')

    # Combined loss
    axes[0].semilogy(train_loss, 'tab:blue', label="train")
    axes[0].semilogy(val_loss, 'tab:orange', label="val")
    if has_test:
        axes[0].axhline(data['test_loss'], color='tab:green', linestyle=':', label=f"test: {data['test_loss']:.2e}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title('Combined Loss')
    axes[0].legend()
    axes[0].annotate(f"train: {train_loss[-1]:.2e}\nval: {val_loss[-1]:.2e}",
                     xy=(0.55, 0.8), xycoords='axes fraction', fontsize=9)

    # Prediction loss
    axes[1].semilogy(train_pred, 'tab:blue', label="train")
    axes[1].semilogy(val_pred, 'tab:orange', label="val")
    if has_test:
        axes[1].axhline(data['test_pred'], color='tab:green', linestyle=':', label=f"test: {data['test_pred']:.2e}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCEWithLogitsLoss")
    axes[1].set_title('Prediction Loss')
    axes[1].legend()
    axes[1].annotate(f"train: {train_pred[-1]:.2e}\nval: {val_pred[-1]:.2e}",
                     xy=(0.55, 0.8), xycoords='axes fraction', fontsize=9)

    # Per-concept losses
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, name in enumerate(concept_names):
        c = colors[i % len(colors)]
        axes[2].semilogy(train_per_concept[name], color=c, linestyle='-', label=f'{name} (train)')
        axes[2].semilogy(val_per_concept[name], color=c, linestyle='--', label=f'{name} (val)')
        if has_test and name in data.get('test_per_concept', {}):
            axes[2].axhline(data['test_per_concept'][name], color=c, linestyle=':', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MSELoss')
    axes[2].set_title('Per-Concept Loss')
    axes[2].legend(fontsize=8, ncol=2)

    model_type = config['MODEL']['type']
    fig.suptitle(f'{model_type} Training Summary', fontsize=14)
    fig.savefig(f'{output_dir}/losses.png', dpi=300)
    plt.close(fig)
    print(f'Saved {output_dir}/losses.png', flush=True)

def plot_sample(model_dir=None, input_norm=None, concept_norm=None, val_loader=None, val_sample_idx=None, output_dir=None):
    """Plot prediction and concept maps for a validation sample across training epochs."""
    if model_dir is None:
        model_dir = find_output_dir()
    if output_dir is None:
        output_dir = model_dir
    if val_sample_idx is None:
        val_sample_idx = config.getint('VISUALIZATION', 'val_sample_idx', fallback=1)
    concepts = try_cast(config['DATASET']['concepts'])

    # Load land mask
    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    land_mask = (mask_2d == 0)

    if input_norm is None or concept_norm is None or val_loader is None:
        input_norm, concept_norm, _, _, val_loader, _ = get_dataset()
    val_dataset = val_loader.dataset
    data, concept_true, target_true = val_dataset[val_sample_idx]
    data = data.unsqueeze(0)
    concept_true = concept_true.unsqueeze(0)
    target_true = target_true.unsqueeze(0)

    # Compute target date for this sample
    window = config.getint('DATASET', 'context_window')
    offsets = try_cast(config['DATASET']['offset'])
    n_members = len(try_cast(config['DATASET']['members']))
    start_date = datetime.strptime(config['DATASET']['start'], "%Y-%m")
    end_date = datetime.strptime(config['DATASET']['end'], "%Y-%m")
    dates = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur.strftime("%Y-%m"))
        cur += relativedelta(months=1)
    n_times = len(dates) - window - max(offsets) + 1
    train_time_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n_times)
    train_end = train_time_end * n_members
    original_idx = train_end + val_sample_idx
    member = original_idx % n_members
    time_idx = original_idx // n_members
    target_dates = {off: dates[time_idx + window - 1 + off] for off in offsets}
    print(f'Val sample: idx={val_sample_idx}, original idx={original_idx}, member=opa{member}, target dates={target_dates}')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config['MODEL']['type']
    model = get_model()
    model.to(DEVICE)
    print(f'initialized model on {DEVICE}')

    data_gpu = torch.nan_to_num(input_norm.normalize(data), nan=0.0).to(DEVICE)

    n_epochs = config.getint('TRAINING', 'epochs')
    n_ckpt = config.getint('OUTPUT', 'n_epochs_between_checkpoints')
    epochs_to_check = list(range(0, n_epochs, n_ckpt))
    steps_mapping = [1, 3, 6]
    n_panels = len(epochs_to_check) + 1  # +1 for ground truth

    # Cache forward pass results per epoch (so we don't re-run for each concept/lead)
    epoch_results = {}
    for epoch in epochs_to_check:
        checkpoint_path = f'{model_dir}/{model_type}_epoch{epoch}.pt' # CHANGE THIS BACK
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}', flush=True)
            continue
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            output, concept_pred = model(data_gpu)
            epoch_results[epoch] = {
                'pred': output.cpu(),
                'concept': concept_norm.denormalize(concept_pred.cpu()),
            }
        print(f'Forward pass epoch {epoch} done', flush=True)

    # --- Prediction plots (one per lead time) ---
    for time_step in range(3):
        fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4, 3.5), layout='constrained')
        for ax in axes:
            ax.axis('off')

        gt = target_true[0, 0, time_step, :, :].cpu().numpy()
        gt_masked = np.ma.masked_where(land_mask, gt)
        axes[0].set_facecolor('white')
        im0 = axes[0].imshow(gt_masked, vmin=0, vmax=1, cmap='RdYlBu_r', aspect='equal', origin='lower')
        axes[0].set_title("Ground Truth")
        axes[0].axis('on')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label('P(MLHC Event)')

        for i, epoch in enumerate(epochs_to_check):
            if epoch not in epoch_results:
                continue
            pred_2d = epoch_results[epoch]['pred'][0, 0, time_step, :, :].numpy()
            masked = np.ma.masked_where(land_mask, pred_2d)
            vmin_p = float(np.nanmin(pred_2d))
            vmax_p = float(np.nanmax(pred_2d))
            ax = axes[i + 1]
            ax.set_facecolor('white')
            ax.axis('on')
            im = ax.imshow(masked, vmin=vmin_p, vmax=vmax_p, cmap='RdYlBu_r', aspect='equal', origin='lower')
            ax.set_title(f"Epoch {epoch}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('P(MLHC Event)')
        current_step = steps_mapping[time_step]
        target_month = target_dates[current_step]
        fig.suptitle(f'{model_type} Predictions: Lead {current_step}mo (target: {target_month}, opa{member})', fontsize=14)
        save_name = f'{output_dir}/{model_type}_pred_lead{current_step}.png'
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_name}', flush=True)

    # --- Concept plots (one per concept per lead time) ---
    for ci, cname in enumerate(concepts):
        for time_step in range(3):
            fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4, 3.5), layout='constrained')
            for ax in axes:
                ax.axis('off')

            gt = concept_true[0, ci, time_step, :, :].cpu().numpy()
            gt_masked = np.ma.masked_where(land_mask, gt)
            clean_arr = np.ma.filled(gt_masked.astype(float), np.nan)
            vmin, vmax = np.nanmin(clean_arr), np.nanmax(clean_arr)

            axes[0].set_facecolor('white')
            im0 = axes[0].imshow(gt_masked, cmap='RdYlBu_r', aspect='equal', vmin=vmin, vmax=vmax, origin='lower')
            axes[0].set_title("Ground Truth")
            axes[0].axis('on')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label(cname)

            for ei, epoch in enumerate(epochs_to_check):
                if epoch not in epoch_results:
                    continue
                pred_2d = epoch_results[epoch]['concept'][0, ci, time_step, :, :].numpy()
                masked = np.ma.masked_where(land_mask, pred_2d)
                clean = np.ma.filled(masked.astype(float), np.nan)
                vmin_p, vmax_p = np.nanmin(clean), np.nanmax(clean)
                ax = axes[ei + 1]
                ax.set_facecolor('white')
                ax.axis('on')
                im = ax.imshow(masked, cmap='RdYlBu_r', aspect='equal', vmin=vmin_p, vmax=vmax_p, origin='lower')
                ax.set_title(f"Epoch {epoch}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cname)
            current_step = steps_mapping[time_step]
            target_month = target_dates[current_step]
            fig.suptitle(f'{model_type} {cname.upper()}: Lead {current_step}mo (target: {target_month}, opa{member})', fontsize=14)
            save_name = f'{output_dir}/{model_type}_{cname}_lead{current_step}.png'
            fig.savefig(save_name, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {save_name}', flush=True)

def plot_sample_pred_only(model_dir=None, input_norm=None, val_loader=None,
                          val_sample_idx=None, thresholds=[0.5, 0.4, 0.3, 0.25]):
    """Plot binary prediction maps at multiple thresholds vs ground truth."""
    if model_dir is None:
        model_dir = find_output_dir()
    if val_sample_idx is None:
        val_sample_idx = config.getint('VISUALIZATION', 'val_sample_idx', fallback=1)

    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    land_mask = (mask_2d == 0)

    if input_norm is None or val_loader is None:
        input_norm, _, _, _, val_loader, _ = get_dataset()
    val_dataset = val_loader.dataset
    data, _, target_true = val_dataset[val_sample_idx]
    data = data.unsqueeze(0)
    target_true = target_true.unsqueeze(0)

    # Compute target dates
    window = config.getint('DATASET', 'context_window')
    offsets = try_cast(config['DATASET']['offset'])
    n_members = len(try_cast(config['DATASET']['members']))
    start_date = datetime.strptime(config['DATASET']['start'], "%Y-%m")
    end_date = datetime.strptime(config['DATASET']['end'], "%Y-%m")
    dates = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur.strftime("%Y-%m"))
        cur += relativedelta(months=1)
    n_times = len(dates) - window - max(offsets) + 1
    train_time_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n_times)
    train_end = train_time_end * n_members
    original_idx = train_end + val_sample_idx
    member = original_idx % n_members
    time_idx = original_idx // n_members
    target_dates = {off: dates[time_idx + window - 1 + off] for off in offsets}

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = config['MODEL']['type']
    model = get_model()
    model.to(DEVICE)

    checkpoints = sorted(glob.glob(f'{model_dir}/{model_type}_epoch*.pt'),
                         key=lambda p: int(p.split('epoch')[-1].split('.')[0]))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found in {model_dir}')
    checkpoint = torch.load(checkpoints[-1], map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Loaded {checkpoints[-1]}', flush=True)

    data_gpu = torch.nan_to_num(input_norm.normalize(data), nan=0.0).to(DEVICE)
    with torch.no_grad():
        output, _ = model(data_gpu)
    pred = output.cpu().numpy()  # (1, 1, n_leads, Y, X)

    cmap = ListedColormap(['#d0d0d0', '#d62728'])  # gray=no event, red=event

    for time_step, lead in enumerate(offsets):
        pred_2d = pred[0, 0, time_step]  # (Y, X)
        thresh = (float(np.nanmin(pred_2d)) + float(np.nanmax(pred_2d))) / 2

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
        for ax in axes:
            ax.axis('off')

        # Ground truth
        gt = target_true[0, 0, time_step, :, :].numpy()
        gt_masked = np.ma.masked_where(land_mask, gt)
        gt_pos_rate = float(np.nanmean(gt_masked))
        axes[0].set_facecolor('white')
        axes[0].imshow(gt_masked, cmap=cmap, vmin=0, vmax=1, aspect='equal', origin='lower')
        axes[0].set_title(f'Ground Truth\n({100*gt_pos_rate:.1f}% events)')
        axes[0].axis('on')

        # Binary prediction at midpoint threshold
        binary = np.ma.masked_where(land_mask, (pred_2d >= thresh).astype(float))
        pred_pos_rate = float(np.nanmean(binary))
        axes[1].set_facecolor('white')
        axes[1].axis('on')
        axes[1].imshow(binary, cmap=cmap, vmin=0, vmax=1, aspect='equal', origin='lower')
        axes[1].set_title(f'Prediction (t={thresh:.3f})\n({100*pred_pos_rate:.1f}% predicted)')

        target_month = target_dates[lead]
        fig.suptitle(f'{model_type} Binary Predictions: Lead {lead}mo (target: {target_month}, opa{member})', fontsize=12)
        save_name = f'{model_dir}/{model_type}_binary_lead{lead}.png'
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_name}', flush=True)


if __name__ == "__main__":
    plot_sample(model_dir='/home/kkringel/temp/20260222_MLParch/maike_c+b_concepts_64_64_64/PointwiseCBM_lam0.5_ep50_lr0.001_bs64_BCELoss')