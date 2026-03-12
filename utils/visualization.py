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

def plot_sample(model_dir=None, input_norm=None, concept_norm=None, output_norm=None, val_loader=None, val_sample_idx=None, output_dir=None):
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

    if input_norm is None or concept_norm is None or output_norm is None or val_loader is None:
        input_norm, concept_norm, output_norm, _, val_loader, _ = get_dataset()
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
            output, concept_pred, _ = model(data_gpu)
            epoch_results[epoch] = {
                'pred': output_norm.denormalize(output.cpu()),
                'concept': concept_norm.denormalize(concept_pred.cpu()),
            }
        print(f'Forward pass epoch {epoch} done', flush=True)

    # --- Prediction plots (one per lead time) ---
    for time_step in range(len(try_cast(config['DATASET']['offset']))):
        fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4, 3.5), layout='constrained')
        for ax in axes:
            ax.axis('off')

        gt = target_true[0, 0, time_step, :, :].cpu().numpy()
        gt_masked = np.ma.masked_where(land_mask, gt)
        vmin_gt, vmax_gt = float(np.nanmin(gt)), float(np.nanmax(gt))
        axes[0].set_facecolor('white')
        im0 = axes[0].imshow(gt_masked, vmin=vmin_gt, vmax=vmax_gt, cmap='RdYlBu_r', aspect='equal', origin='lower')
        axes[0].set_title("Ground Truth")
        axes[0].axis('on')
        label = try_cast(config['DATASET']['labels'])[0]
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label(label)

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
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(label)
        current_step = steps_mapping[time_step]
        target_month = target_dates[current_step]
        fig.suptitle(f'{model_type} Predictions: Lead {current_step}mo (target: {target_month}, opa{member})', fontsize=14)
        save_name = f'{output_dir}/{model_type}_pred_lead{current_step}.png'
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_name}', flush=True)

    # --- Concept plots (one per concept per lead time) ---
    for ci, cname in enumerate(concepts):
        for time_step in range(len(try_cast(config['DATASET']['offset']))):
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
        output, _, _ = model(data_gpu)
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

            for ei, epoch in enumerate(epochs_to_check):
                checkpoint_path = f'{model_dir}/unet_epoch{epoch}.pt'
                if not os.path.exists(checkpoint_path):
                    print(f'Checkpoint not found: {checkpoint_path}', flush=True)
                    continue
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                with torch.no_grad():
                    _, concept, _ = model(data_gpu)
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

def plot_free_concept(model_dir=None):
    """Visualize free (unsupervised) concept spatial maps across epochs."""
    from models import UNetCBM
    from utils.get_config import config, try_cast
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    if model_dir is None:
        model_dir = find_output_dir()

    n_free = config.getint('MODEL', 'n_free_concepts', fallback=0)
    if n_free == 0:
        print('No free concepts configured, skipping.')
        return

    # Load land mask
    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    land_mask = (mask_2d == 0)

    # Setup validation set
    from utils.get_data import get_dataset_preload
    input_norm, concept_norm, _, _, val_loader, _ = get_dataset_preload()
    data, concept_true, target_true = next(iter(val_loader))
    val_sample_idx = 1
    data = data[val_sample_idx:val_sample_idx+1]

    # Compute target date
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

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetCBM(
        n_features=len(try_cast(config['DATASET']['features'])) * config.getint('DATASET', 'context_window'),
        n_concepts=len(try_cast(config['DATASET']['concepts'])),
        output_dim=len(try_cast(config['DATASET']['offset'])),
        n_free_concepts=n_free,
    )
    model.to(DEVICE)

    data_gpu = torch.nan_to_num(input_norm.normalize(data), nan=0.0).to(DEVICE)

    n_epochs = config.getint('TRAINING', 'epochs')
    n_ckpt = config.getint('OUTPUT', 'n_epochs_between_checkpoints')
    epochs_to_check = list(range(0, n_epochs, n_ckpt))
    steps_mapping = [1, 3, 6]

    for fi in range(n_free):
        for time_step in range(3):
            print(f'starting free concept {fi} time step {time_step}', flush=True)
            n_panels = len(epochs_to_check)
            fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4, 3.5), layout='constrained')
            if n_panels == 1:
                axes = [axes]

            for ax in axes:
                ax.axis('off')

            # Collect predictions to set consistent colorscale
            preds = []
            for ei, epoch in enumerate(epochs_to_check):
                checkpoint_path = f'{model_dir}/unet_epoch{epoch}.pt'
                if not os.path.exists(checkpoint_path):
                    print(f'Checkpoint not found: {checkpoint_path}', flush=True)
                    continue
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                with torch.no_grad():
                    _, _, free = model(data_gpu)
                    pred_2d = free[0, fi, time_step, :, :].cpu().numpy()
                preds.append((ei, epoch, pred_2d))

            # Compute shared colorscale from all epochs
            all_vals = np.concatenate([np.ma.filled(np.ma.masked_where(land_mask, p[2]).astype(float), np.nan).ravel() for p in preds])
            vmin, vmax = np.nanpercentile(all_vals, [2, 98])

            for ei, epoch, pred_2d in preds:
                ax = axes[ei]
                ax.set_facecolor('white')
                ax.axis('on')
                pred_masked = np.ma.masked_where(land_mask, pred_2d)
                im = ax.imshow(pred_masked, cmap='RdYlBu_r', aspect='equal', vmin=vmin, vmax=vmax, origin='lower')
                ax.set_title(f"Epoch {epoch}")

            cbar = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)

            current_step = steps_mapping[time_step]
            target_month = target_dates[current_step]
            fig.suptitle(f'Free Concept {fi}: Lead Time {current_step} months (target: {target_month}, opa{member})', fontsize=14)
            save_name = f'{model_dir}/unet_free{fi}_lead{current_step}.png'
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
    n_free = config.getint('MODEL', 'n_free_concepts', fallback=0)
    out_loss_fn = getattr(torch.nn, config['TRAINING']['out_loss_fn'])()

    model = UNetCBM(
        n_features=len(try_cast(config['DATASET']['features'])) * config.getint('DATASET', 'context_window'),
        n_concepts=n_concepts,
        output_dim=output_dim,
        n_free_concepts=n_free,
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
                pred, _, _ = model(batch)
                pred = pred * mask_tensor
                pred_loss_accum += out_loss_fn(pred, y).item()

                # Ground-truth concepts through output_head
                # Need to include free concepts (from model) alongside GT supervised concepts
                B, C, T, Y, X = concept_y.shape
                gt_flat = concept_y.view(B, C * T, Y, X)  # (B, n_concepts*output_dim, Y, X)
                if n_free > 0:
                    # Run encoder/decoder to get free concepts
                    _, _, free_pred = model(batch)
                    free_flat = free_pred.view(B, n_free * T, Y, X)
                    gt_flat = torch.cat([gt_flat, free_flat], dim=1)
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
        target_month = target_dates[lead]
        fig.suptitle(f'{model_type} Binary Predictions: Lead {lead}mo (target: {target_month}, opa{member})', fontsize=12)
        save_name = f'{model_dir}/{model_type}_binary_lead{lead}.png'
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_name}', flush=True)


def plot_concept_importance(model_dir=None):
    """Analyze which concepts and inputs matter most for prediction."""
    from models import UNetCBM
    from utils.get_config import config, try_cast
    from utils.get_data import get_dataset_preload

    if model_dir is None:
        model_dir = find_output_dir()

    concept_names = try_cast(config['DATASET']['concepts'])
    feature_names = try_cast(config['DATASET']['features'])
    n_concepts = len(concept_names)
    n_features = len(feature_names)
    output_dim = len(try_cast(config['DATASET']['offset']))
    n_free = config.getint('MODEL', 'n_free_concepts', fallback=0)
    window = config.getint('DATASET', 'context_window')

    model = UNetCBM(
        n_features=n_features * window,
        n_concepts=n_concepts,
        output_dim=output_dim,
        n_free_concepts=n_free,
    )

    n_epochs = config.getint('TRAINING', 'epochs')
    n_ckpt = config.getint('OUTPUT', 'n_epochs_between_checkpoints')
    epochs_to_check = list(range(0, n_epochs, n_ckpt))

    # --- Panel 1: Output head weight magnitude per concept across epochs ---
    weight_importance = {name: [] for name in concept_names}
    if n_free > 0:
        for fi in range(n_free):
            weight_importance[f'free_{fi}'] = []

    for epoch in epochs_to_check:
        checkpoint_path = f'{model_dir}/unet_epoch{epoch}.pt'
        if not os.path.exists(checkpoint_path):
            continue
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # output_head[0] is Conv2d((n_concepts+n_free)*output_dim, 64, 3, 3)
        w = model.output_head[0].weight.data  # (64, (n_concepts+n_free)*output_dim, 3, 3)
        for ci, name in enumerate(concept_names):
            # Each concept owns output_dim channels
            start = ci * output_dim
            end = start + output_dim
            weight_importance[name].append(w[:, start:end, :, :].norm().item())
        for fi in range(n_free):
            start = (n_concepts + fi) * output_dim
            end = start + output_dim
            weight_importance[f'free_{fi}'].append(w[:, start:end, :, :].norm().item())

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), layout='constrained')
    for name, vals in weight_importance.items():
        style = '--' if name.startswith('free') else '-'
        ax.plot(epochs_to_check[:len(vals)], vals, style, label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L2 Norm of Output Head Weights')
    ax.set_title('Concept Importance (Weight Magnitude)')
    ax.legend()
    save_name = f'{model_dir}/concept_weight_importance.png'
    fig.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_name}')

    # --- Panel 2: Gradient-based sensitivity on validation data ---
    input_norm, concept_norm, _, _, val_loader, _ = get_dataset_preload()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use the last checkpoint
    last_epoch = epochs_to_check[-1]
    checkpoint_path = f'{model_dir}/unet_epoch{last_epoch}.pt'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Load land mask for masking
    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    mask_tensor = torch.tensor(mask_2d, dtype=torch.float32)[None, None, None, :, :].to(DEVICE)

    concept_grad_accum = np.zeros(n_concepts)
    input_grad_accum = np.zeros(n_features)
    n_samples = 0

    for batch, concept_y, y in val_loader:
        batch = torch.nan_to_num(input_norm.normalize(batch), nan=0.0).to(DEVICE)
        batch.requires_grad_(True)

        pred, concepts_pred, _ = model(batch)
        pred = pred * mask_tensor

        # Gradient of prediction w.r.t. concepts
        # concepts_pred: (B, n_concepts, output_dim, Y, X)
        concepts_pred.retain_grad()
        loss = pred.sum()
        loss.backward()

        # Concept sensitivity: mean |grad| per concept
        if concepts_pred.grad is not None:
            cg = concepts_pred.grad.abs().mean(dim=(0, 2, 3, 4)).cpu().numpy()  # (n_concepts,)
            concept_grad_accum += cg

        # Input sensitivity: mean |grad| per feature
        # batch: (B, n_features, window, Y, X)
        if batch.grad is not None:
            ig = batch.grad.abs().mean(dim=(0, 2, 3, 4)).cpu().numpy()  # (n_features,)
            input_grad_accum += ig

        n_samples += 1
        model.zero_grad()
        if n_samples >= 10:  # enough samples for stable estimate
            break

    concept_grad_accum /= n_samples
    input_grad_accum /= n_samples

    # Bar chart: concepts and inputs side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')

    ax1.bar(concept_names, concept_grad_accum, color='steelblue')
    ax1.set_ylabel('Mean |gradient|')
    ax1.set_title(f'Concept Sensitivity (epoch {last_epoch})')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(feature_names, input_grad_accum, color='coral')
    ax2.set_ylabel('Mean |gradient|')
    ax2.set_title(f'Input Feature Sensitivity (epoch {last_epoch})')
    ax2.tick_params(axis='x', rotation=45)

    save_name = f'{model_dir}/sensitivity_analysis.png'
    fig.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_name}')


if __name__ == "__main__":
    plot_sample(model_dir='/home/kkringel/temp/20260222_MLParch/maike_c+b_concepts_64_64_64/PointwiseCBM_lam0.5_ep50_lr0.001_bs64_BCELoss')
