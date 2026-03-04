import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import xarray as xr
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, brier_score_loss,
                             roc_curve, precision_recall_curve, average_precision_score)
from utils.get_data import get_dataset
from utils.get_config import config, try_cast, get_model
from utils.visualization import find_output_dir, plot_sample, visualize

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_dir):
    model_type = config['MODEL']['type']
    checkpoints = sorted(glob.glob(f'{model_dir}/{model_type}_epoch*.pt'),
                         key=lambda p: int(p.split('epoch')[-1].split('.')[0]))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found in {model_dir}')
    latest = checkpoints[-1]
    print(f'Loading {latest}')
    ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
    model = get_model()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

def save_val_preds(model_dir='/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.5_ep50_lr0.001_bs64_BCELoss_ZScore'):
    input_norm, _, _, _, val_loader, _ = get_dataset()

    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    ocean_mask = mask_2d == 1
    mask_tensor = torch.tensor(mask_2d, dtype=torch.float32)[None, None, None, :, :].to(DEVICE)

    offsets = try_cast(config['DATASET']['offset'])
    model = load_model(model_dir)

    preds, targets = [], []
    print('Running val inference (lead 0 only)...')
    with torch.no_grad():
        for batch, _, y in val_loader:
            batch = torch.nan_to_num(input_norm.normalize(batch), nan=0.0).to(DEVICE)
            pred, _ = model(batch)
            pred = (pred * mask_tensor).cpu().numpy()   # (B, 1, n_leads, Y, X)
            preds.append(pred[:, 0, 0])                 # (B, Y, X) — lead 0 only
            targets.append(y.numpy()[:, 0, 0])

    preds   = np.concatenate(preds,   axis=0)   # (N, Y, X)
    targets = np.concatenate(targets, axis=0)

    save_path = os.path.join(model_dir, 'val_preds_lead0.npz')
    np.savez_compressed(save_path, preds=preds, targets=targets,
                        ocean_mask=ocean_mask, lead=offsets[0])
    print(f'Saved {save_path}')



def run_inference(model_dir=None):
    if model_dir is None:
        model_dir = find_output_dir()

    input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()

    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    ocean_mask = mask_2d == 1
    mask_tensor = torch.tensor(mask_2d, dtype=torch.float32)[None, None, None, :, :].to(DEVICE)

    offsets = try_cast(config['DATASET']['offset'])
    n_leads = len(offsets)
    Y, X = mask_2d.shape

    model = load_model(model_dir)

    pixel_preds   = [[] for _ in range(n_leads)]
    pixel_targets = [[] for _ in range(n_leads)]

    print('Running inference on test set...')
    with torch.no_grad():
        for batch, _, y in test_loader:
            batch = torch.nan_to_num(input_norm.normalize(batch), nan=0.0).to(DEVICE)

            pred, _ = model(batch)
            pred = (pred * mask_tensor).cpu().numpy()  # (B, 1, n_leads, Y, X)
            y = y.numpy()                              # (B, 1, n_leads, Y, X)

            for li in range(n_leads):
                pixel_preds[li].append(pred[:, 0, li])    # (B, Y, X)
                pixel_targets[li].append(y[:, 0, li])     # (B, Y, X)

    for li in range(n_leads):
        pixel_preds[li]   = np.concatenate(pixel_preds[li],   axis=0)  # (N, Y, X)
        pixel_targets[li] = np.concatenate(pixel_targets[li], axis=0)  # (N, Y, X)

    # --- Ocean-only global metrics ---
    print('\n=== Ocean-Only Metrics ===')
    for li, lead in enumerate(offsets):
        pp = pixel_preds[li][:, ocean_mask].flatten()    # (N*ocean_pixels,)
        tt = pixel_targets[li][:, ocean_mask].flatten()
        valid = ~np.isnan(tt) & ~np.isnan(pp)
        pp, tt = pp[valid], tt[valid]

        bp = (pp >= 0.5).astype(int)
        bce = -np.mean(tt * np.log(pp + 1e-8) + (1 - tt) * np.log(1 - pp + 1e-8))

        print(f'\nLead {lead}mo:')
        print(f'  Ocean BCE Loss : {bce:.4f}')
        print(f'  Accuracy       : {accuracy_score(tt, bp):.4f}')
        print(f'  Precision      : {precision_score(tt, bp, zero_division=0):.4f}')
        print(f'  Recall         : {recall_score(tt, bp, zero_division=0):.4f}')
        print(f'  F1             : {f1_score(tt, bp, zero_division=0):.4f}')
        print(f'  AUC-ROC        : {roc_auc_score(tt, pp):.4f}')
        print(f'  Brier Score    : {brier_score_loss(tt, pp):.4f}')
        print(f'  Positive rate  : {tt.mean():.4f}')
        print(f'  Mean pred      : {pp.mean():.4f}')

    # --- Per-pixel skill maps ---
    print('\nComputing per-pixel skill maps...')
    fig, axes = plt.subplots(3, n_leads, figsize=(n_leads * 5, 10), layout='constrained')
    if n_leads == 1:
        axes = axes[:, None]

    for li, lead in enumerate(offsets):
        pp = pixel_preds[li]    # (N, Y, X)
        tt = pixel_targets[li]  # (N, Y, X)

        acc_map    = np.where(ocean_mask, np.nanmean((pp >= 0.5) == tt, axis=0), np.nan)
        brier_map  = np.where(ocean_mask, np.nanmean((pp - tt) ** 2,   axis=0), np.nan)
        mean_pred  = np.where(ocean_mask, np.nanmean(pp,                axis=0), np.nan)

        for ri, (data, title, cmap, vmin, vmax) in enumerate([
            (acc_map,   f'Accuracy — Lead {lead}mo',    'RdYlGn',   0,    1),
            (brier_map, f'Brier Score — Lead {lead}mo', 'RdYlGn_r', 0,    0.25),
            (mean_pred, f'Mean Pred — Lead {lead}mo',   'RdYlBu_r', 0,    1),
        ]):
            ax = axes[ri, li]
            im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap,
                           vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_path = f'{model_dir}/skill_maps.png'
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')

    # --- Threshold analysis ---
    threshold_analysis(pixel_preds, pixel_targets, ocean_mask, offsets, model_dir)


def threshold_analysis(pixel_preds, pixel_targets, ocean_mask, offsets, model_dir):
    """Find optimal classification threshold using ROC and PR curves."""
    n_leads = len(offsets)
    fig_roc, axes_roc = plt.subplots(1, n_leads, figsize=(n_leads * 5, 4), layout='constrained')
    fig_pr,  axes_pr  = plt.subplots(1, n_leads, figsize=(n_leads * 5, 4), layout='constrained')
    fig_dist, axes_dist = plt.subplots(1, n_leads, figsize=(n_leads * 5, 4), layout='constrained')
    if n_leads == 1:
        axes_roc  = [axes_roc]
        axes_pr   = [axes_pr]
        axes_dist = [axes_dist]

    print('\n=== Threshold Analysis ===')
    optimal_thresholds = {}

    for li, lead in enumerate(offsets):
        pp = pixel_preds[li][:, ocean_mask].flatten()
        tt = pixel_targets[li][:, ocean_mask].flatten()
        valid = ~np.isnan(tt) & ~np.isnan(pp)
        pp, tt = pp[valid], tt[valid]

        # --- ROC curve ---
        fpr, tpr, roc_thresholds = roc_curve(tt, pp)
        auc = roc_auc_score(tt, pp)
        youden_idx = np.argmax(tpr - fpr)
        youden_thresh = roc_thresholds[youden_idx]

        ax = axes_roc[li]
        ax.plot(fpr, tpr, color='tab:blue', lw=2, label=f'AUC = {auc:.3f}')
        ax.scatter(fpr[youden_idx], tpr[youden_idx], color='tab:red', zorder=5,
                   label=f'Youden J (t={youden_thresh:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f'ROC — Lead {lead}mo')
        ax.legend(fontsize=8)

        # --- PR curve ---
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(tt, pp)
        ap = average_precision_score(tt, pp)
        f1_vals = 2 * precision_vals[:-1] * recall_vals[:-1] / (precision_vals[:-1] + recall_vals[:-1] + 1e-8)
        best_f1_idx = np.argmax(f1_vals)
        best_f1_thresh = pr_thresholds[best_f1_idx]

        ax = axes_pr[li]
        ax.plot(recall_vals, precision_vals, color='tab:orange', lw=2, label=f'AP = {ap:.3f}')
        ax.scatter(recall_vals[best_f1_idx], precision_vals[best_f1_idx], color='tab:red', zorder=5,
                   label=f'Best F1 (t={best_f1_thresh:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'PR Curve — Lead {lead}mo')
        ax.legend(fontsize=8)

        # --- Prediction distribution ---
        ax = axes_dist[li]
        pos_preds = pp[tt == 1]
        neg_preds = pp[tt == 0]
        bins = np.linspace(0, 1, 60)
        ax.hist(neg_preds, bins=bins, alpha=0.6, color='tab:blue',  label=f'No event (n={len(neg_preds):,})', density=True)
        ax.hist(pos_preds, bins=bins, alpha=0.6, color='tab:red',   label=f'Event (n={len(pos_preds):,})', density=True)
        ax.axvline(0.5,             color='gray',    linestyle='--', lw=1, label='t=0.5')
        ax.axvline(youden_thresh,   color='tab:red', linestyle=':',  lw=1.5, label=f'Youden t={youden_thresh:.3f}')
        ax.axvline(best_f1_thresh,  color='tab:orange', linestyle=':', lw=1.5, label=f'Best F1 t={best_f1_thresh:.3f}')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Prediction Distribution — Lead {lead}mo')
        ax.legend(fontsize=7)

        # --- Report ---
        for label, thresh in [('0.5', 0.5), ('Youden J', youden_thresh), ('Best F1', best_f1_thresh)]:
            bp = (pp >= thresh).astype(int)
            print(f'\n  Lead {lead}mo | threshold={thresh:.3f} ({label}):')
            print(f'    Precision : {precision_score(tt, bp, zero_division=0):.4f}')
            print(f'    Recall    : {recall_score(tt, bp, zero_division=0):.4f}')
            print(f'    F1        : {f1_score(tt, bp, zero_division=0):.4f}')
            print(f'    Accuracy  : {accuracy_score(tt, bp):.4f}')

        optimal_thresholds[lead] = {'youden': youden_thresh, 'best_f1': best_f1_thresh}

    for fig, name in [(fig_roc, 'roc_curves'), (fig_pr, 'pr_curves'), (fig_dist, 'pred_distributions')]:
        path = f'{model_dir}/{name}.png'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')

    return optimal_thresholds


def concept_inference(model_dir=None, input_norm=None, concept_norm=None, val_loader=None, output_dir=None):
    """Evaluate concept predictions with MAE, RMSE and pattern correlation on validation set."""
    if model_dir is None:
        model_dir = find_output_dir()
    if output_dir is None:
        output_dir = model_dir

    if input_norm is None or concept_norm is None or val_loader is None:
        input_norm, concept_norm, _, _, val_loader, _ = get_dataset()

    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    ocean_mask = mask_2d == 1

    concept_names = try_cast(config['DATASET']['concepts'])
    offsets = try_cast(config['DATASET']['offset'])
    n_concepts = len(concept_names)
    n_leads = len(offsets)

    model = load_model(model_dir)

    # (n_concepts, n_leads): lists of (B, Y, X) arrays
    concept_preds   = [[[] for _ in range(n_leads)] for _ in range(n_concepts)]
    concept_targets = [[[] for _ in range(n_leads)] for _ in range(n_concepts)]

    print('Running concept inference on val set...')
    with torch.no_grad():
        for batch, concept_y, _ in val_loader:
            batch = torch.nan_to_num(input_norm.normalize(batch), nan=0.0).to(DEVICE)
            _, concept_pred = model(batch)
            # concept_pred is in normalized space; denormalize to physical units
            concept_pred = concept_norm.denormalize(concept_pred.cpu())  # (B, C, n_leads, Y, X)
            # concept_y from loader is in raw (unnormalized) space

            for ci in range(n_concepts):
                for li in range(n_leads):
                    concept_preds[ci][li].append(concept_pred[:, ci, li].numpy())    # (B, Y, X)
                    concept_targets[ci][li].append(concept_y[:, ci, li].numpy())

    for ci in range(n_concepts):
        for li in range(n_leads):
            concept_preds[ci][li]   = np.concatenate(concept_preds[ci][li],   axis=0)  # (N, Y, X)
            concept_targets[ci][li] = np.concatenate(concept_targets[ci][li], axis=0)

    # --- Global metrics ---
    print('\n=== Concept Metrics (ocean pixels only) ===')
    for ci, cname in enumerate(concept_names):
        print(f'\n  {cname}:')
        for li, lead in enumerate(offsets):
            pp = concept_preds[ci][li][:, ocean_mask].flatten()
            tt = concept_targets[ci][li][:, ocean_mask].flatten()
            valid = ~np.isnan(tt) & ~np.isnan(pp)
            pp, tt = pp[valid], tt[valid]

            mae  = np.mean(np.abs(pp - tt))
            rmse = np.sqrt(np.mean((pp - tt) ** 2))
            # pattern correlation: per timestep spatial r, averaged over time (vectorized)
            p_all = concept_preds[ci][li][:, ocean_mask]   # (N, ocean_pixels)
            t_all = concept_targets[ci][li][:, ocean_mask]
            p_m = np.nanmean(p_all, axis=1, keepdims=True)
            t_m = np.nanmean(t_all, axis=1, keepdims=True)
            p_d = p_all - p_m
            t_d = t_all - t_m
            num = np.nansum(p_d * t_d, axis=1)
            denom = np.sqrt(np.nansum(p_d ** 2, axis=1) * np.nansum(t_d ** 2, axis=1))
            pattern_corr = np.nanmean(np.where(denom > 0, num / denom, np.nan))

            print(f'    Lead {lead}mo | MAE={mae:.4g}  RMSE={rmse:.4g}  PatCorr={pattern_corr:.4f}')

    # --- Spatial correlation maps (one figure per concept) ---
    print('\nSaving spatial correlation maps...')
    for ci, cname in enumerate(concept_names):
        fig, axes = plt.subplots(1, n_leads, figsize=(n_leads * 5, 4), layout='constrained')
        if n_leads == 1:
            axes = [axes]

        for li, lead in enumerate(offsets):
            pp = concept_preds[ci][li]   # (N, Y, X)
            tt = concept_targets[ci][li]

            # Vectorized Pearson r over time axis for all pixels at once
            pp_m = np.nanmean(pp, axis=0, keepdims=True)
            tt_m = np.nanmean(tt, axis=0, keepdims=True)
            pp_d = pp - pp_m
            tt_d = tt - tt_m
            num = np.nansum(pp_d * tt_d, axis=0)
            denom = np.sqrt(np.nansum(pp_d ** 2, axis=0) * np.nansum(tt_d ** 2, axis=0))
            corr_map = np.where((denom > 0) & ocean_mask, num / denom, np.nan)

            ax = axes[li]
            im = ax.imshow(np.ma.masked_invalid(corr_map), cmap='RdYlGn',
                           vmin=-1, vmax=1, origin='lower', aspect='equal')
            ax.set_title(f'Lead {lead}mo', fontsize=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Pearson r')

        fig.suptitle(f'Pattern Correlation — {cname}', fontsize=12)
        path = f'{output_dir}/concept_corr_{cname}.png'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {path}')


if __name__ == '__main__':
    save_val_preds()
    # Run with config.ini set to norm_type = MinMax:
    # MODEL_DIR = '/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.5_ep50_lr0.001_bs64_BCELoss_MinMax'
    # Run with config.ini set to norm_type = ZScore:
    #MODEL_DIR = '/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.5_ep50_lr0.001_bs64_BCELoss_ZScore'

    #input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()

    #visualize()
    #plot_sample(model_dir=MODEL_DIR, input_norm=input_norm, concept_norm=concept_norm, val_loader=val_loader)
    #plot_sample_pred_only(model_dir=MODEL_DIR, input_norm=input_norm, val_loader=val_loader)
    #run_inference(model_dir=MODEL_DIR)
    #concept_inference(model_dir=MODEL_DIR, input_norm=input_norm, concept_norm=concept_norm, val_loader=val_loader)
