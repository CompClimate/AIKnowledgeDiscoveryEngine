import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import xarray as xr
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, brier_score_loss,
                             roc_curve, precision_recall_curve, average_precision_score)
from utils.get_data import get_dataset
from utils.get_config import config, try_cast, get_model
from utils.visualization import find_output_dir, plot_sample, visualize
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# claude written function to find the latest directory (but i have changed directories many times so may be redundant)
def load_model(model_dir, epoch=None):
    model_type = config['MODEL']['type']
    if epoch is not None:
        ckpt_path = f'{model_dir}/{model_type}_epoch{epoch}.pt'
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    else:
        checkpoints = sorted(glob.glob(f'{model_dir}/{model_type}_epoch*.pt'),
                             key=lambda p: int(p.split('epoch')[-1].split('.')[0]))
        if not checkpoints:
            raise FileNotFoundError(f'No checkpoints found in {model_dir}')
        ckpt_path = checkpoints[-1]
    print(f'Loading {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = get_model()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

# saves all predictions for the entire input dataset (needed for ACC calculation)
def save_all_preds(model_dir=None, input_norm=None, concept_norm=None, output_norm=None,
                   output_dir=None, full_loader=None):
    if model_dir is None:
        model_dir = find_output_dir()
    if output_dir is None:
        output_dir = model_dir

    config.read(f'{model_dir}/config.ini')

    from torch.utils.data import DataLoader
    if full_loader is None:
        if input_norm is None or concept_norm is None or output_norm is None:
            input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
        full_dataset = train_loader.dataset.dataset  # underlying EmulatorDataset
        full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    ocean_mask = mask_2d == 1
    mask_tensor = torch.tensor(mask_2d, dtype=torch.float32)[None, None, None, :, :].to(DEVICE)

    offsets = try_cast(config['DATASET']['offset'])
    model = load_model(model_dir)

    concept_names = try_cast(config['DATASET']['concepts'])
    n_concepts = len(concept_names)

    preds, targets = [], []
    concept_preds   = [[] for _ in range(n_concepts)]
    concept_targets = [[] for _ in range(n_concepts)]

    print('Running inference on full dataset...')
    with torch.no_grad():
        for batch, concept_y, y in full_loader:
            batch = torch.nan_to_num(input_norm.normalize(batch), nan=0.0).to(DEVICE)
<<<<<<< HEAD
            pred, cpred, *_ = model(batch)
=======
            pred, cpred, _ = model(batch)
>>>>>>> 2efb97f (mergin free concept and regression)
            pred = (pred * mask_tensor).cpu()
            preds.append(pred[:, 0, 0])
            targets.append(output_norm.normalize(y).numpy()[:, 0, 0])
            cpred = cpred.cpu()
            for ci in range(n_concepts):
                concept_preds[ci].append(cpred[:, ci, 0].numpy())
                concept_targets[ci].append(concept_norm.normalize(concept_y)[:, ci, 0].numpy())

    preds   = np.concatenate(preds,   axis=0)
    targets = np.concatenate(targets, axis=0)
    for ci in range(n_concepts):
        concept_preds[ci]   = np.concatenate(concept_preds[ci],   axis=0)
        concept_targets[ci] = np.concatenate(concept_targets[ci], axis=0)

    save_path = os.path.join(output_dir, 'all_preds.npz')
    np.savez_compressed(
        save_path,
        preds=preds,
        targets=targets,
        ocean_mask=ocean_mask,
        lead=offsets[0],
        concept_preds=np.stack(concept_preds),
        concept_targets=np.stack(concept_targets),
        concept_names=np.array(concept_names),
        output_mean=output_norm.mean.numpy(),
        output_std=output_norm.std.numpy(),
        concept_mean=concept_norm.mean.numpy(),
        concept_std=concept_norm.std.numpy(),
    )
    print(f'Saved {save_path}')



# saving prediction for validation and test set, including the free concept
def save_val_preds(model_dir=None, input_norm=None, concept_norm=None, output_norm=None, val_loader=None, test_loader=None, output_dir=None):
    if model_dir is None:
        model_dir = find_output_dir()
    if output_dir is None:
        output_dir = model_dir

    config.read(f'{model_dir}/config.ini')

    if input_norm is None or concept_norm is None or output_norm is None or val_loader is None:
        input_norm, concept_norm, output_norm, _, val_loader, test_loader = get_dataset()


    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    ocean_mask = mask_2d == 1
    mask_tensor = torch.tensor(mask_2d, dtype=torch.float32)[None, None, None, :, :].to(DEVICE)

    offsets = try_cast(config['DATASET']['offset'])
    model = load_model(model_dir)

    concept_names = try_cast(config['DATASET']['concepts'])
    n_concepts = len(concept_names)

    preds, targets = [], []
    concept_preds   = [[] for _ in range(n_concepts)]
    concept_targets = [[] for _ in range(n_concepts)]

    n_free_concepts = config.getint('MODEL.HYPERPARAMETERS', 'n_free_concepts', fallback=0)
    free_preds = [[] for _ in range(n_free_concepts)]

    print('Running val+test inference (lead 0 only)...')
    loaders = [val_loader] if test_loader is None else [val_loader, test_loader]
    with torch.no_grad():
        for loader in loaders:
            for batch, concept_y, y in loader:
                batch = torch.nan_to_num(input_norm.normalize(batch), nan=0.0).to(DEVICE)
                pred, cpred, free = model(batch)
                pred = (pred * mask_tensor).cpu()
                pred = output_norm.denormalize(pred).numpy()    # (B, 1, n_leads, Y, X)
                cpred = concept_norm.denormalize(cpred.cpu())   # (B, n_concepts, n_leads, Y, X)
                preds.append(pred[:, 0, 0])                     # (B, Y, X)
                targets.append(y.numpy()[:, 0, 0])
                for ci in range(n_concepts):
                    concept_preds[ci].append(cpred[:, ci, 0].numpy())    # (B, Y, X)
                    concept_targets[ci].append(concept_y[:, ci, 0].numpy())
                if free is not None:
                    for fi in range(n_free_concepts):
                        free_preds[fi].append((free[:, fi, 0] * mask_tensor[0, 0, 0]).cpu().numpy())

    preds   = np.concatenate(preds,   axis=0)   # (N, Y, X)
    targets = np.concatenate(targets, axis=0)
    for ci in range(n_concepts):
        concept_preds[ci]   = np.concatenate(concept_preds[ci],   axis=0)
        concept_targets[ci] = np.concatenate(concept_targets[ci], axis=0)
    for fi in range(n_free_concepts):
        free_preds[fi] = np.concatenate(free_preds[fi], axis=0)

    save_path = os.path.join(output_dir, 'val_preds_lead0.npz')
    save_dict = dict(
        preds=preds,
        targets=targets,
        ocean_mask=ocean_mask,
        lead=offsets[0],
        concept_preds=np.stack(concept_preds),      # (n_concepts, N, Y, X)
        concept_targets=np.stack(concept_targets),  # (n_concepts, N, Y, X)
        concept_names=np.array(concept_names),
    )
    if n_free_concepts > 0:
        save_dict['free_preds'] = np.stack(free_preds)  # (n_free, N, Y, X) normalized
    np.savez_compressed(save_path, **save_dict)
    print(f'Saved {save_path}')


# comparing mlhc and sst events (not really inference)
def compare_mlhc_sst():
    # NA crop indices from mesh
    lon_bounds = (-80, 20)
    lat_bounds = (20, 66)
    mesh_ds = xr.open_dataset('/quobyte/maikesgrp/kkringel/oras5/ORCA025/mesh/mesh_mask.nc')
    nav_lon = mesh_ds['nav_lon'].squeeze()
    nav_lat = mesh_ds['nav_lat'].squeeze()
    mask_na = (nav_lon >= lon_bounds[0]) & (nav_lon <= lon_bounds[1]) & (nav_lat >= lat_bounds[0]) & (nav_lat <= lat_bounds[1])
    y_inds = mask_na.any(dim='x')
    x_inds = mask_na.any(dim='y')
    ocean_mask_na = mesh_ds['tmaskutil'].squeeze().isel(y=y_inds, x=x_inds).values == 1

    # sst anomalies
    sst_anom = xr.open_dataset('/quobyte/maikesgrp/sanah/sst_anomaly/sst_anomalies_2010_detrended.nc')
    sst_thresh = xr.open_dataset('/quobyte/maikesgrp/sanah/sst_anomaly/sst_anomalies_90th_percentile_detrended_latest.nc')
    mhw_sst = (sst_anom.isel(y=y_inds, x=x_inds).sst_anomaly.values > sst_thresh.isel(y=y_inds, x=x_inds).sst_anomaly.values).astype(float)
    mhw_sst = np.where(ocean_mask_na[..., None], mhw_sst, np.nan)
    # mlhc anomalies
    mlhc_anom = xr.open_dataset('/quobyte/maikesgrp/sanah/mlhc_anomaly/opa0/mlhc_anomalies_2010_detrended.nc')
    mlhc_thresh = xr.open_dataset('/quobyte/maikesgrp/sanah/mlhc_anomaly/mlhc_anomalies_90th_percentile_detrended_opa0.nc')
    mhw_mlhc = (mlhc_anom.isel(y=y_inds, x=x_inds).mlhc_anomaly.values > mlhc_thresh.isel(y=y_inds, x=x_inds).mlhc_anomaly.values).astype(float)
    mhw_mlhc = np.where(ocean_mask_na[..., None], mhw_mlhc, np.nan)
    # # overlap: 0=no event, 1=sst only, 2=mlhc only, 3=both
    # s = mhw_sst[:, :, 1]
    # m = mhw_mlhc[:, :, 0]
    # overlap = np.zeros_like(s)
    # overlap = np.where((s == 1) & (m == 0), 1, overlap)   # sst only
    # overlap = np.where((s == 0) & (m == 1), 2, overlap)   # mlhc only
    # overlap = np.where((s == 1) & (m == 1), 3, overlap)   # both
    # overlap = np.where(~np.isnan(s), overlap, np.nan)

    # from matplotlib.colors import ListedColormap, BoundaryNorm
    # cmap_overlap = ListedColormap(['#d0d0d0', '#1f77b4', '#ff7f0e', '#d62728'])
    # norm_overlap = BoundaryNorm([0, 1, 2, 3, 4], cmap_overlap.N)

    # # mld
    # loc = config['DATASET']['location']
    # mld = xr.open_zarr(f'{loc}/opa0/somxl010_na.zarr').somxl010
    # mld_month = mld.sel(time_counter='2010-01').values.squeeze()
    # mld_month = np.where(ocean_mask_na, mld_month, np.nan)

    # fig, ax = plt.subplots(1, 4, figsize=(16, 3))
    # ax[0].imshow(mhw_sst[:, :, 0], origin='lower', cmap='Reds', vmin=0, vmax=1)
    # ax[1].imshow(mhw_mlhc[:, :, 0], origin='lower', cmap='Reds', vmin=0, vmax=1)
    # im2 = ax[2].imshow(overlap, origin='lower', cmap=cmap_overlap, norm=norm_overlap)
    # im3 = ax[3].imshow(mld_month, origin='lower', cmap='viridis_r')
    # ax[0].set_title('SST MHW')
    # ax[1].set_title('MLHC MHW')
    # ax[2].set_title('Overlap')
    # ax[3].set_title('MLD (m)')
    # cbar = fig.colorbar(im2, ax=ax[2], ticks=[0.5, 1.5, 2.5, 3.5])
    # cbar.ax.set_yticklabels(['no event', 'SST only', 'MLHC only', 'both'])
    # fig.colorbar(im3, ax=ax[3])
    # fig.suptitle('Comparing MHWs')
    # fig.tight_layout()
    # fig.savefig('comparing_mhw')

    # time series of overlap fractions across all years
    import pandas as pd
    from scipy.stats import pearsonr
    loc = config['DATASET']['location']
    mld_zarr = xr.open_zarr(f'{loc}/opa0/somxl010_na.zarr').somxl010

    frac_sst_only, frac_mlhc_only, frac_both = [], [], []
    mean_sst_anom, mean_mlhc_mld = [], []  # for MLHC/MLD vs SST comparison
    dates = []
    for year in range(1980, 2019):
        sst_anom_yr = xr.open_dataset(f'/quobyte/maikesgrp/sanah/sst_anomaly/sst_anomalies_{year}_detrended.nc')
        mlhc_anom_yr = xr.open_dataset(f'/quobyte/maikesgrp/sanah/mlhc_anomaly/opa0/mlhc_anomalies_{year}_detrended.nc')
        mhw_sst_yr = (sst_anom_yr.isel(y=y_inds, x=x_inds).sst_anomaly.values > sst_thresh.isel(y=y_inds, x=x_inds).sst_anomaly.values).astype(float)
        mhw_sst_yr = np.where(ocean_mask_na[..., None], mhw_sst_yr, np.nan)
        mhw_mlhc_yr = (mlhc_anom_yr.isel(y=y_inds, x=x_inds).mlhc_anomaly.values > mlhc_thresh.isel(y=y_inds, x=x_inds).mlhc_anomaly.values).astype(float)
        mhw_mlhc_yr = np.where(ocean_mask_na[..., None], mhw_mlhc_yr, np.nan)
        mlhc_na = mlhc_anom_yr.isel(y=y_inds, x=x_inds).mlhc_anomaly.values  # (Y, X, 12)
        mld_yr = mld_zarr.sel(time_counter=str(year)).values  # (12, Y, X)
        mld_yr = np.moveaxis(mld_yr, 0, -1)  # (Y, X, 12)
        rho_cp = 1026.0 * 3990.0  # from concepts.py mlhc(): rho=1026, cp=3990 J/(m³·K)
        mlhc_mld_yr = mlhc_na / ((mld_yr + 1e-6) * rho_cp)  # units: K (temperature anomaly proxy)
        sst_na = sst_anom_yr.isel(y=y_inds, x=x_inds).sst_anomaly.values  # (Y, X, 12)
        n_months = mhw_sst_yr.shape[2]
        for t in range(n_months):
            s = mhw_sst_yr[:, :, t]
            m = mhw_mlhc_yr[:, :, t]
            valid = ~np.isnan(s)
            n_valid = valid.sum()
            frac_sst_only.append(np.nansum((s == 1) & (m == 0)) / n_valid)
            frac_mlhc_only.append(np.nansum((s == 0) & (m == 1)) / n_valid)
            frac_both.append(np.nansum((s == 1) & (m == 1)) / n_valid)
            mean_sst_anom.append(np.nanmean(sst_na[:, :, t]))
            mean_mlhc_mld.append(np.nanmean(mlhc_mld_yr[:, :, t]))
            dates.append(pd.Timestamp(f'{year}-{t+1:02d}'))

    # Pearson correlations
    r1, p1 = pearsonr(frac_sst_only, frac_mlhc_only)
    r2, p2 = pearsonr(frac_sst_only, frac_both)
    r3, p3 = pearsonr(frac_mlhc_only, frac_both)
    r4, p4 = pearsonr(mean_sst_anom, mean_mlhc_mld)
    print(f'SST-only vs MLHC-only:  r={r1:.3f}, p={p1:.4f}')
    print(f'SST-only vs Both:       r={r2:.3f}, p={p2:.4f}')
    print(f'MLHC-only vs Both:      r={r3:.3f}, p={p3:.4f}')
    print(f'Mean SST anom vs mean MLHC/MLD: r={r4:.3f}, p={p4:.4f}')

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(dates, frac_sst_only, label='SST only', color='#1f77b4')
    axes[0].plot(dates, frac_mlhc_only, label='MLHC only', color='#ff7f0e')
    axes[0].plot(dates, frac_both, label='Both', color='#d62728')
    axes[0].set_ylabel('Fraction of ocean points')
    axes[0].set_title(f'MHW overlap time series (opa0, NA) | SST vs MLHC r={r1:.3f}')
    axes[0].legend()
    axes[1].plot(dates, mean_sst_anom, label='Mean SST anomaly', color='#1f77b4')
    ax2 = axes[1].twinx()
    ax2.plot(dates, mean_mlhc_mld, label='Mean MLHC/MLD', color='#2ca02c', alpha=0.7)
    axes[1].set_ylabel('SST anomaly (°C)')
    ax2.set_ylabel('MLHC/MLD')
    axes[1].set_title(f'Mean SST anomaly vs MLHC/MLD | r={r4:.3f}')
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1].set_xlabel('Date')
    fig.tight_layout()
    fig.savefig('comparing_mhw_timeseries')

# plotting pearonr spatially and temporally over the entire validation set
def plot_pearsonr(model_dir):
    if model_dir is None:
        model_dir = find_output_dir()

    config.read(f'{model_dir}/config.ini')

    import pandas as pd
    results = np.load(f'{model_dir}/val_preds_lead0.npz', allow_pickle=True)
    preds, targets, concept_preds, concept_targets, ocean_mask, concept_names = results['preds'], results['targets'], results['concept_preds'], results['concept_targets'], results['ocean_mask'], results['concept_names']
    T, Y, X = preds.shape
    ocean_mask_flat = ocean_mask.reshape(-1)

    # Reconstruct validation dates
    window  = config.getint('DATASET', 'context_window')
    offset  = try_cast(config['DATASET']['offset'])[0]
    n_members = len(try_cast(config['DATASET']['members']))
    dates   = pd.date_range(start=config['DATASET']['start'], end=config['DATASET']['end'], freq='MS')
    n_times = len(dates) - window - offset + 1
    train_time_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n_times)
    val_dates = [dates[t + window - 1 + offset] for t in range(train_time_end, train_time_end + T // n_members)]
    # val loader iterates all members per timestep; take every n_members-th sample for opa0
    sample_dates = np.array([val_dates[i // n_members] for i in range(T)])

    def compute_and_plot(preds, targets, title, save_path):
        preds_flat = preds.reshape(T, -1)
        targets_flat = targets.reshape(T, -1)
        preds_ocean = np.nan_to_num(preds_flat[:, ocean_mask_flat], nan=0.0)
        targets_ocean = np.nan_to_num(targets_flat[:, ocean_mask_flat], nan=0.0)
        # pp = concept_preds[ci][li]   # (N, Y, X)
        # tt = concept_targets[ci][li]

        # Vectorized Pearson r over time axis for all pixels at once
        # pp_m = np.nanmean(pp, axis=0, keepdims=True)
        # tt_m = np.nanmean(tt, axis=0, keepdims=True)
        # pp_d = pp - pp_m
        # tt_d = tt - tt_m
        # num = np.nansum(pp_d * tt_d, axis=0)
        # denom = np.sqrt(np.nansum(pp_d ** 2, axis=0) * np.nansum(tt_d ** 2, axis=0))
        # corr_map = np.where((denom > 0) & ocean_mask, num / denom, np.nan)
        
        r = pearsonr(preds_ocean, targets_ocean, axis=0)
        r_map = np.full(Y * X, np.nan)
        r_map[ocean_mask_flat] = np.abs(r.statistic)
        r_spatial = r_map.reshape(Y, X)
        r_pearsonr_t = pearsonr(preds_ocean.T, targets_ocean.T, axis=0)
        r_t = np.abs(r_pearsonr_t.statistic)
        mean_r = np.mean(r_t)
        median_r = np.median(r_t)
        max_r = np.max(r_t)
        min_r = np.min(r_t)
        dip_idx = np.where(r_t < mean_r)[0]
        print(f'\n{title}')
        print(f'Mean r = {mean_r:.4f}')
        print(f'Median r = {median_r:.4f}')
        print(f'Max r = {max_r:.4f}')
        print(f'Min r = {min_r:.4f}')
        print(f'Dips (r < mean):')
        for i in dip_idx:
            print(f'  sample {i}: {sample_dates[i].strftime("%Y-%m")}  r={r_t[i]:.4f}')
        r_spatial_masked = np.ma.masked_where(~ocean_mask_flat.reshape(Y, X), r_spatial)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].set_title('Spatial Pearson $r$ (averaged over time)')
        im = ax[0].imshow(r_spatial_masked, origin='lower', cmap='RdYlBu', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax[0])
        ax[1].plot(sample_dates, r_t)
        ax[1].axhline(mean_r, color='r', linestyle='--', label=f'mean={mean_r:.3f}')
        ax[1].scatter(sample_dates[dip_idx], r_t[dip_idx], color='red', zorder=5, s=20)
        ax[1].set_title('Pattern Correlation per Timestep')
        ax[1].set_ylabel('Pearson $r$')
        ax[1].set_xlabel('Date')
        ax[1].legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    compute_and_plot(preds, targets, 'Pearson Correlation Coefficient on Validation', f'{model_dir}/pearsonr_abs.png')
    n_concepts = concept_preds.shape[0]
    for i in range(n_concepts):
        compute_and_plot(concept_preds[i], concept_targets[i], f'Pearson Correlation Coefficient on Validation for {concept_names[i]}', f'{model_dir}/pearsonr_abs_{concept_names[i]}.png')

# comparing free pred monthly climatology with predicted and target mlhc
def compare_free_pred(model_dir):
    month_names = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]
    results = np.load(f'{model_dir}/val_preds_lead0.npz')
    
    mlhc = results['preds'].reshape(46, 5, 302, 400)
    free = results['free_preds'].reshape(46, 5, 302, 400)
    ocean_mask = results['ocean_mask']
    land_mask = ~ocean_mask

    # Global normalization
    mlhc_mean, mlhc_std = mlhc.mean(), mlhc.std()
    free_mean, free_std = free.mean(), free.std()

    months = (np.arange(46) + 5 - 1) % 12
    
    fig1, ax1 = plt.subplots(4, 3, figsize=(12, 10))
    fig2, ax2 = plt.subplots(4, 3, figsize=(12, 10))
    fig3, ax3 = plt.subplots(4, 3, figsize=(12, 10))
    
    fig1.suptitle('Free Concept Monthly Climatology (Normalized)')
    fig2.suptitle('MLHC Predicted Monthly Climatology (Normalized)')
    fig3.suptitle('Bias (MLHC - Free) Monthly Climatology')
    
    ax1 = ax1.flatten()
    ax2 = ax2.flatten()
    ax3 = ax3.flatten()

    for i in range(12):
        month_idx = np.where(months == i)[0]
        
        month_free = ((free[month_idx] - free_mean) / free_std).mean(axis=(0, 1))
        month_mlhc = ((mlhc[month_idx] - mlhc_mean) / mlhc_std).mean(axis=(0, 1))
        month_bias = month_mlhc - month_free

        r = pearsonr(month_free[ocean_mask], month_mlhc[ocean_mask])[0]

        # Use the same vmin/vmax for Climatology
        im1 = ax1[i].imshow(np.ma.masked_where(land_mask, month_free), origin='lower', vmin=-2, vmax=2)
        im2 = ax2[i].imshow(np.ma.masked_where(land_mask, month_mlhc), origin='lower', vmin=-2, vmax=2)
        
        # Use diverging for Bias
        im3 = ax3[i].imshow(np.ma.masked_where(land_mask, month_bias), origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)

        ax1[i].set_title(f'{month_names[i]} | r = {r:.2f}')
        ax2[i].set_title(month_names[i])
        ax3[i].set_title(month_names[i])

        for ax in [ax1[i], ax2[i], ax3[i]]:
            ax.set_xticks([])
            ax.set_yticks([])

    # Add shared colorbars to the right of each figure
    fig1.colorbar(im1, ax=ax1.tolist(), shrink=0.8, label='Std Dev ($\sigma$)')
    fig2.colorbar(im2, ax=ax2.tolist(), shrink=0.8, label='Std Dev ($\sigma$)')
    fig3.colorbar(im3, ax=ax3.flatten().tolist(), shrink=0.8, label='Bias ($\Delta \sigma$)')

    fig1.savefig(f'{model_dir}/free_pred_clim.png', bbox_inches='tight')
    fig2.savefig(f'{model_dir}/mlhc_pred_clim.png', bbox_inches='tight')
    fig3.savefig(f'{model_dir}/bias_clim.png', bbox_inches='tight')

# helper function to caluclate monthly and seasonal acc 
def calculate_correlation(p_val, t_val, t_all, val_months, all_months, label):
    months_names = ["January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December"]
    
    anom_store = {'pred': {}, 'target': {}}
    
    print(f"\n=== Evaluating: {label} ===")
    
    # monthly Loop
    for i in range(12):
        # climatology from FULL dataset
        m_clim = np.nanmean(t_all[all_months == i], axis=0)
        
        # validation slices and calculate anomalies
        mask = (val_months == i)
        m_p_anom = p_val[mask] - m_clim
        m_t_anom = t_val[mask] - m_clim
        
        anom_store['pred'][i] = m_p_anom
        anom_store['target'][i] = m_t_anom
        
        # pearson R
        m_r = pearsonr(m_p_anom, m_t_anom, axis=0)
        print(f"{months_names[i]} - Mean ACC: {np.nanmean(m_r.statistic):.4f}")

    # seasonal pooling
    seasons = {"DJF": [11, 0, 1], "MAM": [2, 3, 4], "JJA": [5, 6, 7], "SON": [8, 9, 10]}
    print(f"\n--- Seasonal ACC (Pooled) for {label} ---")
    for sea_name, m_indices in seasons.items():
        s_p = np.concatenate([anom_store['pred'][m] for m in m_indices], axis=0)
        s_t = np.concatenate([anom_store['target'][m] for m in m_indices], axis=0)
        s_r = pearsonr(s_p, s_t, axis=0)
        print(f"{sea_name} - Mean ACC: {np.nanmean(s_r.statistic):.4f}")

# acc for specific model
def model_acc(model_dir):
    results = np.load(f'{model_dir}/all_preds.npz', allow_pickle=True)
    
    # getting main target
    out_mean, out_std = results['output_mean'], results['output_std']
    preds = results['preds'] * out_std + out_mean
    targets = results['targets'] * out_std + out_mean

    # getting concepts
    c_preds = results['concept_preds']
    c_targets = results['concept_targets']
    c_names = results['concept_names']

    val_start = 1852
    all_months = np.tile((np.arange(463) + 6) % 12, 5)
    val_months = all_months[val_start:]

    # corr for mlhc
    calculate_correlation(preds[val_start:], targets[val_start:], targets, 
                          val_months, all_months, "MLHC Output")

    # corr for each concept
    for j, name in enumerate(c_names):
        p_val_c = c_preds[j, val_start:]
        t_val_c = c_targets[j, val_start:]
        t_all_c = c_targets[j, :]
        
        calculate_correlation(p_val_c, t_val_c, t_all_c, 
                              val_months, all_months, f"Concept: {name}")

# denormalizing acc calculation helper
def calculate_correlation_denorm(p_val, t_val, t_all, val_months, all_months, label):
    
    months_names = ["January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December"]
    
    anom_store = {'pred': {}, 'target': {}}
    
    print(f"\n=== Evaluating: {label} (Physical Units) ===")
    
    # Monthly Loop
    for i in range(12):
        # 1. Climatology from FULL denormalized dataset
        m_clim = np.nanmean(t_all[all_months == i], axis=0)
        
        # 2. Extract Validation slices and calculate physical anomalies
        mask = (val_months == i)
        m_p_anom = p_val[mask] - m_clim
        m_t_anom = t_val[mask] - m_clim
        
        anom_store['pred'][i] = m_p_anom
        anom_store['target'][i] = m_t_anom
        
        # 3. Monthly Pearson R (Temporal)
        m_r = pearsonr(m_p_anom, m_t_anom, axis=0)
        print(f"{months_names[i]:<10} - Mean ACC: {np.nanmean(m_r.statistic):.4f}")

    # Seasonal Pooling
    seasons = {"DJF": [11, 0, 1], "MAM": [2, 3, 4], "JJA": [5, 6, 7], "SON": [8, 9, 10]}
    print(f"\n--- Seasonal ACC (Pooled) for {label} ---")
    for sea_name, m_indices in seasons.items():
        s_p = np.concatenate([anom_store['pred'][m] for m in m_indices], axis=0)
        s_t = np.concatenate([anom_store['target'][m] for m in m_indices], axis=0)
        s_r = pearsonr(s_p, s_t, axis=0)
        print(f"{sea_name} - Mean ACC: {np.nanmean(s_r.statistic):.4f}")

# acc but for denormalzed concepts and targets 
def model_acc_denorm(model_dir):
    results = np.load(f'{model_dir}/all_preds.npz', allow_pickle=True)
    
    # 1. Denormalize Main MLHC Output
    # Broadcasting (1,) or (1, Y, X) over (Time, Y, X)
    out_mean, out_std = results['output_mean'], results['output_std']
    preds = (results['preds'] * out_std) + out_mean
    targets = (results['targets'] * out_std) + out_mean

    # 2. Denormalize Concepts
    # Shape of concept_preds: (Time, Num_Concepts, Y, X)
    # Shape of concept_mean: (Num_Concepts, 1, 1) or (Num_Concepts, Y, X)
    c_preds_raw = results['concept_preds']
    c_targets_raw = results['concept_targets']
    c_mean = results['concept_mean']
    c_std = results['concept_std']
    c_names = results['concept_names']

    # Indices and slicing
    val_start = 1852
    all_months = np.tile((np.arange(463) + 6) % 12, 5)
    val_months = all_months[val_start:]

    # --- Run for MLHC ---
    calculate_correlation_denorm(preds[val_start:], targets[val_start:], targets, 
                                 val_months, all_months, "MLHC Output")

    # --- Run for each Concept ---
    for j, name in enumerate(c_names):
        # Denormalize specific concept j
        # We index j first, then multiply/add to ensure broadcasting matches (Time, Y, X)
        p_val_c = (c_preds_raw[val_start:, j] * c_std[j]) + c_mean[j]
        t_val_c = (c_targets_raw[val_start:, j] * c_std[j]) + c_mean[j]
        t_all_c = (c_targets_raw[:, j] * c_std[j]) + c_mean[j]
        
        calculate_correlation_denorm(p_val_c, t_val_c, t_all_c, 
                                     val_months, all_months, f"Concept: {name}")

# reconstructing the time series of the validation and test set (averaged spatially, comparing to target)
# do this per opa so the the plot looks smooth
def plot_ensemble_time_series(model_dir, split='val', n_members=5):
    
    # load data
    data_path = os.path.join(model_dir, f'all_preds.npz')
    data = np.load(data_path)
    
    preds = data['preds'][1852:]
    targets = data['targets'][1852:]
    
    # reshape to separate opa since they are interleaved
    # (time_steps, opa, y, x)
    n_steps = len(preds) // n_members
    preds_ens = preds[:n_steps * n_members].reshape(n_steps, n_members, 302, 400)
    targets_ens = targets[:n_steps * n_members].reshape(n_steps, n_members, 302, 400)
    
    # calculate mean over basin
    # resulting shape: (time_steps, n_members)
    basin_preds = np.nanmean(preds_ens, axis=(2, 3))
    basin_targets = np.nanmean(targets_ens, axis=(2, 3))
    
    # calculate ensemble mean
    ens_mean = np.mean(basin_preds, axis=1)
    ens_std = np.std(basin_preds, axis=1)
    # target remains the same for all members at each time step (just opa0)
    target_line = basin_targets[:, 0] 
    
    # plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    
    time_idx = np.arange(len(ens_mean))
    
    # plot the spread 
    ax.fill_between(time_idx, ens_mean - ens_std, ens_mean + ens_std, 
                    color='orange', alpha=0.3, label='Ensemble Spread ($\sigma$)')
    
    # mean prediction
    ax.plot(time_idx, ens_mean, color='darkorange', linewidth=2, label='ConceptBUOY')
    
    # ground truth
    ax.plot(time_idx, target_line, color='blue', linestyle='--', linewidth=1.5, label='ORAS5 Target')
    
    ax.set_title(f"Basin-Averaged MLHC Evolution ({split.capitalize()} Set)", fontsize=14)
    ax.set_ylabel("Mixed Layer Heat Content (Standardized Units)")
    ax.set_xlabel("Months in Test Period")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=False)
    
    plt.tight_layout()
    
    # save
    save_path = os.path.join(model_dir, f'ensemble_ts_{split}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")
    
    return ens_mean, target_line

# same function as above, but for the concepts
def plot_concept_ensemble_ts(model_dir, split='val', n_members=5):
   
    data_path = os.path.join(model_dir, 'all_preds.npz')
    data = np.load(data_path)
    
    # load concept data 
    # the same temporal slice [1852:] as MLHC plot
    c_preds = data[f'concept_preds'][:, 1852:, :, :]
    c_targets = data[f'concept_targets'][:, 1852:, :, :]
    concept_names = data['concept_names']
    
    num_concepts = c_preds.shape[0]
    n_steps = c_preds.shape[1] // n_members
    
    if concept_names is None:
        concept_names = [f"Concept {i}" for i in range(num_concepts)]

    for i in range(num_concepts):
        # same reshaping as above, per concept
        this_c_pred = c_preds[i, :n_steps*n_members].reshape(n_steps, n_members, 302, 400)
        this_c_targ = c_targets[i, :n_steps*n_members].reshape(n_steps, n_members, 302, 400)
        
        # basin average
        basin_p = np.nanmean(this_c_pred, axis=(2, 3))
        basin_t = np.nanmean(this_c_targ, axis=(2, 3))
        
        # mean
        mu = np.mean(basin_p, axis=1)
        sigma = np.std(basin_p, axis=1)
        truth = basin_t[:, 0]
        
        # plotting
        fig, ax = plt.subplots(figsize=(10, 4))
        time_idx = np.arange(len(mu))
        
        ax.fill_between(time_idx, mu - sigma, mu + sigma, color='teal', alpha=0.2, label='Ensemble Spread')
        ax.plot(time_idx, mu, color='teal', linewidth=2, label=f'Predicted {concept_names[i]}')
        ax.plot(time_idx, truth, color='black', linestyle='--', alpha=0.7, label='Target')
        
        ax.set_title(f"Mechanism Reconstruction: {concept_names[i]}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Standardized Concept")
        ax.set_xlabel("Months in Test Period")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, loc='upper left')
        
        plt.tight_layout()
        save_path = os.path.join(model_dir, f'concept_{i}_ts.png')
        plt.savefig(save_path, dpi=300)
        plt.close() # Close to save memory
        print(f"Saved concept plot: {save_path}")


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

def concept_correlation(results_path='/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.5_ep50_lr0.001_bs64_BCELoss_ZScore_v3'):
    results = np.load(f'{results_path}/val_preds_lead0.npz', allow_pickle=True)
    preds = results['preds']
    concept_preds = results['concept_preds']
    concept_names = results['concept_names']
    ocean_mask = results['ocean_mask']
    n = len(concept_names)
    fig, ax = plt.subplots(1, n, figsize=(4*n, 3))
    for i in range(n):
        corr = stats.pearsonr(preds, concept_preds[i, :, :, :], axis=0).statistic
        corr[~ocean_mask] = np.nan
        mean_corr = np.nanmean(corr)
        im = ax[i].imshow(corr, origin='lower', aspect='auto', vmin=-1, vmax=1, cmap='RdYlBu')
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title(f'{concept_names[i]}\n(r = {mean_corr:.2f})') 
    fig.suptitle('Concept correlation with Prediction')
    fig.tight_layout()
    fig.savefig(f'{results_path}/concept_corr')

def concept_weights(model_dir = '/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.5_ep50_lr0.001_bs64_BCELoss_ZScore_v3'):
    config.read(f'{model_dir}/config.ini')
    checkpoints = sorted(glob.glob(f'{model_dir}/UNetCBM_epoch*.pt'),
                       key=lambda p: int(p.split('epoch')[-1].split('.')[0]))
    ckpt = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)
    model = get_model()
    model.load_state_dict(ckpt['model_state_dict'])
    weights = model.output_head[0].weight.squeeze()
    results = np.load(f'{model_dir}/val_preds_lead0.npz', allow_pickle=True)
    concept_names = results['concept_names']
    for name, w in zip(concept_names, weights):
        print(f'{name}: {w.item():.4f}')
    concept_preds = results['concept_preds']
    weights_np = weights.detach().numpy()
    contributions = concept_preds * weights_np[:, None, None, None]
    mean_contrib = contributions.mean(axis=1)
    ocean_mask = results['ocean_mask']
    n = mean_contrib.shape[0]
    fig, ax = plt.subplots(1, n, figsize=(4*n, 3))
    for i in range(n):
        contrib = mean_contrib[i, :, :]
        contrib[~ocean_mask] = np.nan
        im = ax[i].imshow(contrib, origin='lower', aspect='auto', cmap='RdYlBu')
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title(f'{concept_names[i]} weight: {weights_np[i]}') 
    fig.suptitle('Concept contribution to Prediction')
    fig.tight_layout()
    fig.savefig(f'{model_dir}/concept_contrib')

def plot_pred_anomaly(model_dir='/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.15_ep50_lr0.001_bs64_MSELoss_ZScore_v2'):
    results = np.load(f'{model_dir}/val_preds_lead0.npz')
    pred0 = results['preds'][0]
    target0 = results['targets'][0]
    ocean_mask = results['ocean_mask']
    pred0 = np.ma.masked_where(~ocean_mask, pred0)

    # find the member and yr and month for pred0 and target0
    # val preds index 0: member=0%n_members=0, time=0//n_members=0 within val set
    import pandas as pd
    dates = pd.date_range(start=config['DATASET']['start'], end=config['DATASET']['end'], freq='MS')
    window = config.getint('DATASET', 'context_window')
    offset = try_cast(config['DATASET']['offset'])[0]
    n_members = len(try_cast(config['DATASET']['members']))
    n_times = len(dates) - window - offset + 1
    train_time_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n_times)
    target_date = dates[train_time_end + window - 1 + offset]
    member = 0
    yr = target_date.year
    month = target_date.month

    # NA bounds
    lon_bounds=(-80, 20)
    lat_bounds=(20, 66)

    # NA climatology 
    clim = xr.open_dataset('/quobyte/maikesgrp/sanah/climatologies/vomlhc_climatology_1979_2018.nc')
    mask = (
                (clim.nav_lon >= lon_bounds[0]) & (clim.nav_lon <= lon_bounds[1]) &
                (clim.nav_lat >= lat_bounds[0]) & (clim.nav_lat <= lat_bounds[1])
            )
    y_inds = mask.any(dim="x")
    x_inds = mask.any(dim="y")
    clim_na = clim.isel(y=y_inds, x=x_inds)
    clim_na_month = clim_na.sel(month=month).vomlhc.isel(y=slice(0, 302), x=slice(0, 400))

    # calculating anomaly 
    pred0_a = pred0 - clim_na_month 
    target0_a = target0 - clim_na_month

    # threshold with 90th percentile
    thresh = xr.open_dataset(f'/quobyte/maikesgrp/sanah/mlhc_anomaly/mlhc_anomalies_90th_percentile_detrended_opa{member}.nc')
    thresh_na = thresh.isel(y=y_inds, x=x_inds)
    thresh_na_month = thresh_na.sel(time_counter=month).mlhc_anomaly.isel(y=slice(0, 302), x=slice(0, 400))
    
    # binary threshold for pred0_a and target0_a but preserve nans for land 
    pred0_binary = np.where(ocean_mask, (pred0_a > thresh_na_month).astype(float), np.nan)                                                                           
    target0_binary = np.where(ocean_mask, (target0_a > thresh_na_month).astype(float), np.nan)

    target0_masked = np.where(ocean_mask, target0, np.nan)
    pred0_masked = np.where(ocean_mask, np.array(pred0), np.nan)

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    im00 = ax[0, 0].imshow(target0_masked, origin='lower', cmap='RdYlBu_r')
    im01 = ax[0, 1].imshow(pred0_masked, origin='lower', cmap='RdYlBu_r')
    im10 = ax[1, 0].imshow(target0_binary, origin='lower', cmap='RdYlBu_r', vmin=0, vmax=1)
    im11 = ax[1, 1].imshow(pred0_binary, origin='lower', cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(im00, ax=ax[0, 0])
    plt.colorbar(im01, ax=ax[0, 1])
    plt.colorbar(im10, ax=ax[1, 0])
    plt.colorbar(im11, ax=ax[1, 1])
    ax[0, 0].set_title(f'Target vomlhc ({yr}-{month:02d})')
    ax[0, 1].set_title(f'Predicted vomlhc ({yr}-{month:02d})')
    ax[1, 0].set_title('Target binary anomaly')
    ax[1, 1].set_title('Predicted binary anomaly')
    for a in ax.flat:
        a.axis('off')
    fig.tight_layout()
    fig.savefig('vomlhc_anomaly_event_0')

def compare_mlhc_sst():
    # NA crop indices from mesh
    lon_bounds = (-80, 20)
    lat_bounds = (20, 66)
    mesh_ds = xr.open_dataset('/quobyte/maikesgrp/kkringel/oras5/ORCA025/mesh/mesh_mask.nc')
    nav_lon = mesh_ds['nav_lon'].squeeze()
    nav_lat = mesh_ds['nav_lat'].squeeze()
    mask_na = (nav_lon >= lon_bounds[0]) & (nav_lon <= lon_bounds[1]) & (nav_lat >= lat_bounds[0]) & (nav_lat <= lat_bounds[1])
    y_inds = mask_na.any(dim='x')
    x_inds = mask_na.any(dim='y')
    ocean_mask_na = mesh_ds['tmaskutil'].squeeze().isel(y=y_inds, x=x_inds).values == 1

    # sst anomalies
    sst_anom = xr.open_dataset('/quobyte/maikesgrp/sanah/sst_anomaly/sst_anomalies_2010_detrended.nc')
    sst_thresh = xr.open_dataset('/quobyte/maikesgrp/sanah/sst_anomaly/sst_anomalies_90th_percentile_detrended_latest.nc')
    mhw_sst = (sst_anom.isel(y=y_inds, x=x_inds).sst_anomaly.values > sst_thresh.isel(y=y_inds, x=x_inds).sst_anomaly.values).astype(float)
    mhw_sst = np.where(ocean_mask_na[..., None], mhw_sst, np.nan)
    # mlhc anomalies
    mlhc_anom = xr.open_dataset('/quobyte/maikesgrp/sanah/mlhc_anomaly/opa0/mlhc_anomalies_2010_detrended.nc')
    mlhc_thresh = xr.open_dataset('/quobyte/maikesgrp/sanah/mlhc_anomaly/mlhc_anomalies_90th_percentile_detrended_opa0.nc')
    mhw_mlhc = (mlhc_anom.isel(y=y_inds, x=x_inds).mlhc_anomaly.values > mlhc_thresh.isel(y=y_inds, x=x_inds).mlhc_anomaly.values).astype(float)
    mhw_mlhc = np.where(ocean_mask_na[..., None], mhw_mlhc, np.nan)
    # # overlap: 0=no event, 1=sst only, 2=mlhc only, 3=both
    # s = mhw_sst[:, :, 1]
    # m = mhw_mlhc[:, :, 0]
    # overlap = np.zeros_like(s)
    # overlap = np.where((s == 1) & (m == 0), 1, overlap)   # sst only
    # overlap = np.where((s == 0) & (m == 1), 2, overlap)   # mlhc only
    # overlap = np.where((s == 1) & (m == 1), 3, overlap)   # both
    # overlap = np.where(~np.isnan(s), overlap, np.nan)

    # from matplotlib.colors import ListedColormap, BoundaryNorm
    # cmap_overlap = ListedColormap(['#d0d0d0', '#1f77b4', '#ff7f0e', '#d62728'])
    # norm_overlap = BoundaryNorm([0, 1, 2, 3, 4], cmap_overlap.N)

    # # mld
    # loc = config['DATASET']['location']
    # mld = xr.open_zarr(f'{loc}/opa0/somxl010_na.zarr').somxl010
    # mld_month = mld.sel(time_counter='2010-01').values.squeeze()
    # mld_month = np.where(ocean_mask_na, mld_month, np.nan)

    # fig, ax = plt.subplots(1, 4, figsize=(16, 3))
    # ax[0].imshow(mhw_sst[:, :, 0], origin='lower', cmap='Reds', vmin=0, vmax=1)
    # ax[1].imshow(mhw_mlhc[:, :, 0], origin='lower', cmap='Reds', vmin=0, vmax=1)
    # im2 = ax[2].imshow(overlap, origin='lower', cmap=cmap_overlap, norm=norm_overlap)
    # im3 = ax[3].imshow(mld_month, origin='lower', cmap='viridis_r')
    # ax[0].set_title('SST MHW')
    # ax[1].set_title('MLHC MHW')
    # ax[2].set_title('Overlap')
    # ax[3].set_title('MLD (m)')
    # cbar = fig.colorbar(im2, ax=ax[2], ticks=[0.5, 1.5, 2.5, 3.5])
    # cbar.ax.set_yticklabels(['no event', 'SST only', 'MLHC only', 'both'])
    # fig.colorbar(im3, ax=ax[3])
    # fig.suptitle('Comparing MHWs')
    # fig.tight_layout()
    # fig.savefig('comparing_mhw')

    # time series of overlap fractions across all years
    import pandas as pd
    from scipy.stats import pearsonr
    loc = config['DATASET']['location']
    mld_zarr = xr.open_zarr(f'{loc}/opa0/somxl010_na.zarr').somxl010

    frac_sst_only, frac_mlhc_only, frac_both = [], [], []
    mean_sst_anom, mean_mlhc_mld = [], []  # for MLHC/MLD vs SST comparison
    dates = []
    for year in range(1980, 2019):
        sst_anom_yr = xr.open_dataset(f'/quobyte/maikesgrp/sanah/sst_anomaly/sst_anomalies_{year}_detrended.nc')
        mlhc_anom_yr = xr.open_dataset(f'/quobyte/maikesgrp/sanah/mlhc_anomaly/opa0/mlhc_anomalies_{year}_detrended.nc')
        mhw_sst_yr = (sst_anom_yr.isel(y=y_inds, x=x_inds).sst_anomaly.values > sst_thresh.isel(y=y_inds, x=x_inds).sst_anomaly.values).astype(float)
        mhw_sst_yr = np.where(ocean_mask_na[..., None], mhw_sst_yr, np.nan)
        mhw_mlhc_yr = (mlhc_anom_yr.isel(y=y_inds, x=x_inds).mlhc_anomaly.values > mlhc_thresh.isel(y=y_inds, x=x_inds).mlhc_anomaly.values).astype(float)
        mhw_mlhc_yr = np.where(ocean_mask_na[..., None], mhw_mlhc_yr, np.nan)
        mlhc_na = mlhc_anom_yr.isel(y=y_inds, x=x_inds).mlhc_anomaly.values  # (Y, X, 12)
        mld_yr = mld_zarr.sel(time_counter=str(year)).values  # (12, Y, X)
        mld_yr = np.moveaxis(mld_yr, 0, -1)  # (Y, X, 12)
        rho_cp = 1026.0 * 3990.0  # from concepts.py mlhc(): rho=1026, cp=3990 J/(m³·K)
        mlhc_mld_yr = mlhc_na / ((mld_yr + 1e-6) * rho_cp)  # units: K (temperature anomaly proxy)
        sst_na = sst_anom_yr.isel(y=y_inds, x=x_inds).sst_anomaly.values  # (Y, X, 12)
        n_months = mhw_sst_yr.shape[2]
        for t in range(n_months):
            s = mhw_sst_yr[:, :, t]
            m = mhw_mlhc_yr[:, :, t]
            valid = ~np.isnan(s)
            n_valid = valid.sum()
            frac_sst_only.append(np.nansum((s == 1) & (m == 0)) / n_valid)
            frac_mlhc_only.append(np.nansum((s == 0) & (m == 1)) / n_valid)
            frac_both.append(np.nansum((s == 1) & (m == 1)) / n_valid)
            mean_sst_anom.append(np.nanmean(sst_na[:, :, t]))
            mean_mlhc_mld.append(np.nanmean(mlhc_mld_yr[:, :, t]))
            dates.append(pd.Timestamp(f'{year}-{t+1:02d}'))

    # Pearson correlations
    r1, p1 = pearsonr(frac_sst_only, frac_mlhc_only)
    r2, p2 = pearsonr(frac_sst_only, frac_both)
    r3, p3 = pearsonr(frac_mlhc_only, frac_both)
    r4, p4 = pearsonr(mean_sst_anom, mean_mlhc_mld)
    print(f'SST-only vs MLHC-only:  r={r1:.3f}, p={p1:.4f}')
    print(f'SST-only vs Both:       r={r2:.3f}, p={p2:.4f}')
    print(f'MLHC-only vs Both:      r={r3:.3f}, p={p3:.4f}')
    print(f'Mean SST anom vs mean MLHC/MLD: r={r4:.3f}, p={p4:.4f}')

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(dates, frac_sst_only, label='SST only', color='#1f77b4')
    axes[0].plot(dates, frac_mlhc_only, label='MLHC only', color='#ff7f0e')
    axes[0].plot(dates, frac_both, label='Both', color='#d62728')
    axes[0].set_ylabel('Fraction of ocean points')
    axes[0].set_title(f'MHW overlap time series (opa0, NA) | SST vs MLHC r={r1:.3f}')
    axes[0].legend()
    axes[1].plot(dates, mean_sst_anom, label='Mean SST anomaly', color='#1f77b4')
    ax2 = axes[1].twinx()
    ax2.plot(dates, mean_mlhc_mld, label='Mean MLHC/MLD', color='#2ca02c', alpha=0.7)
    axes[1].set_ylabel('SST anomaly (°C)')
    ax2.set_ylabel('MLHC/MLD')
    axes[1].set_title(f'Mean SST anomaly vs MLHC/MLD | r={r4:.3f}')
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1].set_xlabel('Date')
    fig.tight_layout()
    fig.savefig('comparing_mhw_timeseries')

def plot_pearsonr(results_path='/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.15_ep50_lr0.001_bs64_MSELoss_ZScore_v2'):
    import pandas as pd
    results = np.load(f'{results_path}/val_preds_lead0.npz', allow_pickle=True)
    preds, targets, concept_preds, concept_targets, ocean_mask, concept_names = results['preds'], results['targets'], results['concept_preds'], results['concept_targets'], results['ocean_mask'], results['concept_names']
    T, Y, X = preds.shape
    ocean_mask_flat = ocean_mask.reshape(-1)

    # Reconstruct validation dates
    window  = config.getint('DATASET', 'context_window')
    offset  = try_cast(config['DATASET']['offset'])[0]
    n_members = len(try_cast(config['DATASET']['members']))
    dates   = pd.date_range(start=config['DATASET']['start'], end=config['DATASET']['end'], freq='MS')
    n_times = len(dates) - window - offset + 1
    train_time_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n_times)
    val_dates = [dates[t + window - 1 + offset] for t in range(train_time_end, train_time_end + T // n_members)]
    # val loader iterates all members per timestep; take every n_members-th sample for opa0
    sample_dates = np.array([val_dates[i // n_members] for i in range(T)])

    def compute_and_plot(preds, targets, title, save_path):
        preds_flat = preds.reshape(T, -1)
        targets_flat = targets.reshape(T, -1)
        preds_ocean = np.nan_to_num(preds_flat[:, ocean_mask_flat], nan=0.0)
        targets_ocean = np.nan_to_num(targets_flat[:, ocean_mask_flat], nan=0.0)
        r = pearsonr(preds_ocean, targets_ocean, axis=0)
        r_map = np.full(Y * X, np.nan)
        r_map[ocean_mask_flat] = r.statistic
        r_spatial = r_map.reshape(Y, X)
        r_pearsonr_t = pearsonr(preds_ocean.T, targets_ocean.T, axis=0)
        r_t = r_pearsonr_t.statistic
        mean_r = np.mean(r_t)
        dip_idx = np.where(r_t < mean_r)[0]
        print(f'\n{title}')
        print(f'Mean r = {mean_r:.4f}')
        print(f'Dips (r < mean):')
        for i in dip_idx:
            print(f'  sample {i}: {sample_dates[i].strftime("%Y-%m")}  r={r_t[i]:.4f}')
        r_spatial_masked = np.ma.masked_where(~ocean_mask_flat.reshape(Y, X), r_spatial)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].set_title('Spatial Pearson $r$ (averaged over time)')
        im = ax[0].imshow(r_spatial_masked, origin='lower', cmap='RdYlBu', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax[0])
        ax[1].plot(sample_dates, r_t)
        ax[1].axhline(mean_r, color='r', linestyle='--', label=f'mean={mean_r:.3f}')
        ax[1].scatter(sample_dates[dip_idx], r_t[dip_idx], color='red', zorder=5, s=20)
        ax[1].set_title('Pattern Correlation per Timestep')
        ax[1].set_ylabel('Pearson $r$')
        ax[1].set_xlabel('Date')
        ax[1].legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    compute_and_plot(preds, targets, 'Pearson Correlation Coefficient on Validation', f'{results_path}/pearsonr.png')
    n_concepts = concept_preds.shape[0]
    for i in range(n_concepts):
        compute_and_plot(concept_preds[i], concept_targets[i], f'Pearson Correlation Coefficient on Validation for {concept_names[i]}', f'{results_path}/pearsonr_{concept_names[i]}.png')

if __name__ == '__main__':
    # paths = [
    # "/quobyte/maikesgrp/mlhc_cbm/runs_041326/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore",
    # "/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.0_ep101_lr0.001_bs64_L1Loss_ZScore_v4",
    # "/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v5",
    # "/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v6",
    # "/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v7",
    # "/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v8",
    # "/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v9",
    # "/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v10"]
    # for path in paths:
    #     ckpt = ckpt = torch.load(f'{path}/UNetCBM_epoch100.pt', map_location='cpu', weights_only=False)
    #     print(path)
    #     print(ckpt['model_state_dict']['output_head.weight'].squeeze())
    # breakpoint()
    # plot_concept_ensemble_ts('/quobyte/maikesgrp/mlhc_cbm/runs_041326/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore')
    # breakpoint()
    #model_acc('/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v5')
    #plot_concept_ensemble_ts('/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v5')
    #plot_ensemble_time_series('/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v5')
    # breakpoint()
    from torch.utils.data import DataLoader                                                                                                                  

    model_dirs = ['/quobyte/maikesgrp/mlhc_cbm/runs_040726/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v2']
    input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    full_loader = DataLoader(train_loader.dataset.dataset, batch_size=64, shuffle=False, num_workers=0)  
    for model_dir in model_dirs:       
        save_all_preds(model_dir=model_dir, input_norm=input_norm,                                   
                        concept_norm=concept_norm, output_norm=output_norm,                           
                        full_loader=full_loader)               
    # save_all_preds(model_dir=model_dir, input_norm=input_norm,
    #                    concept_norm=concept_norm, output_norm=output_norm,
    #                    full_loader=full_loader)
    

    # model_dirs = ['/quobyte/maikesgrp/mlhc_cbm/trended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore',
    # '/quobyte/maikesgrp/mlhc_cbm/trended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v2',
    # '/quobyte/maikesgrp/mlhc_cbm/trended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v3'
    # ]
    # for model_dir in model_dirs:
    #     pred_clim(model_dir=model_dir)
    #     pred_concept_clim(model_dir=model_dir)
    
    # model_dirs = ['/quobyte/maikesgrp/mlhc_cbm/trended/UNetCBM_lam0.0_ep101_bs64_L1Loss_ZScore_v1',
    # '/quobyte/maikesgrp/mlhc_cbm/trended/UNetCBM_lam0.0_ep101_bs64_L1Loss_ZScore_v2',
    # '/quobyte/maikesgrp/mlhc_cbm/trended/UNetCBM_lam0.0_ep101_bs64_L1Loss_ZScore_v3']
    # for model_dir in model_dirs:
    #     pred_clim(model_dir=model_dir)

    # config.read(f'{model_dirs[0]}/config.ini')
    # input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    # full_loader = DataLoader(train_loader.dataset.dataset, batch_size=64, shuffle=False)

    # for model_dir in model_dirs:
    #     print(f'Saving predictions for {model_dir}')
    #     save_all_preds(model_dir=model_dir, input_norm=input_norm,
    #                    concept_norm=concept_norm, output_norm=output_norm,
    #                    full_loader=full_loader)
    
    #MODEL_DIR = '/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.15_ep50_lr0.001_bs64_MSELoss_ZScore_v2'
    #save_all_preds(model_dir=MODEL_DIR)
    #compute_mhw_events(results_path=MODEL_DIR)
    #compare_mlhc_sst()
    #plot_pred_anomaly()
    #concept_weights()
    #save_val_preds()
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
