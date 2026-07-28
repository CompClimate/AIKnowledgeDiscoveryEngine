import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import xarray as xr
import pandas as pd
from utils.get_data import get_dataset
from utils.get_config import config, try_cast, get_model
from utils.visualization import find_output_dir, plot_sample, visualize
from scipy import stats, signal
from scipy.stats import pearsonr
import cartopy.crs as ccrs
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
        full_loader = DataLoader(full_dataset, batch_size=8, shuffle=False)

    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/{config.get("DATASET", "tmask_name", fallback="tmask_g")}.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0).values
    sy = config.getint('DATASET', 'spatial_y', fallback=0)
    sx = config.getint('DATASET', 'spatial_x', fallback=0)
    if sy > 0 and sx > 0:
        mask_2d = mask_2d[:sy, :sx]
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
            pred, cpred, *_ = model(batch)
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
    mesh = xr.open_zarr(f'{loc}/{config.get("DATASET", "tmask_name", fallback="tmask_g")}.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0).values
    sy = config.getint('DATASET', 'spatial_y', fallback=0)
    sx = config.getint('DATASET', 'spatial_x', fallback=0)
    if sy > 0 and sx > 0:
        mask_2d = mask_2d[:sy, :sx]
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
    mesh_ds = xr.open_dataset('/path/to/data/oras5/ORCA025/mesh/mesh_mask.nc')
    nav_lon = mesh_ds['nav_lon'].squeeze()
    nav_lat = mesh_ds['nav_lat'].squeeze()
    mask_na = (nav_lon >= lon_bounds[0]) & (nav_lon <= lon_bounds[1]) & (nav_lat >= lat_bounds[0]) & (nav_lat <= lat_bounds[1])
    y_inds = mask_na.any(dim='x')
    x_inds = mask_na.any(dim='y')
    ocean_mask_na = mesh_ds['tmaskutil'].squeeze().isel(y=y_inds, x=x_inds).values == 1

    # sst anomalies
    sst_anom = xr.open_dataset('/path/to/data/sst_anomaly/sst_anomalies_2010_detrended.nc')
    sst_thresh = xr.open_dataset('/path/to/data/sst_anomaly/sst_anomalies_90th_percentile_detrended_latest.nc')
    mhw_sst = (sst_anom.isel(y=y_inds, x=x_inds).sst_anomaly.values > sst_thresh.isel(y=y_inds, x=x_inds).sst_anomaly.values).astype(float)
    mhw_sst = np.where(ocean_mask_na[..., None], mhw_sst, np.nan)
    # mlhc anomalies
    mlhc_anom = xr.open_dataset('/path/to/data/mlhc_anomaly/opa0/mlhc_anomalies_2010_detrended.nc')
    mlhc_thresh = xr.open_dataset('/path/to/data/mlhc_anomaly/mlhc_anomalies_90th_percentile_detrended_opa0.nc')
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
        sst_anom_yr = xr.open_dataset(f'/path/to/data/sst_anomaly/sst_anomalies_{year}_detrended.nc')
        mlhc_anom_yr = xr.open_dataset(f'/path/to/data/mlhc_anomaly/opa0/mlhc_anomalies_{year}_detrended.nc')
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
def compare_free_pred(model_dirs, nc_path=None, domain_lat=(20, 66), domain_lon=(-80, 20)):
    if isinstance(model_dirs, str):
        model_dirs = [model_dirs]

    first = np.load(f'{model_dirs[0]}/val_preds_lead0.npz')
    ocean_mask = first['ocean_mask']
    land_mask = ~ocean_mask
    Y, X = ocean_mask.shape

    all_mlhc, all_free = [], []
    all_free_c = []
    for md in model_dirs:
        r = np.load(f'{md}/val_preds_lead0.npz')
        n_members = 5
        N = r['preds'].shape[0]
        n_steps = N // n_members
        all_mlhc.append(r['preds'].reshape(n_steps, n_members, Y, X))
        all_free.append(r['free_preds'].reshape(n_steps, n_members, Y, X))
        ckpt = torch.load(f'{md}/UNetCBM_epoch100.pt', map_location='cpu', weights_only=False)
        all_free_c.append(ckpt['model_state_dict']['output_head.weight'].squeeze().numpy()[-1])

    mlhc = np.mean(all_mlhc, axis=0)
    free_c = float(np.mean(all_free_c))
    # Multiply by free_c so sign reflects contribution direction to MLHC output.
    # free_preds is in bottleneck space; free_c orients it correctly (may be negative).
    free = np.mean(all_free, axis=0) #* free_c

    mlhc_mean, mlhc_std = mlhc.mean(), mlhc.std()
    free_mean, free_std = free.mean(), free.std()

    months = (np.arange(n_steps) + 5 - 1) % 12
    seasons = {"DJF": [11, 0, 1], "MAM": [2, 3, 4], "JJA": [5, 6, 7], "SON": [8, 9, 10]}

    season_biases = {}
    for sea_name, m_indices in seasons.items():
        idx = np.concatenate([np.where(months == m)[0] for m in m_indices])
        # Both normalized to their own σ so units are comparable (relative σ).
        # preds is denormalized physical MLHC; free*free_c is in normalized output space —
        # separate σ normalization bridges that gap while preserving spatial sign.
        sea_free = ((free[idx] - free_mean) / free_std).mean(axis=(0, 1))
        sea_mlhc = ((mlhc[idx] - mlhc_mean) / mlhc_std).mean(axis=(0, 1))
        season_biases[sea_name] = sea_mlhc - sea_free

    vmax = max(np.abs(b[ocean_mask]).max() for b in season_biases.values())

    data_proj = ccrs.PlateCarree()
    map_proj  = ccrs.Mercator()

    # load nav coords if available
    if nc_path is not None:
        ds_nc = xr.open_dataset(nc_path)
        domain_mask = (
            (ds_nc['nav_lat'] >= domain_lat[0]) & (ds_nc['nav_lat'] <= domain_lat[1]) &
            (ds_nc['nav_lon'] >= domain_lon[0]) & (ds_nc['nav_lon'] <= domain_lon[1])
        )
        y_crop = np.where(domain_mask.any(dim='x'))[0]
        x_crop = np.where(domain_mask.any(dim='y'))[0]
        nav_lat = ds_nc['nav_lat'].isel(y=y_crop, x=x_crop).values[:Y, :X]
        nav_lon = ds_nc['nav_lon'].isel(y=y_crop, x=x_crop).values[:Y, :X]
    else:
        nav_lat = nav_lon = None

    cmap = plt.get_cmap('PiYG').copy()
    cmap.set_bad(color='lightgray')
    cbar_label = r'MLHC Pred $-$ Free Concept ($\sigma$)'

    def _plot_season(ax, sea_name, bias):
        masked = np.ma.masked_where(land_mask, bias)
        im = ax.pcolormesh(nav_lon, nav_lat, masked,
                           transform=data_proj, cmap=cmap, vmin=-vmax, vmax=vmax, zorder=1)
        ax.set_facecolor('lightgray')
        ax.contourf(nav_lon, nav_lat, land_mask.astype(float),
                    levels=[0.5, 1.5], colors=['lightgray'], transform=data_proj, zorder=2)
        ax.contour(nav_lon, nav_lat, land_mask.astype(float),
                   levels=[0.5], colors='k', linewidths=0.4, transform=data_proj, zorder=3)
        ax.set_extent([-80, 20, 20, 66], crs=data_proj)
        ax.set_title(sea_name)
        return im

    # DJF + JJA only, colorbar on the right
    fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4),
                               subplot_kw={'projection': map_proj},
                               layout='constrained')
    for ax, sea_name in zip(axes2, ['DJF', 'JJA']):
        im2 = _plot_season(ax, sea_name, season_biases[sea_name])
    fig2.colorbar(im2, ax=axes2.tolist(), orientation='vertical', shrink=0.8,
                  pad=0.02, label=cbar_label)
    fig2.savefig('paper_figs/free_pred_seasonal_bias_djf_jja.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # All 4 seasons, colorbar on the bottom
    fig4, axes4 = plt.subplots(1, 4, figsize=(16, 4),
                               subplot_kw={'projection': map_proj},
                               layout='constrained')
    for ax, (sea_name, bias) in zip(axes4, season_biases.items()):
        im4 = _plot_season(ax, sea_name, bias)
    fig4.colorbar(im4, ax=axes4.tolist(), orientation='horizontal', shrink=0.6,
                  pad=0.02, label=cbar_label)
    fig4.savefig('paper_figs/free_pred_seasonal_bias.png', dpi=200, bbox_inches='tight')
    plt.close(fig4)


# helper function to caluclate monthly and seasonal acc 
def calculate_correlation(p_val, t_val, t_all, val_months, all_months, label,
                          ocean_mask=None, nav_lat=None, nav_lon=None, save_dir=None):
    concept_dict = {'vori': 'Richardson number', 'vos2': 'Vertical shear', 'von2': 'Buoyancy frequency',
    'vohfe': 'Heat flux entrainment', 'mxl_tendency': 'Mixed layer depth tendency', 'MLHC': 'Mixed layer heat content'}
    months_names = ["January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December"]
    
    p_val  = np.nan_to_num(p_val,  nan=0.0)
    t_val  = np.nan_to_num(t_val,  nan=0.0)
    t_all  = np.nan_to_num(t_all,  nan=0.0)

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
        stat = s_r.statistic.copy()
        if ocean_mask is not None:
            stat[~ocean_mask] = np.nan
        mean_acc = np.nanmean(stat[ocean_mask] if ocean_mask is not None else stat)
        data_proj = ccrs.PlateCarree()
        map_proj  = ccrs.Robinson()
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': map_proj})
        import cartopy.feature as cfeature
        cmap = plt.get_cmap('RdBu').copy()
        cmap.set_bad(color='lightgray')
        land_mask = ~ocean_mask if ocean_mask is not None else np.zeros_like(stat, dtype=bool)
        # mask cells where the tripolar fold causes longitude discontinuities
        lon_wrap = np.zeros(stat.shape, dtype=bool)
        lon_wrap[:, :-1] |= np.abs(np.diff(nav_lon, axis=1)) > 90
        lon_wrap[:-1, :] |= np.abs(np.diff(nav_lon, axis=0)) > 90
        masked = np.ma.masked_where(land_mask | lon_wrap, stat)
        ax.set_facecolor('lightgray')
        im = ax.pcolormesh(nav_lon, nav_lat, masked,
                           transform=data_proj, cmap=cmap, vmin=-1, vmax=1, zorder=1)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)
        ax.set_global()
        plt.colorbar(im, ax=ax, label='Pearson r')
        ax.set_title(f'{concept_dict[label]} | {sea_name} ACC: {mean_acc:.2f}')
        plt.tight_layout()
        out_path = os.path.join(save_dir, f'{label}_{sea_name}_acc.png') if save_dir else f'{label}_{sea_name}_acc.png'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"{sea_name} - Mean ACC: {mean_acc:.4f}")

# acc for specific model(s)
def model_acc(model_dirs, nc_path=None, domain_lat=(20, 66), domain_lon=(-80, 20), save_dir=None):
    print('in it!')
    all_preds = []
    all_c_preds = []

    # targets and metadata from the first model once
    first_results = np.load(f'{model_dirs[0]}/all_preds.npz', allow_pickle=True)
    out_mean, out_std = first_results['output_mean'], first_results['output_std']
    ocean_mask = first_results['ocean_mask']
    Y, X = ocean_mask.shape

    nav_lat, nav_lon = None, None
    if nc_path is not None:
        ds_nc = xr.open_dataset(nc_path)
        domain_mask = (
            (ds_nc['nav_lat'] >= domain_lat[0]) & (ds_nc['nav_lat'] <= domain_lat[1]) &
            (ds_nc['nav_lon'] >= domain_lon[0]) & (ds_nc['nav_lon'] <= domain_lon[1])
        )
        #domain_mask = ((ds_nc.nav_lat >= -60) & (ds_nc.nav_lat <= 60) & ((ds_nc.nav_lon >= 120) | (ds_nc.nav_lon <= -75)))
        y_crop = np.where(domain_mask.any(dim='x'))[0]
        x_crop = np.where(domain_mask.any(dim='y'))[0]
        nav_lat = ds_nc['nav_lat'].isel(y=y_crop, x=x_crop).values[:Y, :X]
        nav_lon = ds_nc['nav_lon'].isel(y=y_crop, x=x_crop).values[:Y, :X]

    targets = first_results['targets'] * out_std + out_mean
    c_targets = first_results['concept_targets']
    c_names = first_results['concept_names']

    # predictions from all models in the ensemble
    for model_dir in model_dirs:
        results = np.load(f'{model_dir}/all_preds.npz', allow_pickle=True)
        # Rescale predictions using the standard normalization parameters
        m_preds = results['preds'] * out_std + out_mean
        m_c_preds = results['concept_preds']

        all_preds.append(m_preds)
        all_c_preds.append(m_c_preds)
        print(f'done {model_dir}')

    # ensemble by averaging across the model dimension (axis 0)
    preds = np.mean(all_preds, axis=0)
    c_preds = np.mean(all_c_preds, axis=0)

    # validation indexing
    val_start = 1852
    all_months = np.tile((np.arange(463) + 6) % 12, 5)
    val_months = all_months[val_start:]

    # correlation for the final MLHC Output
    calculate_correlation(preds[val_start:], targets[val_start:], targets,
                          val_months, all_months, "MLHC",
                          ocean_mask=ocean_mask, nav_lat=nav_lat, nav_lon=nav_lon, save_dir=save_dir)

    # correlation for each physical concept
    for j, name in enumerate(c_names):
        p_val_c = c_preds[j, val_start:]
        t_val_c = c_targets[j, val_start:]
        t_all_c = c_targets[j, :]

        calculate_correlation(p_val_c, t_val_c, t_all_c,
                              val_months, all_months, name,
                              ocean_mask=ocean_mask, nav_lat=nav_lat, nav_lon=nav_lon, save_dir=save_dir)


# reconstructing the time series of the validation and test set (averaged spatially, comparing to target)
# do this per opa so the the plot looks smooth
def plot_ml_ensemble(model_dirs_cbm, model_dirs_no_cbm=None, model_dirs_no_free=None, split='val', n_members=5, save_dir=None):
    COLORS = {
        'target':   '#333333',
        'cbm':      '#2a9d8f',  # teal  — OceanCBM (free1)
        'no_free':  '#e76f51',  # coral — prescription-only (free0)
        'no_cbm':   '#9b5de5',  # purple — no-concept baseline (unsup)
    }

    def _collect(model_dirs):
        all_pred_anoms = []
        target_anom = None

        for model_dir in model_dirs:
            data = np.load(os.path.join(model_dir, 'all_preds.npz'), allow_pickle=True)
            out_std    = data['output_std']
            out_mean   = data['output_mean']
            ocean_mask = data['ocean_mask']  # (Y, X) bool

            # denormalize to physical units (W m⁻²)
            targets_all = data['targets'] * out_std + out_mean
            preds_all   = data['preds']   * out_std + out_mean

            N_total   = len(targets_all)
            n_total_t = N_total // n_members

            Y, X = ocean_mask.shape
            targets_r = targets_all[:n_total_t * n_members].reshape(n_total_t, n_members, Y, X)
            preds_r   = preds_all[:n_total_t * n_members].reshape(n_total_t, n_members, Y, X)

            # mask land before spatial mean
            targets_r = np.where(ocean_mask[None, None], targets_r, np.nan)
            preds_r   = np.where(ocean_mask[None, None], preds_r,   np.nan)

            targets_ts = np.nanmean(targets_r, axis=(1, 2, 3))  # (n_total_t,)
            preds_ts   = np.nanmean(preds_r,   axis=(1, 2, 3))

            # val/test slice (sample 1852 onward = time step 370 onward)
            val_start_t = 1852 // n_members
            n_steps     = (N_total - 1852) // n_members

            val_t = targets_ts[val_start_t : val_start_t + n_steps]
            val_p = preds_ts[val_start_t   : val_start_t + n_steps]

            all_pred_anoms.append(val_p)
            if target_anom is None:
                target_anom = val_t

        mean = np.mean(all_pred_anoms, axis=0)
        std  = np.std(all_pred_anoms,  axis=0)
        return mean, std, target_anom

    cbm_mean, cbm_std, target_mean = _collect(model_dirs_cbm)
    r_cbm = pearsonr(cbm_mean, target_mean)
    print(f'CBM (free1)          r={r_cbm.statistic:.4f}  p={r_cbm.pvalue:.4g}')

    if model_dirs_no_free is not None:
        no_free_mean, no_free_std, _ = _collect(model_dirs_no_free)
        r_no_free = pearsonr(no_free_mean, target_mean)
        print(f'Prescription-only baseline  r={r_no_free.statistic:.4f}  p={r_no_free.pvalue:.4g}')

    if model_dirs_no_cbm is not None:
        no_cbm_mean, no_cbm_std, _ = _collect(model_dirs_no_cbm)
        r_no_cbm = pearsonr(no_cbm_mean, target_mean)
        print(f'Prediction-only baseline  r={r_no_cbm.statistic:.4f}  p={r_no_cbm.pvalue:.4g}')

    fig, ax = plt.subplots(figsize=(6, 3))
    time_idx    = np.arange(len(cbm_mean))
    date_range  = pd.date_range(start="2010-05-01", periods=len(cbm_mean), freq='MS')
    date_labels = [d.strftime('%Y') for d in date_range]

    ax.plot(time_idx, target_mean, color=COLORS['target'], lw=0.5, linestyle='--', label='ORAS5')

    if model_dirs_no_cbm is not None:
        ax.fill_between(time_idx, no_cbm_mean - no_cbm_std, no_cbm_mean + no_cbm_std,
                        color=COLORS['no_cbm'], alpha=0.25)
        ax.plot(time_idx, no_cbm_mean, color=COLORS['no_cbm'], linewidth=2, label='Prediction-only')

    if model_dirs_no_free is not None:
        ax.fill_between(time_idx, no_free_mean - no_free_std, no_free_mean + no_free_std,
                        color=COLORS['no_free'], alpha=0.25)
        ax.plot(time_idx, no_free_mean, color=COLORS['no_free'], linewidth=2, label='Prescription-only')

    ax.fill_between(time_idx, cbm_mean - 2*cbm_std, cbm_mean + 2*cbm_std,
                    color=COLORS['cbm'], alpha=0.3)
    ax.plot(time_idx, cbm_mean, color=COLORS['cbm'], lw=0.5, label='OceanCBM')

    ax.set_xticks(time_idx[::12])
    ax.set_xticklabels(date_labels[::12], rotation=0)
    #ax.set_title("Spatially averaged MLHC time series")
    ax.set_ylabel("MLHC (J m⁻²)")
    ax.set_xlabel("Date")
    leg = ax.legend(loc='upper right', frameon=True, framealpha=0.5,
                    facecolor='#e8e8e8', edgecolor='#aaaaaa')
    leg.get_frame().set_linewidth(0.8)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"paper_figs/concepts_4/{save_dir}/time_series_pred.pdf")

    return cbm_mean, target_mean

def plot_concept_ensemble_ts(model_dirs, model_dirs_no_free=None, split='val', n_members=5, save_dir=None):
    concept_dict = {'vori': 'Richardson number', 'vos2': 'Vertical shear', 'von2': 'Buoyancy frequency',
    'vohfe': 'Heat flux entrainment', 'mxl_tendency': 'Mixed layer depth tendency', 'MLHC': 'Mixed layer heat content'}
    concept_units = {'vori': 'Dimensionless', 'vos2': 's⁻²', 'von2': 's⁻²',
    'vohfe': 'W m⁻²', 'mxl_tendency': 'm/month', 'MLHC': 'J m⁻²'}

    COLORS = {
        'target':   '#333333',
        'cbm':      '#2a9d8f',  # teal  — OceanCBM (free1)
        'no_free':  '#e76f51',  # coral — prescription-only (free0)
    }

    def _collect_concept(dirs, concept_idx):
        all_preds_ts = []
        target_ts = None
        for model_dir in dirs:
            data = np.load(os.path.join(model_dir, 'all_preds.npz'), allow_pickle=True)
            ocean_mask = data['ocean_mask']
            c_mean     = data['concept_mean'][concept_idx]
            c_std      = data['concept_std'][concept_idx]
            c_preds_all   = data['concept_preds'][concept_idx]   * c_std + c_mean
            c_targets_all = data['concept_targets'][concept_idx] * c_std + c_mean
            N_total   = len(c_preds_all)
            n_total_t = N_total // n_members
            val_start_t = 1852 // n_members
            n_steps     = (N_total - 1852) // n_members
            Y, X = ocean_mask.shape
            c_preds_r = c_preds_all[:n_total_t * n_members].reshape(n_total_t, n_members, Y, X)
            c_targs_r = c_targets_all[:n_total_t * n_members].reshape(n_total_t, n_members, Y, X)
            c_preds_r = np.where(ocean_mask[None, None], c_preds_r, np.nan)
            c_targs_r = np.where(ocean_mask[None, None], c_targs_r, np.nan)
            c_preds_ts  = np.nanmean(c_preds_r, axis=(1, 2, 3))
            c_targs_ts  = np.nanmean(c_targs_r, axis=(1, 2, 3))
            all_preds_ts.append(c_preds_ts[val_start_t : val_start_t + n_steps])
            if target_ts is None:
                target_ts = c_targs_ts[val_start_t : val_start_t + n_steps]
        mean = np.mean(all_preds_ts, axis=0)
        std  = np.std(all_preds_ts,  axis=0)
        return mean, std, target_ts

    # load concept names from first model
    first_data = np.load(os.path.join(model_dirs[0], 'all_preds.npz'))
    concept_names = first_data['concept_names']

    for i, cname in enumerate(concept_names):
        name = concept_dict[cname]

        cbm_mean, cbm_std, target_c_mean = _collect_concept(model_dirs, i)
        r_cbm = pearsonr(cbm_mean, target_c_mean)
        print(f'{cname} OceanCBM          r={r_cbm.statistic:.4f}')

        if model_dirs_no_free is not None:
            nf_mean, nf_std, _ = _collect_concept(model_dirs_no_free, i)
            r_nf = pearsonr(nf_mean, target_c_mean)
            print(f'{cname} Prescription-only  r={r_nf.statistic:.4f}')

        fig, ax = plt.subplots(figsize=(6, 3))
        time_idx   = np.arange(len(cbm_mean))
        date_range = pd.date_range(start="2010-05-01", periods=len(cbm_mean), freq='MS')
        date_labels = [d.strftime('%Y') for d in date_range]

        ax.plot(time_idx, target_c_mean, color=COLORS['target'], linestyle='--',
                linewidth=0.5, label='ORAS5')

        if model_dirs_no_free is not None:
            ax.fill_between(time_idx, nf_mean - nf_std, nf_mean + nf_std,
                            color=COLORS['no_free'], alpha=0.2)
            ax.plot(time_idx, nf_mean, color=COLORS['no_free'], linewidth=2,
                    label='Prescription-only')

        ax.fill_between(time_idx, cbm_mean - 2*cbm_std, cbm_mean + 2*cbm_std,
                        color=COLORS['cbm'], alpha=0.3)
        ax.plot(time_idx, cbm_mean, color=COLORS['cbm'], lw=0.5, label='OceanCBM')

        ax.set_xticks(time_idx[::12])
        ax.set_xticklabels(date_labels[::12], rotation=0)
        #ax.set_title(f"Spatially averaged {name} time series")
        ax.set_ylabel(f"{concept_units[cname]}")
        ax.set_xlabel("Date")
        leg = ax.legend(loc='upper right', frameon=True, framealpha=0.5,
                        facecolor='#e8e8e8', edgecolor='#aaaaaa')
        leg.get_frame().set_linewidth(0.8)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        save_path = f"paper_figs/concepts_4/{save_dir}/{cname}_ts.pdf"
        fig.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved ensemble concept plot: {save_path}")


if __name__ == '__main__':
    output_dir = '/path/to/data/global_cbm/UNetCBM_adaptive_ep101_lr0.001_bs64_L1Loss_ZScore_v1'
    nc_path = '/path/to/data/oras5/somxl010/opa0/somxl010_ORAS5_1m_199812_grid_T_02.nc'
    #visualize(output_dir)
    # input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    # from torch.utils.data import DataLoader  
    # full_loader = DataLoader(train_loader.dataset.dataset, batch_size=8, shuffle=False, num_workers=0) 
    # save_all_preds(model_dir=output_dir, input_norm=input_norm, concept_norm=concept_norm,                
    #                    output_norm=output_norm, output_dir=output_dir, full_loader=full_loader) 
    model_acc([output_dir], nc_path=nc_path, domain_lat=(-90, 90), domain_lon=(-180, 180), save_dir=output_dir)
    
    # for i in [4]:
    #     # Use list comprehensions to iterate through versions 1 to 5
    #     dirs_free1 = [f'/path/to/data/paper_cbm/concepts_{i}/free1/UNetCBM_adaptive_ep101_lr0.001_bs64_L1Loss_ZScore_v{v}' for v in range(1, 6)]

    #     dirs_free0 = [f'/path/to/data/paper_cbm/concepts_{i}/free0/UNetCBM_adaptive_ep101_lr0.001_bs64_L1Loss_ZScore_v{v}' for v in range(1, 6)]

    #     dirs_unsup = [f'/path/to/data/paper_cbm/concepts_{i}/unsup/UNetCBM_lam0_ep101_lr0.001_bs64_L1Loss_ZScore_v{v}' for v in range(1, 6)]

    #     #compare_free_pred(model_dirs=dirs_free1, nc_path=nc_path)
    #     # Run accuracy checks
    #     # model_acc(model_dirs=dirs_free1, nc_path=nc_path, save_dir='free1')
    #     # model_acc(model_dirs=dirs_free0, nc_path=nc_path, save_dir='free0')
    #     # model_acc(model_dirs=dirs_unsup, nc_path=nc_path, save_dir='unsup')

    #     # Time series 
    #     #plot_ml_ensemble(model_dirs_cbm=dirs_free1, save_dir='free1')
    #     plot_concept_ensemble_ts(model_dirs=dirs_free1, save_dir='free1')
        
        
                                                                                             
    # input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    # full_loader = DataLoader(train_loader.dataset.dataset, batch_size=64, shuffle=False, num_workers=0)      
    
    # model_dirs = []                                                                                          
    # for i in range(1, 12):
    #     print(i, flush=True)
    #     model_dir = f'{path}/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v{i}'
    #     ckpt = torch.load(f'{model_dir}/UNetCBM_epoch100.pt', map_location='cpu', weights_only=False)        
    #     print(i, ckpt['model_state_dict']['output_head.weight'].squeeze())                                   
    #     save_all_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm,                
    #                     output_norm=output_norm, output_dir=model_dir, full_loader=full_loader)               
    #     model_dirs.append(model_dir)                                                                         
                                                                                                            
    #plot_ml_ensemble(model_dirs)                                                                             
    #plot_concept_ensemble_ts(model_dirs)
    # for path in paths:
    #     ckpt = ckpt = torch.load(f'{path}/UNetCBM_epoch100.pt', map_location='cpu', weights_only=False)
    #     print(path)
    #     print(ckpt['model_state_dict']['output_head.weight'].squeeze())
    # breakpoint()
    # plot_concept_ensemble_ts('/path/to/data/runs_041326/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore')
    # breakpoint()
    #model_acc('/path/to/data/no_free/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v6')
    #plot_concept_ensemble_ts('/path/to/data/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v5')
    #plot_ensemble_time_series('/path/to/data/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v5')
    # breakpoint()
    # from torch.utils.data import DataLoader                                                                                                                  

    # model_dirs = ['/path/to/data/runs_040726/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v10']
    # input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    # full_loader = DataLoader(train_loader.dataset.dataset, batch_size=64, shuffle=False, num_workers=0)  
    # for model_dir in model_dirs:       
    #     save_all_preds(model_dir=model_dir, input_norm=input_norm,                                   
    #                     concept_norm=concept_norm, output_norm=output_norm,                           
    #                     full_loader=full_loader)               
    # save_all_preds(model_dir=model_dir, input_norm=input_norm,
    #                    concept_norm=concept_norm, output_norm=output_norm,
    #                    full_loader=full_loader)
    

    # model_dirs = ['/path/to/data/trended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore',
    # '/path/to/data/trended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v2',
    # '/path/to/data/trended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v3'
    # ]
    # for model_dir in model_dirs:
    #     pred_clim(model_dir=model_dir)
    #     pred_concept_clim(model_dir=model_dir)
    
    # model_dirs = ['/path/to/data/trended/UNetCBM_lam0.0_ep101_bs64_L1Loss_ZScore_v1',
    # '/path/to/data/trended/UNetCBM_lam0.0_ep101_bs64_L1Loss_ZScore_v2',
    # '/path/to/data/trended/UNetCBM_lam0.0_ep101_bs64_L1Loss_ZScore_v3']
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
    
    #MODEL_DIR = '/path/to/data/runs/UNetCBM_lam0.15_ep50_lr0.001_bs64_MSELoss_ZScore_v2'
    #save_all_preds(model_dir=MODEL_DIR)
    #compute_mhw_events(results_path=MODEL_DIR)
    #compare_mlhc_sst()
    #plot_pred_anomaly()
    #concept_weights()
    #save_val_preds()
    # Run with config.ini set to norm_type = MinMax:
    # MODEL_DIR = '/path/to/data/runs/UNetCBM_lam0.5_ep50_lr0.001_bs64_BCELoss_MinMax'
    # Run with config.ini set to norm_type = ZScore:
    #MODEL_DIR = '/path/to/data/runs/UNetCBM_lam0.5_ep50_lr0.001_bs64_BCELoss_ZScore'

    #input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    #for i in np.arange(2, 11, 1):
    #    visualize(f'/path/to/data/detrended/UNetCBM_lam0.0_ep101_lr0.001_bs64_L1Loss_ZScore_v{i}')
    #plot_sample(model_dir=MODEL_DIR, input_norm=input_norm, concept_norm=concept_norm, val_loader=val_loader)
    #plot_sample_pred_only(model_dir=MODEL_DIR, input_norm=input_norm, val_loader=val_loader)
    #run_inference(model_dir=MODEL_DIR)
    #concept_inference(model_dir=MODEL_DIR, input_norm=input_norm, concept_norm=concept_norm, val_loader=val_loader)