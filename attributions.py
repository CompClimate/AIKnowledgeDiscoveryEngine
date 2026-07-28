from captum.attr import GradientShap
from utils.get_data import get_dataset
from inference import save_val_preds
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob
import xarray as xr
from utils.get_config import config, try_cast
import utils.get_config as get_config
from utils.visualization import find_output_dir
from matplotlib.colors import ListedColormap
from scipy import stats, signal

# --- Wrappers ---

class ConceptWrapper(nn.Module):
    """Wraps model to return spatial-mean concept values per lead time.
    Output shape: (B, n_concepts * output_dim)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, concepts, _, _, _ = self.model(x)  # (B, n_concepts, output_dim, Y, X)
        return concepts.mean(dim=(-2, -1)).reshape(x.shape[0], -1)


class OutputWrapper(nn.Module):
    """Wraps model to return spatial-mean prediction per lead time.
    Output shape: (B, output_dim)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, _, _, _, _ = self.model(x)  # (B, 1, output_dim, Y, X)
        return output.mean(dim=(-2, -1)).squeeze(1)


class FreeConceptWrapper(nn.Module):
    """Wraps model to return spatial-mean of a specific free concept.
    Output shape: (B, output_dim)
    """
    def __init__(self, model, free_concept_idx=0):
        super().__init__()
        self.model = model
        self.free_concept_idx = free_concept_idx

    def forward(self, x):
        _, _, free, _, _ = self.model(x)  # free: (B, n_free, output_dim, Y, X)
        return free[:, self.free_concept_idx].mean(dim=(-2, -1))  # (B, output_dim)


# --- Helper functions ---

def _load_model(model_dir, config_path=None):
    """Load model from the latest checkpoint in model_dir."""
    if config_path is not None:
        config.read(config_path)
        print(f'Loaded config from {config_path}', flush=True)
    else:
        saved_config = f'{model_dir}/config.ini'
        if os.path.exists(saved_config):
            config.read(saved_config)
            print(f'Loaded config from {saved_config}', flush=True)

    model_type = config['MODEL']['type']
    model = get_config.get_model()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    pattern = f'{model_dir}/{model_type}_epoch*.pt'
    checkpoints = sorted(glob.glob(pattern),
                         key=lambda p: int(p.split('epoch')[-1].split('.')[0]))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found matching {pattern}')
    latest = checkpoints[-1]
    print(f'Loading checkpoint: {latest}', flush=True)

    checkpoint = torch.load(latest, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, DEVICE


def _get_val_sample(input_norm, val_loader, val_sample_idx, DEVICE):
    """Get a single normalized validation sample on device."""
    val_dataset = val_loader.dataset
    data, concept_true, target_true = val_dataset[val_sample_idx]
    data = data.unsqueeze(0)  # (1, V, T, Y, X)
    data_norm = torch.nan_to_num(input_norm.normalize(data), nan=0.0).to(DEVICE)
    return data_norm, concept_true, target_true


def _load_domain_coords(nc_path, Y, X, domain_lat=(20, 66), domain_lon=(-80, 20)):
    """Return (nav_lat, nav_lon) arrays of shape (Y, X) for the model domain."""
    ds = xr.open_dataset(nc_path)
    domain_mask = (
        (ds['nav_lat'] >= domain_lat[0]) & (ds['nav_lat'] <= domain_lat[1]) &
        (ds['nav_lon'] >= domain_lon[0]) & (ds['nav_lon'] <= domain_lon[1])
    )
    y_crop = np.where(domain_mask.any(dim='x'))[0]
    x_crop = np.where(domain_mask.any(dim='y'))[0]
    nav_lat = ds['nav_lat'].isel(y=y_crop, x=x_crop).values[:Y, :X]
    nav_lon = ds['nav_lon'].isel(y=y_crop, x=x_crop).values[:Y, :X]
    return nav_lat, nav_lon


def _setup_geoax(ax, nav_lon, nav_lat, land_mask, data_proj):
    """Configure a cartopy GeoAxes: white ocean, grey land with outline."""
    ax.set_facecolor('lightgray')
    ax.contourf(nav_lon, nav_lat, land_mask.astype(float),
                levels=[0.5, 1.5], colors=['lightgray'], transform=data_proj, zorder=1)
    ax.contour(nav_lon, nav_lat, land_mask.astype(float),
               levels=[0.5], colors='k', linewidths=0.4, transform=data_proj, zorder=2)
    ax.set_extent([float(nav_lon.min()), float(nav_lon.max()),
                   float(nav_lat.min()), float(nav_lat.max())], crs=data_proj)


# --- Main attribution function ---

def gradient_shap_inputs(model_dir=None, input_norm=None, concept_norm=None,
                         val_loader=None, val_sample_idx=None, n_baselines=20,
                         output_dir=None, config_path=None,
                         nc_path=None, gom_lat=(39, 46), gom_lon=(-71, -62),
                         domain_lat=(20, 66), domain_lon=(-80, 20),
                         gom_suffix=None):
    """GradientSHAP attribution: pixel-level spatial maps of input importance.

    Saves spatial heatmaps per variable per lead time for both
    input → prediction and input → concepts.
    """
    if model_dir is None:
        model_dir = find_output_dir()
    if output_dir is None:
        output_dir = model_dir

    model, DEVICE = _load_model(model_dir, config_path=config_path)

    if val_sample_idx is None:
        val_sample_idx = config.getint('VISUALIZATION', 'val_sample_idx', fallback=1)
    if input_norm is None or concept_norm is None or val_loader is None:
        input_norm, concept_norm, _, _, val_loader, _ = get_dataset()

    features = try_cast(config['DATASET']['features'])
    concepts = try_cast(config['DATASET']['concepts'])
    offsets = try_cast(config['DATASET']['offset'])
    n_features = len(features)
    n_concepts = len(concepts)
    n_leads = len(offsets)

    data_norm, _, _ = _get_val_sample(input_norm, val_loader, val_sample_idx, DEVICE)

    # Build baselines from other validation samples
    baseline_list = []
    val_dataset = val_loader.dataset
    for i in range(min(n_baselines, len(val_dataset))):
        if i == val_sample_idx:
            continue
        b, _, _ = val_dataset[i]
        b = b.unsqueeze(0)
        b = torch.nan_to_num(input_norm.normalize(b), nan=0.0)
        baseline_list.append(b)
    baselines = torch.cat(baseline_list, dim=0).to(DEVICE)  # (n_baselines, V, T, Y, X)
    print(f'Built {baselines.shape[0]} baselines', flush=True)

    # Load land mask
    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    land_mask = (mask_2d == 0)

    # --- Optional GOM crop ---
    if nc_path is not None:
        ds = xr.open_dataset(nc_path)
        domain_mask = (
            (ds['nav_lat'] >= domain_lat[0]) & (ds['nav_lat'] <= domain_lat[1]) &
            (ds['nav_lon'] >= domain_lon[0]) & (ds['nav_lon'] <= domain_lon[1])
        )
        y_crop = np.where(domain_mask.any(dim='x'))[0]
        x_crop = np.where(domain_mask.any(dim='y'))[0]
        nav_lat = ds['nav_lat'].isel(y=y_crop, x=x_crop).values
        nav_lon = ds['nav_lon'].isel(y=y_crop, x=x_crop).values
        gom_mask = (
            (nav_lat >= gom_lat[0]) & (nav_lat <= gom_lat[1]) &
            (nav_lon >= gom_lon[0]) & (nav_lon <= gom_lon[1])
        )[:302, :400]
        gy, gx   = np.where(gom_mask)
        y_min, y_max = gy.min(), gy.max()
        x_min, x_max = gx.min(), gx.max()
        def _crop(arr2d):
            sub = arr2d[y_min:y_max+1, x_min:x_max+1]
            return np.ma.masked_where(land_mask[y_min:y_max+1, x_min:x_max+1], sub)
        suffix = f'_gom_{gom_suffix}' if gom_suffix else '_gom'
        print(f'GOM crop: y={y_min}:{y_max} x={x_min}:{x_max}')
    else:
        def _crop(arr2d):
            return np.ma.masked_where(land_mask, arr2d)
        suffix = f'_{gom_suffix}' if gom_suffix else ''

    # --- GradientSHAP: inputs → prediction ---
    print('Computing GradientSHAP input → prediction...', flush=True)
    output_wrapper = OutputWrapper(model)
    gs = GradientShap(output_wrapper)

    for li in range(n_leads):
        attr = gs.attribute(data_norm, baselines=baselines, target=li)
        attr_np = attr.detach().cpu().numpy()[0]  # (V, T, Y, X)
        spatial_attr = attr_np.mean(axis=1)  # (V, Y, X) — signed, mean over time steps

        fig, axes = plt.subplots(1, n_features, figsize=(n_features * 4, 3.5),
                                 layout='constrained')
        for vi in range(n_features):
            ax = axes[vi]
            vals = _crop(spatial_attr[vi])
            im = ax.imshow(vals, cmap='RdBu_r', aspect='equal', origin='lower')
            ax.set_title(features[vi])
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Attribution')
        fig.suptitle(f'GradientSHAP: Input → Prediction (Lead {offsets[li]}mo)')
        save_path = f'{output_dir}/gshap_input_pred_spatial_lead{offsets[li]}{suffix}.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_path}', flush=True)

    # --- GradientSHAP: inputs → concepts ---
    print('Computing GradientSHAP input → concepts...', flush=True)
    concept_wrapper = ConceptWrapper(model)
    gs_concept = GradientShap(concept_wrapper)

    for li in range(n_leads):
        fig, axes = plt.subplots(n_features, n_concepts,
                                 figsize=(n_concepts * 4, n_features * 3),
                                 layout='constrained')
        for ci in range(n_concepts):
            target_idx = ci * n_leads + li
            attr = gs_concept.attribute(data_norm, baselines=baselines, target=target_idx)
            attr_np = attr.detach().cpu().numpy()[0]  # (V, T, Y, X)
            spatial_attr = attr_np.mean(axis=1)  # (V, Y, X) — signed, mean over time steps

            for vi in range(n_features):
                ax = axes[vi, ci] if n_features > 1 else axes[ci]
                vals = _crop(spatial_attr[vi])
                im = ax.imshow(vals, cmap='RdBu_r', aspect='equal', origin='lower')
                if vi == 0:
                    ax.set_title(concepts[ci])
                if ci == 0:
                    ax.set_ylabel(features[vi])
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f'GradientSHAP: Input → Concepts (Lead {offsets[li]}mo)')
        save_path = f'{output_dir}/gshap_input_concept_spatial_lead{offsets[li]}{suffix}.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_path}', flush=True)


def gradient_shap_free_concept(model_dir=None, input_norm=None, val_loader=None,
                               free_concept_idx=0, n_baselines=20,
                               output_dir=None, config_path=None):
    """GradientSHAP attribution: input importance for a specific free concept.

    Saves a spatial heatmap per input feature showing which inputs drive the free concept.
    """
    if model_dir is None:
        model_dir = find_output_dir()
    if output_dir is None:
        output_dir = model_dir

    model, DEVICE = _load_model(model_dir, config_path=config_path)

    val_sample_idx = config.getint('VISUALIZATION', 'val_sample_idx', fallback=1)
    if input_norm is None or val_loader is None:
        input_norm, _, _, _, val_loader, _ = get_dataset()

    features = try_cast(config['DATASET']['features'])
    offsets = try_cast(config['DATASET']['offset'])
    n_features = len(features)
    n_leads = len(offsets)

    data_norm, _, _ = _get_val_sample(input_norm, val_loader, val_sample_idx, DEVICE)

    # Build baselines from other validation samples
    baseline_list = []
    val_dataset = val_loader.dataset
    for i in range(min(n_baselines, len(val_dataset))):
        if i == val_sample_idx:
            continue
        b, _, _ = val_dataset[i]
        b = b.unsqueeze(0)
        b = torch.nan_to_num(input_norm.normalize(b), nan=0.0)
        baseline_list.append(b)
    baselines = torch.cat(baseline_list, dim=0).to(DEVICE)
    print(f'Built {baselines.shape[0]} baselines', flush=True)

    # Load land mask
    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    land_mask = (mask_2d == 0)

    free_wrapper = FreeConceptWrapper(model, free_concept_idx=free_concept_idx)
    gs = GradientShap(free_wrapper)

    for li in range(n_leads):
        print(f'Computing GradientSHAP input → free concept {free_concept_idx} (lead {offsets[li]}mo)...', flush=True)
        attr = gs.attribute(data_norm, baselines=baselines, target=li)
        attr_np = attr.detach().cpu().numpy()[0]  # (V, T, Y, X)
        spatial_attr = attr_np.mean(axis=1)       # (V, Y, X) — mean over time steps

        fig, axes = plt.subplots(1, n_features, figsize=(n_features * 4, 3.5),
                                 layout='constrained')
        for vi in range(n_features):
            ax = axes[vi]
            vals = np.ma.masked_where(land_mask, spatial_attr[vi])
            im = ax.imshow(vals, cmap='RdBu_r', aspect='equal', origin='lower')
            ax.set_title(features[vi])
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Attribution')

        fig.suptitle(f'GradientSHAP: Input → Free Concept {free_concept_idx} (Lead {offsets[li]}mo)')
        save_path = f'{output_dir}/gshap_input_free{free_concept_idx}_lead{offsets[li]}.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_path}', flush=True)

def analyze_mhw(model_dir, config_path=None, save_dir=None,
                val_start_month=5, val_start_year=2011,
                mhw_year=2012, mhw_month=7, n_members=5,
                mhw_mask_year=None,
                all_preds_path=None, all_start_year=None, all_start_month=None,
                nc_path=None, domain_lat=(20, 66), domain_lon=(-80, 20)):
    """Analyze a marine heatwave event using saved val predictions.

    1. Computes SST-based MHW mask for the target month/year (opa0)
    2. Plots predicted vs true MLHC (raw + anomaly) with MHW contour
    3. Plots concept predictions (raw + anomaly) with MHW contour
    4. Plots free concept (raw + anomaly) with MHW contour if present

    all_preds_path : path to all_preds.npz (full dataset predictions).
        When provided, uses the predicted MLHC distribution to compute the
        event threshold (instead of the true distribution), giving a fair
        comparison for event detection.
    all_start_year / all_start_month : first time step of the full dataset,
        needed to index into all_preds correctly.
    """
    if config_path is not None:
        config.read(config_path)
    else:
        saved_config = f'{model_dir}/config.ini'
        if os.path.exists(saved_config):
            config.read(saved_config)

    if save_dir is None:
        save_dir = model_dir

    loc = config['DATASET']['location']

    # --- Load predictions (all_preds if available, else val_preds) ---
    if all_preds_path is not None and all_start_year is not None and all_start_month is not None:
        results       = np.load(all_preds_path, allow_pickle=True)
        start_year, start_month = all_start_year, all_start_month
        print('Using all_preds for main predictions')
    else:
        results       = np.load(os.path.join(model_dir, 'val_preds_lead0.npz'), allow_pickle=True)
        start_year, start_month = val_start_year, val_start_month
        print('Using val_preds for main predictions')

    preds         = results['preds']          # (N, Y, X)
    targets       = results['targets']        # (N, Y, X)
    concept_preds = results['concept_preds']  # (n_concepts, N, Y, X)
    concept_names = results['concept_names']
    ocean_mask    = results['ocean_mask']     # (Y, X)
    land_mask     = ~ocean_mask
    free_preds    = results['free_preds'] if 'free_preds' in results else None

    Y, X = ocean_mask.shape
    proj = ccrs.PlateCarree()
    nav_lat, nav_lon = _load_domain_coords(nc_path, Y, X, domain_lat, domain_lon)

    # --- Find time index of target month ---
    from_year  = (mhw_year - start_year) * 12
    from_month = mhw_month - start_month
    time_idx   = from_year + from_month
    opa0_idx   = time_idx * n_members

    # All target-month indices for climatology (opa0 only)
    n_times = preds.shape[0] // n_members
    start_offset = (start_year * 12 + start_month - 1)
    month_time_idxs = [t for t in range(n_times)
                       if (start_offset + t) % 12 == (mhw_month - 1)]
    month_opa0_idxs = [t * n_members for t in month_time_idxs]
    print(f'time_idx={time_idx}, opa0_idx={opa0_idx}, n_clim_months={len(month_opa0_idxs)}')

    # --- SST MHW mask (opa0, full record) ---
    sst_ds    = xr.open_zarr(f'{loc}/opa0/sosstsst_na.zarr')
    sst_month = sst_ds.sel(time_counter=sst_ds.time_counter.dt.month == mhw_month
                           )['sosstsst'].isel(y=slice(0, Y), x=slice(0, X)).values
    sst_years = sst_ds.sel(time_counter=sst_ds.time_counter.dt.month == mhw_month
                           ).time_counter.dt.year.values
    sst_clim  = np.nanmean(sst_month, axis=0)
    sst_anom  = sst_month - sst_clim
    nan_mask  = np.isnan(sst_anom)
    sst_anom[nan_mask] = 0.0
    sst_anom[:] = signal.detrend(sst_anom, axis=0)
    sst_anom[nan_mask] = np.nan
    sst_thresh   = np.nanpercentile(sst_anom, 90, axis=0)
    mask_year    = mhw_mask_year if mhw_mask_year is not None else mhw_year
    year_idx     = np.where(sst_years == mask_year)[0][0]
    mhw_mask     = np.where(ocean_mask, sst_anom[year_idx] > sst_thresh, False)
    print(f'MHW cells (SST-based): {mhw_mask.sum()}')

    # --- True MLHC climatology (opa0, full record) ---
    mlhc_ds    = xr.open_zarr(f'{loc}/opa0/vomlhc_na.zarr')
    mlhc_month = mlhc_ds.sel(time_counter=mlhc_ds.time_counter.dt.month == mhw_month
                             )['vomlhc'].isel(y=slice(0, Y), x=slice(0, X)).values
    mlhc_clim  = np.nanmean(mlhc_month, axis=0)

    pred_july   = preds[opa0_idx]
    target_july = targets[opa0_idx]
    pred_anom   = pred_july   - mlhc_clim
    target_anom = target_july - mlhc_clim

    # --- Event detection ---
    # True threshold: 90th percentile of true MLHC anomaly across all years
    mlhc_anom_all = mlhc_month - mlhc_clim
    true_thresh   = np.nanpercentile(mlhc_anom_all, 90, axis=0)
    target_events = np.where(ocean_mask, target_anom > true_thresh, np.nan).astype(float)

    # Predicted threshold from all July years (preds already loaded from all_preds if available)
    pred_month_all = preds[month_opa0_idxs]   # (n_july_years, Y, X)
    pred_clim_all  = np.nanmean(pred_month_all, axis=0)
    pred_anom_all  = pred_month_all - pred_clim_all
    pred_thresh    = np.nanpercentile(pred_anom_all, 90, axis=0)
    pred_anom_for_detection = preds[opa0_idx] - pred_clim_all
    pred_events    = np.where(ocean_mask, pred_anom_for_detection > pred_thresh, np.nan).astype(float)
    print(f'Predicted threshold from {len(month_opa0_idxs)} July years')

    # Hit rate and false alarm rate
    valid        = ocean_mask & ~np.isnan(target_events) & ~np.isnan(pred_events)
    hits         = np.sum((target_events == 1) & (pred_events == 1) & valid)
    misses       = np.sum((target_events == 1) & (pred_events == 0) & valid)
    false_alarms = np.sum((target_events == 0) & (pred_events == 1) & valid)
    hit_rate     = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far          = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    print(f'Hit rate: {hit_rate:.3f}, False alarm rate: {far:.3f}')

    def _iplot(ax, data, title, cmap='RdBu_r', symmetric=True):
        masked = np.ma.masked_where(land_mask, data)
        if symmetric:
            vabs = np.nanpercentile(np.abs(masked.compressed()), 98)
            im = ax.pcolormesh(nav_lon, nav_lat, masked,
                               transform=proj, cmap=cmap, vmin=-vabs, vmax=vabs)
        else:
            im = ax.pcolormesh(nav_lon, nav_lat, masked, transform=proj, cmap=cmap)
        _setup_geoax(ax, nav_lon, nav_lat, land_mask, proj)
        ax.contour(nav_lon, nav_lat, mhw_mask.astype(float),
                   levels=[0.5], colors='red', linewidths=0.8, transform=proj)
        ax.set_title(title, fontsize=9)
        return im

    # --- Load MLD for sanity check (pred MLHC / MLD ≈ T_ml ≈ SST) ---
    mld_ds    = xr.open_zarr(f'{loc}/opa0/somxl010_na.zarr')
    mld_month = mld_ds.sel(time_counter=mld_ds.time_counter.dt.month == mhw_month
                           )['somxl010'].isel(y=slice(0, Y), x=slice(0, X)).values
    mld_years = mld_ds.sel(time_counter=mld_ds.time_counter.dt.month == mhw_month
                           ).time_counter.dt.year.values
    mld_year_idx = np.where(mld_years == mhw_year)[0][0]
    mld_july     = np.where(mld_month[mld_year_idx] > 0, mld_month[mld_year_idx], np.nan)
    mld_clim     = np.where(np.nanmean(mld_month, axis=0) > 0,
                            np.nanmean(mld_month, axis=0), np.nan)

    # MLHC anomaly normalised by climatological MLD → comparable to SST anomaly
    pred_t_ml_anom   = pred_anom   / mld_clim
    target_t_ml_anom = target_anom / mld_clim

    # SST anomaly for comparison
    sst_ds      = xr.open_zarr(f'{loc}/opa0/sosstsst_na.zarr')
    sst_month   = sst_ds.sel(time_counter=sst_ds.time_counter.dt.month == mhw_month
                             )['sosstsst'].isel(y=slice(0, Y), x=slice(0, X)).values
    sst_years   = sst_ds.sel(time_counter=sst_ds.time_counter.dt.month == mhw_month
                             ).time_counter.dt.year.values
    sst_clim    = np.nanmean(sst_month, axis=0)
    sst_year_idx = np.where(sst_years == mhw_year)[0][0]
    sst_anom    = sst_month[sst_year_idx] - sst_clim

    # --- Figure 1: MLHC pred vs true ---
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), subplot_kw={'projection': proj})
    for ax, data, title, cmap, sym in [
        (axes[0, 0], pred_july,         'Pred MLHC',               'viridis', False),
        (axes[0, 1], target_july,       'True MLHC',               'viridis', False),
        (axes[1, 0], pred_anom,         'Pred MLHC Anomaly',        'RdBu_r',  True),
        (axes[1, 1], target_anom,       'True MLHC Anomaly',        'RdBu_r',  True),
        (axes[2, 0], pred_t_ml_anom,    'Pred MLHC Anom / MLD clim', 'RdBu_r',  True),
        (axes[2, 1], sst_anom,          'True SST Anomaly',         'RdBu_r',  True),
    ]:
        fig.colorbar(_iplot(ax, data, title, cmap=cmap, symmetric=sym),
                     ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'MLHC {mhw_month}/{mhw_year} opa0 — red = SST MHW cells ({mask_year})')
    fig.tight_layout()
    path = os.path.join(save_dir, f'mhw_mlhc_{mhw_month}{mhw_year}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # --- Figure 1b: MLHC event maps (percentile threshold) ---
    from matplotlib.colors import ListedColormap
    event_cmap = ListedColormap(['white', 'red'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={'projection': proj})
    for ax, events, title in [
        (axes[0], pred_events,   f'Pred MLHC Events (hit={hit_rate:.2f}, far={far:.2f})'),
        (axes[1], target_events, 'True MLHC Events'),
    ]:
        masked = np.ma.masked_invalid(events)
        ax.pcolormesh(nav_lon, nav_lat, masked,
                      transform=proj, cmap=event_cmap, vmin=0, vmax=1)
        _setup_geoax(ax, nav_lon, nav_lat, land_mask, proj)
        ax.set_title(title, fontsize=9)
    fig.suptitle(f'MLHC MHW Events {mhw_month}/{mhw_year} opa0 (90th pct threshold)')
    fig.tight_layout()
    path = os.path.join(save_dir, f'mhw_events_{mhw_month}{mhw_year}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # --- Figure 1c: Rank-based event detection ---
    # Flag the top-10% of ocean pixels by anomaly magnitude (no per-pixel threshold needed)
    def _rank_events(anom_map, mask, pct=90):
        """Return bool array: True where anom_map is in top (100-pct)% of ocean pixels."""
        ocean_vals = anom_map[mask]
        thresh = np.nanpercentile(ocean_vals, pct)
        return np.where(mask, anom_map > thresh, False)

    rank_target = _rank_events(target_anom, ocean_mask)
    rank_pred   = _rank_events(pred_anom_for_detection, ocean_mask)

    rank_valid  = ocean_mask
    rank_hits   = np.sum(rank_target & rank_pred & rank_valid)
    rank_misses = np.sum(rank_target & ~rank_pred & rank_valid)
    rank_fa     = np.sum(~rank_target & rank_pred & rank_valid)
    rank_hr     = rank_hits / (rank_hits + rank_misses) if (rank_hits + rank_misses) > 0 else 0.0
    rank_far    = rank_fa   / (rank_hits + rank_fa)     if (rank_hits + rank_fa) > 0 else 0.0
    print(f'Rank-based  hit rate: {rank_hr:.3f}, FAR: {rank_far:.3f}')

    # Category map: TP=both, FN=true only, FP=pred only
    category = np.full((Y, X), np.nan)
    category[rank_pred  & ~rank_target & ocean_mask] = 1  # FP
    category[rank_target & ~rank_pred  & ocean_mask] = 2  # FN
    category[rank_pred  &  rank_target & ocean_mask] = 3  # TP
    cat_cmap = ListedColormap(['#4393c3', '#d6604d', '#1a9641'])  # blue=FP, red=FN, green=TP
    cat_cmap.set_bad('lightgray')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={'projection': proj})
    for ax, events, title in [
        (axes[0], rank_pred.astype(float),   f'Pred top-10% (hit={rank_hr:.2f}, far={rank_far:.2f})'),
        (axes[1], rank_target.astype(float), 'True top-10%'),
    ]:
        masked = np.ma.masked_where(~ocean_mask, events)
        ax.pcolormesh(nav_lon, nav_lat, masked,
                      transform=proj, cmap=event_cmap, vmin=0, vmax=1)
        _setup_geoax(ax, nav_lon, nav_lat, land_mask, proj)
        ax.set_title(title, fontsize=9)
    masked_cat = np.ma.masked_invalid(category)
    axes[2].pcolormesh(nav_lon, nav_lat, masked_cat,
                       transform=proj, cmap=cat_cmap, vmin=0.5, vmax=3.5)
    _setup_geoax(axes[2], nav_lon, nav_lat, land_mask, proj)
    axes[2].set_title('TP=green  FN=red  FP=blue', fontsize=9)
    fig.suptitle(f'Rank-based MLHC Events {mhw_month}/{mhw_year} opa0 (top 10% of domain)')
    fig.tight_layout()
    path = os.path.join(save_dir, f'mhw_events_rank_{mhw_month}{mhw_year}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # --- Figure 1d: Per-pixel correlation across all July years ---
    # true anomalies: mlhc_anom_all (T_true, Y, X) from zarr
    # pred anomalies: pred_anom_all (T_pred, Y, X) from preds array
    n_true_yrs = mlhc_anom_all.shape[0]
    n_pred_yrs = len(month_opa0_idxs)
    n_yrs_corr = min(n_true_yrs, n_pred_yrs)
    ta = mlhc_anom_all[-n_yrs_corr:]   # (T, Y, X)
    pa = pred_anom_all[-n_yrs_corr:]   # (T, Y, X)

    # Pearson r pixel-wise
    ta_c = ta - ta.mean(axis=0)
    pa_c = pa - pa.mean(axis=0)
    num  = (ta_c * pa_c).sum(axis=0)
    den  = np.sqrt((ta_c**2).sum(axis=0) * (pa_c**2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        pix_r = np.where(den > 0, num / den, np.nan)
    pix_r = np.where(ocean_mask, pix_r, np.nan)
    median_r = np.nanmedian(pix_r)
    print(f'Per-pixel r across July years — median: {median_r:.3f}')

    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': proj})
    masked_r = np.ma.masked_invalid(pix_r)
    im = ax.pcolormesh(nav_lon, nav_lat, masked_r,
                       transform=proj, cmap='RdBu_r', vmin=-1, vmax=1)
    _setup_geoax(ax, nav_lon, nav_lat, land_mask, proj)
    ax.contour(nav_lon, nav_lat, mhw_mask.astype(float),
               levels=[0.5], colors='k', linewidths=0.8, transform=proj)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    ax.set_title(f'Per-pixel Pearson r (pred vs true July MLHC anom, {len(ta)} yrs)\n'
                 f'median r = {median_r:.3f}  |  black contour = {mhw_year} SST MHW', fontsize=9)
    fig.tight_layout()
    path = os.path.join(save_dir, f'mhw_pixelr_{mhw_month}{mhw_year}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # --- Figure 2: Concept predictions ---
    n_concepts = len(concept_names)
    fig, axes  = plt.subplots(2, n_concepts, figsize=(n_concepts * 3, 6),
                              subplot_kw={'projection': proj})
    for ci, cname in enumerate(concept_names):
        cp      = concept_preds[ci, opa0_idx]
        cp_clim = np.nanmean([concept_preds[ci, j] for j in month_opa0_idxs], axis=0)
        cp_anom = cp - cp_clim
        fig.colorbar(_iplot(axes[0, ci], cp,      cname,           symmetric=True), ax=axes[0, ci], fraction=0.046, pad=0.04)
        fig.colorbar(_iplot(axes[1, ci], cp_anom, f'{cname} anom', symmetric=True), ax=axes[1, ci], fraction=0.046, pad=0.04)
    fig.suptitle(f'Concept Predictions {mhw_month}/{mhw_year} opa0 — red = SST MHW cells ({mask_year})')
    fig.tight_layout()
    path = os.path.join(save_dir, f'mhw_concepts_{mhw_month}{mhw_year}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # --- Figure 3: Free concept ---
    if free_preds is not None:
        n_free = free_preds.shape[0]
        fig, axes = plt.subplots(2, n_free, figsize=(n_free * 4, 6), squeeze=False,
                                 subplot_kw={'projection': proj})
        for fi in range(n_free):
            fp      = free_preds[fi, opa0_idx]
            fp_clim = np.nanmean([free_preds[fi, j] for j in month_opa0_idxs], axis=0)
            fp_anom = fp - fp_clim
            fig.colorbar(_iplot(axes[0, fi], fp,      f'Free {fi}',        symmetric=True), ax=axes[0, fi], fraction=0.046, pad=0.04)
            fig.colorbar(_iplot(axes[1, fi], fp_anom, f'Free {fi} (anom)', symmetric=True), ax=axes[1, fi], fraction=0.046, pad=0.04)
        fig.suptitle(f'Free Concept {mhw_month}/{mhw_year} opa0 — red = SST MHW cells ({mask_year})')
        fig.tight_layout()
        path = os.path.join(save_dir, f'mhw_free_{mhw_month}{mhw_year}.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')

    # --- Figure 4: Concept targets (ground truth) ---
    concept_targets = results['concept_targets']  # (n_concepts, N, Y, X)
    fig, axes = plt.subplots(2, n_concepts, figsize=(n_concepts * 3, 6),
                             subplot_kw={'projection': proj})
    for ci, cname in enumerate(concept_names):
        ct      = concept_targets[ci, opa0_idx]
        ct_clim = np.nanmean([concept_targets[ci, j] for j in month_opa0_idxs], axis=0)
        ct_anom = ct - ct_clim
        fig.colorbar(_iplot(axes[0, ci], ct,      cname,           symmetric=True), ax=axes[0, ci], fraction=0.046, pad=0.04)
        fig.colorbar(_iplot(axes[1, ci], ct_anom, f'{cname} anom', symmetric=True), ax=axes[1, ci], fraction=0.046, pad=0.04)
    fig.suptitle(f'Concept Targets {mhw_month}/{mhw_year} opa0 — red = SST MHW cells ({mask_year})')
    fig.tight_layout()
    path = os.path.join(save_dir, f'mhw_concept_targets_{mhw_month}{mhw_year}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def find_mlhc_heatwave_years(loc, month=7, pct=90, save_dir='.', n_members=1,
                             opa='opa0', Y=302, X=400,
                             nc_path=None, domain_lat=(20, 66), domain_lon=(-80, 20)):
    """Find years with the largest MLHC heatwave extent for a given month.

    For each year, counts the fraction of ocean pixels where the detrended
    MLHC anomaly exceeds the per-pixel 90th percentile threshold.
    Saves a bar chart ranked by heatwave extent and prints the top years.

    Parameters
    ----------
    loc      : base data path (parent of opa0/, opa1/, etc.)
    month    : calendar month to analyse (1–12)
    pct      : percentile threshold for defining a heatwave pixel (default 90)
    save_dir : where to save the figure
    opa      : which ensemble member zarr to use for the true MLHC
    """
    mlhc_ds    = xr.open_zarr(f'{loc}/{opa}/vomlhc_na.zarr')
    sel        = mlhc_ds.sel(time_counter=mlhc_ds.time_counter.dt.month == month)
    mlhc_month = sel['vomlhc'].isel(y=slice(0, Y), x=slice(0, X)).values  # (T, Y, X)
    years      = sel.time_counter.dt.year.values

    # Ocean mask
    mesh       = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    ocean_mask = mesh['tmaskutil'].isel(t=0, y=slice(0, Y), x=slice(0, X)).values == 1
    n_ocean    = ocean_mask.sum()

    # Detrended anomaly
    mlhc_clim  = np.nanmean(mlhc_month, axis=0)
    mlhc_anom  = mlhc_month - mlhc_clim
    nan_mask   = np.isnan(mlhc_anom)
    mlhc_anom[nan_mask] = 0.0
    mlhc_anom[:] = signal.detrend(mlhc_anom, axis=0)
    mlhc_anom[nan_mask] = np.nan

    # Per-pixel threshold
    thresh     = np.nanpercentile(mlhc_anom, pct, axis=0)  # (Y, X)

    # Fraction of ocean pixels exceeding threshold each year
    hw_frac = np.array([
        np.sum((mlhc_anom[yi] > thresh) & ocean_mask) / n_ocean
        for yi in range(len(years))
    ])

    # Expected fraction at threshold (e.g. 10% for 90th pct)
    expected = (100 - pct) / 100.0

    order      = np.argsort(hw_frac)[::-1]
    print(f'\nTop 10 MLHC heatwave years (month={month}, >{pct}th pct):')
    for rank, yi in enumerate(order[:10]):
        print(f'  {rank+1:2d}. {years[yi]}  HW extent: {hw_frac[yi]*100:.1f}%  '
              f'(expected ~{expected*100:.0f}%)')

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(max(8, len(years) * 0.25), 4))
    colors = ['#d73027' if f > expected * 1.5 else '#fc8d59' if f > expected else '#91bfdb'
              for f in hw_frac]
    ax.bar(years, hw_frac * 100, color=colors, edgecolor='none')
    ax.axhline(expected * 100, color='k', linestyle='--', linewidth=1,
               label=f'Expected ({expected*100:.0f}%)')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'% ocean pixels > {pct}th pct MLHC anom')
    ax.set_title(f'MLHC Heatwave Extent — month {month}  |  red = >1.5× expected')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, f'mlhc_hw_years_month{month}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # --- Spatial map of top-3 years ---
    top3 = order[:3]
    land_mask = ~ocean_mask
    proj = ccrs.PlateCarree()
    nav_lat_dom, nav_lon_dom = _load_domain_coords(nc_path, Y, X, domain_lat, domain_lon)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={'projection': proj})
    vabs = np.nanpercentile(np.abs(mlhc_anom), 98)
    for col, yi in enumerate(top3):
        ax   = axes[col]
        data = np.ma.masked_where(land_mask, mlhc_anom[yi])
        im   = ax.pcolormesh(nav_lon_dom, nav_lat_dom, data,
                             transform=proj, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
        _setup_geoax(ax, nav_lon_dom, nav_lat_dom, land_mask, proj)
        ax.contour(nav_lon_dom, nav_lat_dom, ((mlhc_anom[yi] > thresh) & ocean_mask).astype(float),
                   levels=[0.5], colors='k', linewidths=0.8, transform=proj)
        ax.set_title(f'{years[yi]}  ({hw_frac[yi]*100:.1f}% HW pixels)', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Top-3 MLHC heatwave years (month={month}) — black contour = HW pixels')
    fig.tight_layout()
    path = os.path.join(save_dir, f'mlhc_hw_top3_month{month}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    return years, hw_frac, order


def plot_mhw_inputs(model_dir, config_path=None, save_dir='./',
                    mhw_mask=None, mhw_year=2012, mhw_month=7,
                    compare_year=2011, n_members=5,
                    val_start_month=5, val_start_year=2011,
                    nc_path=None, domain_lat=(20, 66), domain_lon=(-80, 20)):
    """Plot input feature anomalies at MHW cells for mhw_year vs compare_year.

    Loads input zarr files directly. Only plots features with direct zarr files
    (skips derived features like vozocrtx_ml, vosaldiff, etc.).
    Requires mhw_mask (Y, X bool) — use the SST-based mask from analyze_mhw.
    """
    if config_path is not None:
        config.read(config_path)
    else:
        saved_config = f'{model_dir}/config.ini'
        if os.path.exists(saved_config):
            config.read(saved_config)

    if save_dir is None:
        save_dir = model_dir

    loc      = config['DATASET']['location']
    features = try_cast(config['DATASET']['features'])

    # Features with direct zarr files in {loc}/opa0/{feature}_na.zarr
    zarr_features = [f for f in features if f in
                     ['sosstsst', 'sosaline', 'sossheig', 'somxl010', 'sohefldo', 'sowsc']]

    results    = np.load(os.path.join(model_dir, 'val_preds_lead0.npz'), allow_pickle=True)
    ocean_mask = results['ocean_mask']
    land_mask  = ~ocean_mask
    Y, X       = ocean_mask.shape

    def _get_year_idx(ds, month, year):
        years = ds.sel(time_counter=ds.time_counter.dt.month == month).time_counter.dt.year.values
        return np.where(years == year)[0][0]

    def _iplot(ax, data, title, cmap='RdBu_r'):
        masked = np.ma.masked_where(land_mask, data)
        vabs   = np.nanpercentile(np.abs(masked.compressed()), 98)
        im     = ax.imshow(masked, origin='lower', cmap=cmap, vmin=-vabs, vmax=vabs)
        if mhw_mask is not None:
            ax.contour(mhw_mask.astype(float), levels=[0.5], colors='red', linewidths=0.8)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        return im

    n_feats = len(zarr_features)
    fig, axes = plt.subplots(2, n_feats, figsize=(n_feats * 3, 6))

    for fi, feat in enumerate(zarr_features):
        ds       = xr.open_zarr(f'{loc}/opa0/{feat}_na.zarr')
        arr      = ds.sel(time_counter=ds.time_counter.dt.month == mhw_month
                          )[feat].isel(y=slice(0, Y), x=slice(0, X)).values
        clim     = np.nanmean(arr, axis=0)
        yi_mhw   = _get_year_idx(ds, mhw_month, mhw_year)
        yi_cmp   = _get_year_idx(ds, mhw_month, compare_year)
        anom_mhw = arr[yi_mhw] - clim
        anom_cmp = arr[yi_cmp] - clim

        im0 = _iplot(axes[0, fi], anom_mhw, f'{feat} anom {mhw_year}')
        im1 = _iplot(axes[1, fi], anom_cmp, f'{feat} anom {compare_year}')
        fig.colorbar(im0, ax=axes[0, fi], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1, fi], fraction=0.046, pad=0.04)

    fig.suptitle(f'Input Anomalies {mhw_month}/{mhw_year} vs {compare_year} — red = SST MHW cells')
    fig.tight_layout()
    path = os.path.join(save_dir, f'mhw_inputs_{mhw_month}{mhw_year}_vs_{compare_year}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def analyze_gom(model_dirs, nc_path, year=2012, month=8, member=0,
                val_start_year=2011, val_start_month=5, n_members=5,
                n_lookback=6, save_dir=None, compare_year=None,
                gom_lat=(39, 46), gom_lon=(-71, -62),
                domain_lat=(20, 66), domain_lon=(-80, 20),
                all_preds_path=None, all_start_year=None, all_start_month=None):
    """Plot predicted MLHC and concept predictions zoomed into the Gulf of Maine.

    Rows = fields (pred MLHC, von2, vohfe, free).
    Cols = n_lookback months ending at (year, month).

    Parameters
    ----------
    model_dirs       : str or list of str — directories containing val_preds_lead0.npz;
                       if multiple, predictions are averaged across models
    nc_path          : path to any ORCA025 .nc file with nav_lat/nav_lon
    year, month      : final (most recent) month to show
    member           : ensemble member index (0=opa0)
    n_lookback       : number of months to show (counting back from year/month)
    gom_lat/lon      : Gulf of Maine bounding box
    domain_lat/lon   : model domain bounds used to find y/x crop in full grid
    """
    import calendar

    if isinstance(model_dirs, str):
        model_dirs = [model_dirs]

    if save_dir is None:
        save_dir = model_dirs[0]

    # --- Load and average predictions across models ---
    use_all_preds = all_preds_path is not None and all_start_year is not None and all_start_month is not None
    start_year  = all_start_year  if use_all_preds else val_start_year
    start_month = all_start_month if use_all_preds else val_start_month

    all_preds, all_concept_preds, all_free_preds = [], [], []
    for md in model_dirs:
        npz_path = all_preds_path if use_all_preds else os.path.join(md, 'val_preds_lead0.npz')
        r = np.load(npz_path, allow_pickle=True)
        all_preds.append(r['preds'])
        all_concept_preds.append(r['concept_preds'])
        if 'free_preds' in r:
            all_free_preds.append(r['free_preds'])

    first_path = all_preds_path if use_all_preds else os.path.join(model_dirs[0], 'val_preds_lead0.npz')
    first         = np.load(first_path, allow_pickle=True)
    preds         = np.mean(all_preds, axis=0)           # (N, Y, X)
    targets       = first['targets']                      # (N, Y, X)
    concept_preds = np.mean(all_concept_preds, axis=0)   # (n_concepts, N, Y, X)
    concept_names = first['concept_names']
    ocean_mask    = first['ocean_mask']                   # (Y, X)
    free_preds    = np.mean(all_free_preds, axis=0) if all_free_preds else None

    N, Y, X    = preds.shape
    n_times    = N // n_members
    n_concepts = len(concept_names)
    n_free     = free_preds.shape[0] if free_preds is not None else 0

    preds         = preds.reshape(n_times, n_members, Y, X)
    targets       = targets.reshape(n_times, n_members, Y, X)
    concept_preds = concept_preds.reshape(n_concepts, n_times, n_members, Y, X)
    if free_preds is not None:
        free_preds = free_preds.reshape(n_free, n_times, n_members, Y, X)

    # --- Load config for data location ---
    saved_config = os.path.join(model_dirs[0], 'config.ini')
    if os.path.exists(saved_config):
        config.read(saved_config)
    loc = config['DATASET']['location']

    # --- Build list of (time_idx, calendar_month, label) for each row ---
    end_idx = (year - start_year) * 12 + (month - start_month)
    rows = []
    for offset in range(n_lookback - 1, -1, -1):   # oldest first
        ti = end_idx - offset
        abs_month = (val_start_year * 12 + val_start_month - 1) + ti
        row_year  = abs_month // 12
        row_month = abs_month % 12 + 1
        label = f"{calendar.month_abbr[row_month]} {row_year}"
        rows.append((ti, row_month, label))
    print('Rows:', [(r[2], r[0]) for r in rows])

    # --- True climatologies from zarr (opa0, full record) ---
    # MLHC climatology per calendar month
    mlhc_ds  = xr.open_zarr(f'{loc}/opa0/vomlhc_na.zarr')
    mlhc_clims = {}
    for cal_month in set(r[1] for r in rows):
        vals = mlhc_ds.sel(time_counter=mlhc_ds.time_counter.dt.month == cal_month
                           )['vomlhc'].isel(y=slice(0, Y), x=slice(0, X)).values
        mlhc_clims[cal_month] = np.nanmean(vals, axis=0)  # (Y, X)
    print('Loaded MLHC climatologies')

    # Concept climatologies per calendar month
    concept_clims = {}   # concept_name -> {cal_month -> (Y, X)}
    for cname in concept_names:
        concept_clims[cname] = {}
        cds = xr.open_zarr(f'{loc}/opa0/{cname}_na.zarr')
        var_name = [v for v in cds.data_vars][0]
        for cal_month in set(r[1] for r in rows):
            vals = cds.sel(time_counter=cds.time_counter.dt.month == cal_month
                           )[var_name].isel(y=slice(0, Y), x=slice(0, X)).values
            concept_clims[cname][cal_month] = np.nanmean(vals, axis=0)
        print(f'Loaded climatology for {cname}')

    # Free concept climatology from val predictions (no zarr available)
    free_clims = {}   # fi -> {cal_month -> (Y, X)}
    if n_free > 0:
        for fi in range(n_free):
            free_clims[fi] = {}
            for cal_month in set(r[1] for r in rows):
                month_ti_idxs = [r[0] for r in rows if r[1] == cal_month]
                # include all val time steps with this calendar month
                all_month_ti = [t for t in range(n_times)
                                if ((val_start_year * 12 + val_start_month - 1) + t) % 12 + 1 == cal_month]
                vals = np.stack([free_preds[fi, t, :, :, :].mean(axis=0)
                                 for t in all_month_ti if t < n_times], axis=0)
                free_clims[fi][cal_month] = np.nanmean(vals, axis=0)

    # --- GOM mask from full ORCA025 grid ---
    ds = xr.open_dataset(nc_path)
    domain_mask = (
        (ds['nav_lat'] >= domain_lat[0]) & (ds['nav_lat'] <= domain_lat[1]) &
        (ds['nav_lon'] >= domain_lon[0]) & (ds['nav_lon'] <= domain_lon[1])
    )
    y_crop  = np.where(domain_mask.any(dim='x'))[0]
    x_crop  = np.where(domain_mask.any(dim='y'))[0]
    nav_lat = ds['nav_lat'].isel(y=y_crop, x=x_crop).values
    nav_lon = ds['nav_lon'].isel(y=y_crop, x=x_crop).values
    gom_mask = (
        (nav_lat >= gom_lat[0]) & (nav_lat <= gom_lat[1]) &
        (nav_lon >= gom_lon[0]) & (nav_lon <= gom_lon[1])
    )[:Y, :X]

    # --- Bounding box crop ---
    y_inds, x_inds = np.where(gom_mask)
    y_min, y_max   = y_inds.min(), y_inds.max()
    x_min, x_max   = x_inds.min(), x_inds.max()
    land_mask_gom  = ~ocean_mask[y_min:y_max+1, x_min:x_max+1]

    nav_lat_gom = nav_lat[:Y, :X][y_min:y_max+1, x_min:x_max+1]
    nav_lon_gom = nav_lon[:Y, :X][y_min:y_max+1, x_min:x_max+1]
    lon_min, lon_max = nav_lon_gom.min(), nav_lon_gom.max()
    lat_min, lat_max = nav_lat_gom.min(), nav_lat_gom.max()

    def _crop(arr2d):
        return np.ma.masked_where(land_mask_gom, arr2d[y_min:y_max+1, x_min:x_max+1])

    # --- Layout: rows=fields, cols=months ---
    # fields: pred MLHC, von2, vohfe, free (if present)
    concept_names_list = list(concept_names)
    selected_concepts = [c for c in ['von2', 'vohfe'] if c in concept_names_list]
    selected_concepts_names = ['Buoyancy freq.', 'Heat flux']
    row_fields  = ['pred'] + selected_concepts + (['free'] if n_free > 0 else [])
    row_labels  = ['MLHC'] + selected_concepts_names + (['Free'] if n_free > 0 else [])
    n_rows = len(row_fields)
    n_cols = n_lookback   # one column per month

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('lightgray')

    data_proj = ccrs.PlateCarree()
    map_proj  = ccrs.Mercator()
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.2),
                             subplot_kw={'projection': map_proj},
                             gridspec_kw={'hspace': 0.05})

    # pre-build cropped arrays averaged over all ensemble members
    field_arrays = {f: [] for f in row_fields}
    for ti, cal_month, _ in rows:
        field_arrays['pred'].append(_crop(preds[ti].mean(axis=0) - mlhc_clims[cal_month]))
        for cname in selected_concepts:
            ci = concept_names_list.index(cname)
            field_arrays[cname].append(_crop(concept_preds[ci, ti].mean(axis=0) - concept_clims[cname][cal_month]))
        if n_free > 0:
            field_arrays['free'].append(_crop(free_preds[0, ti].mean(axis=0) - free_clims[0][cal_month]))

    for ri, (fname, rlabel) in enumerate(zip(row_fields, row_labels)):
        arrs = field_arrays[fname]
        all_vals = np.concatenate([a.compressed() for a in arrs])
        vabs = np.nanpercentile(np.abs(all_vals), 98)
        for ci, (arr, (_, _, col_label)) in enumerate(zip(arrs, rows)):
            ax = axes[ri, ci]
            im = ax.pcolormesh(nav_lon_gom, nav_lat_gom, arr,
                               transform=data_proj, cmap=cmap, vmin=-vabs, vmax=vabs)
            _setup_geoax(ax, nav_lon_gom, nav_lat_gom, land_mask_gom, data_proj)
            if ri == 0:
                ax.set_title(col_label, fontsize=9)
            # if ci == 0:
            #     ax.set_ylabel(rlabel, fontsize=9)
            if ci == 0:
                # transform=ax.transAxes ensures (0,0) is bottom-left and (1,1) is top-right of the axis
                ax.text(-0.15, 0.5, rlabel, transform=ax.transAxes, 
                        rotation=90, va='center', ha='right', 
                        fontsize=10)
        # one colorbar per row on the right
        cbar_label = {'pred': 'Anomaly (J m⁻²)', 'vohfe': 'Anomaly (W m⁻²)',
                      'von2': 'Anomaly (s⁻²)', 'free': 'Anomaly'}.get(fname, 'Anomaly')
        fig.colorbar(im, ax=axes[ri, :].tolist(), shrink=0.6, pad=0.02, format='%.1e', label=cbar_label)

    fig.suptitle(f'Gulf of Maine anomalies (wrt true clim) — last {n_lookback} months ending {year}-{month:02d}  (ensemble mean)',
                 fontsize=11)
    path = os.path.join(save_dir, f'gom_{year}{month:02d}_ensmean.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

    # --- Difference figure: year anomaly minus compare_year anomaly ---
    if compare_year is not None:
        compare_end_idx = (compare_year - val_start_year) * 12 + (month - val_start_month)
        compare_rows = []
        for offset in range(n_lookback - 1, -1, -1):
            ti = compare_end_idx - offset
            abs_month = (val_start_year * 12 + val_start_month - 1) + ti
            row_month = abs_month % 12 + 1
            compare_rows.append((ti, row_month))

        # build diff arrays: field -> list of cropped (year_anom - compare_anom)
        diff_arrays = {f: [] for f in row_fields}
        for (ti_yr, cal_month, _), (ti_cmp, _) in zip(rows, compare_rows):
            diff_arrays['pred'].append(_crop(
                (preds[ti_yr].mean(axis=0) - mlhc_clims[cal_month]) -
                (preds[ti_cmp].mean(axis=0) - mlhc_clims[cal_month])))
            for cname in selected_concepts:
                ci = concept_names_list.index(cname)
                diff_arrays[cname].append(_crop(
                    (concept_preds[ci, ti_yr].mean(axis=0) - concept_clims[cname][cal_month]) -
                    (concept_preds[ci, ti_cmp].mean(axis=0) - concept_clims[cname][cal_month])))
            if n_free > 0:
                diff_arrays['free'].append(_crop(
                    (free_preds[0, ti_yr].mean(axis=0) - free_clims[0][cal_month]) -
                    (free_preds[0, ti_cmp].mean(axis=0) - free_clims[0][cal_month])))

        fig2, axes2 = plt.subplots(n_rows, n_cols,
                                   figsize=(n_cols * 2.5, n_rows * 3),
                                   subplot_kw={'projection': map_proj},
                                   layout='constrained')

        for ri, (fname, rlabel) in enumerate(zip(row_fields, row_labels)):
            darrs = diff_arrays[fname]
            all_vals2 = np.concatenate([a.compressed() for a in darrs])
            vabs = np.nanpercentile(np.abs(all_vals2), 98)
            for ci, (diff, (_, _, col_label)) in enumerate(zip(darrs, rows)):
                ax = axes2[ri, ci]
                im = ax.pcolormesh(nav_lon_gom, nav_lat_gom, diff,
                                   transform=data_proj, cmap=cmap, vmin=-vabs, vmax=vabs)
                _setup_geoax(ax, nav_lon_gom, nav_lat_gom, land_mask_gom, data_proj)
                if ri == 0:
                    ax.set_title(col_label, fontsize=9)
                if ci == 0:
                    ax.set_ylabel(rlabel, fontsize=9)
            cbar_label = {'pred': 'Diff (W m⁻²)', 'vohfe': 'Diff (W m⁻²)',
                          'von2': 'Diff (s⁻²)', 'free': 'Diff (a.u.)'}.get(fname, 'Diff')
            fig2.colorbar(im, ax=axes2[ri, :], shrink=0.8, pad=0.02, format='%.1e', label=cbar_label)

        fig2.suptitle(f'Gulf of Maine: {year} minus {compare_year} anomaly — last {n_lookback} months ending {month:02d}  opa{member}',
                      fontsize=11)
        path2 = os.path.join(save_dir, f'gom_diff_{year}vs{compare_year}_{month:02d}_opa{member}.png')
        fig2.savefig(path2, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved {path2}')


def _get_sst_mhw_mask(loc, mhw_month, mhw_year, ocean_mask):
    """Helper to compute SST-based MHW mask externally (for use in plot_mhw_inputs)."""
    from scipy import signal
    Y, X      = ocean_mask.shape
    sst_ds    = xr.open_zarr(f'{loc}/opa0/sosstsst_na.zarr')
    sst_month = sst_ds.sel(time_counter=sst_ds.time_counter.dt.month == mhw_month
                           )['sosstsst'].isel(y=slice(0, Y), x=slice(0, X)).values
    sst_years = sst_ds.sel(time_counter=sst_ds.time_counter.dt.month == mhw_month
                           ).time_counter.dt.year.values
    sst_clim  = np.nanmean(sst_month, axis=0)
    sst_anom  = sst_month - sst_clim
    nan_mask  = np.isnan(sst_anom)
    sst_anom[nan_mask] = 0.0
    sst_anom[:] = signal.detrend(sst_anom, axis=0)
    sst_anom[nan_mask] = np.nan
    sst_thresh = np.nanpercentile(sst_anom, 90, axis=0)
    year_idx   = np.where(sst_years == mhw_year)[0][0]
    return np.where(ocean_mask, sst_anom[year_idx] > sst_thresh, False)

def plot_gom_events(model_dirs, nc_path, year, month, pct=90,
                    n_members=5,
                    opas=None, Y=302, X=400,
                    gom_lat=(39, 46), gom_lon=(-71, -62),
                    domain_lat=(20, 66), domain_lon=(-80, 20)):
    """3-panel GOM event plot: SST events | true MLHC events | pred MLHC events.

    Events are defined as anomaly > pct-th percentile threshold (computed from
    the full zarr record for true, from val+test preds for predicted).
    Thresholds for true MLHC and pred MLHC are computed independently.

    Parameters
    ----------
    model_dirs       : str or list of str — model dirs with val_preds_lead0.npz
    nc_path          : ORCA025 .nc file with nav_lat/nav_lon
    year, month      : event month to highlight
    pct              : percentile threshold for event definition (default 90)
    val_start_year/month : start of val period (for indexing preds)
    """
    import calendar as cal_mod
    import pandas as pd
    from matplotlib.colors import ListedColormap

    if isinstance(model_dirs, str):
        model_dirs = [model_dirs]
    if opas is None:
        opas = [f'opa{i}' for i in range(n_members)]

    # load config from first model dir for data location
    config.read(os.path.join(model_dirs[0], 'config.ini'))
    loc = config['DATASET']['location']

    # ocean mask
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    ocean_mask = mesh['tmaskutil'].isel(t=0, y=slice(0, Y), x=slice(0, X)).values == 1

    # GOM crop
    ds_nc = xr.open_dataset(nc_path)
    domain_mask = (
        (ds_nc['nav_lat'] >= domain_lat[0]) & (ds_nc['nav_lat'] <= domain_lat[1]) &
        (ds_nc['nav_lon'] >= domain_lon[0]) & (ds_nc['nav_lon'] <= domain_lon[1])
    )
    nav_lat = ds_nc['nav_lat'].isel(y=np.where(domain_mask.any(dim='x'))[0],
                                    x=np.where(domain_mask.any(dim='y'))[0]).values
    nav_lon = ds_nc['nav_lon'].isel(y=np.where(domain_mask.any(dim='x'))[0],
                                    x=np.where(domain_mask.any(dim='y'))[0]).values
    gom_mask = ((nav_lat >= gom_lat[0]) & (nav_lat <= gom_lat[1]) &
                (nav_lon >= gom_lon[0]) & (nav_lon <= gom_lon[1]))[:Y, :X]
    y_inds, x_inds = np.where(gom_mask)
    y_min, y_max = y_inds.min(), y_inds.max()
    x_min, x_max = x_inds.min(), x_inds.max()
    land_gom = ~ocean_mask[y_min:y_max+1, x_min:x_max+1]
    nav_lat_gom = nav_lat[:Y, :X][y_min:y_max+1, x_min:x_max+1]
    nav_lon_gom = nav_lon[:Y, :X][y_min:y_max+1, x_min:x_max+1]

    def _crop_event(mask2d):
        sub = mask2d[y_min:y_max+1, x_min:x_max+1].astype(float)
        return np.ma.masked_where(land_gom, sub)

    # --- True SST events ---
    sst_data = []
    for opa in opas:
        ds = xr.open_zarr(f'{loc}/{opa}/sosstsst_na.zarr')
        sst_data.append(ds['sosstsst'].isel(y=slice(0, Y), x=slice(0, X)).values)
        times = ds.time_counter.values
    sst_all = np.nanmean(sst_data, axis=0)
    tidx = pd.DatetimeIndex(times)
    sst_m = sst_all[tidx.month == month]
    sst_clim = np.nanmean(sst_m, axis=0)
    sst_anom = sst_m - sst_clim
    sst_thresh = np.nanpercentile(sst_anom, pct, axis=0)
    yr_i = np.where(tidx[tidx.month == month].year == year)[0][0]
    sst_events = (sst_anom[yr_i] > sst_thresh) & ocean_mask

    # --- True MLHC events ---
    mlhc_data = []
    for opa in opas:
        ds = xr.open_zarr(f'{loc}/{opa}/vomlhc_na.zarr')
        mlhc_data.append(ds['vomlhc'].isel(y=slice(0, Y), x=slice(0, X)).values)
    mlhc_all = np.nanmean(mlhc_data, axis=0)
    mlhc_m = mlhc_all[tidx.month == month]
    mlhc_clim = np.nanmean(mlhc_m, axis=0)
    mlhc_anom = mlhc_m - mlhc_clim
    mlhc_thresh = np.nanpercentile(mlhc_anom, pct, axis=0)
    mlhc_events = (mlhc_anom[yr_i] > mlhc_thresh) & ocean_mask

    # --- Pred MLHC events (from all_preds.npz — full record) ---
    raw_preds = []
    for md in model_dirs:
        r = np.load(os.path.join(md, 'all_preds.npz'), allow_pickle=True)
        out_std  = r['output_std']
        out_mean = r['output_mean']
        raw_preds.append(r['preds'] * out_std + out_mean)
    preds = np.mean(raw_preds, axis=0)   # (N, Y, X) — denormalized, full record

    N = preds.shape[0]
    n_times = N // n_members
    preds_r = preds.reshape(n_times, n_members, Y, X)
    pred_mean = preds_r.mean(axis=1)     # (n_times, Y, X)

    # month and year for each dataset timestep (interleaved layout, window=6, start=1979-01)
    start_year  = int(config.get('DATASET', 'start').split('-')[0])
    start_month = int(config.get('DATASET', 'start').split('-')[1]) - 1  # 0=Jan
    window      = config.getint('DATASET', 'context_window')
    pred_months = np.array([(start_month + t + window) % 12 + 1 for t in range(n_times)])
    pred_years  = np.array([start_year + (start_month + t + window) // 12 for t in range(n_times)])

    pred_m = pred_mean[pred_months == month]
    pred_clim   = np.nanmean(pred_m, axis=0)
    pred_anom   = pred_m - pred_clim
    pred_thresh = np.nanpercentile(pred_anom, pct, axis=0)
    yr_pred_i   = np.where(pred_years[pred_months == month] == year)[0]
    if len(yr_pred_i) == 0:
        raise ValueError(f'Year {year} month {month} not found in all_preds')
    pred_events = (pred_anom[yr_pred_i[0]] > pred_thresh) & ocean_mask

    # --- Overlap: pred MLHC vs SST events ---
    intersection = (pred_events & sst_events & ocean_mask).sum()
    union        = ((pred_events | sst_events) & ocean_mask).sum()
    jaccard      = 100 * intersection / union if union > 0 else 0.0
    pct_sst_captured = 100 * intersection / sst_events.sum() if sst_events.sum() > 0 else 0.0
    print(f'Pred MLHC vs SST overlap — Jaccard: {jaccard:.1f}%  |  SST events captured by pred: {pct_sst_captured:.1f}%')

    # --- Plot ---
    cmap_event = ListedColormap(['white', 'red'])
    month_name = cal_mod.month_abbr[month]

    data_proj = ccrs.PlateCarree()
    map_proj  = ccrs.Mercator()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained',
                             subplot_kw={'projection': map_proj})
    titles = [f'SST Events', f'True MLHC Events', f'Pred MLHC Events']
    for ax, ev, title in zip(axes, [sst_events, mlhc_events, pred_events], titles):
        ax.pcolormesh(nav_lon_gom, nav_lat_gom, _crop_event(ev),
                      transform=data_proj, cmap=cmap_event, vmin=0, vmax=1, zorder=3)
        _setup_geoax(ax, nav_lon_gom, nav_lat_gom, land_gom, data_proj)
        ax.set_title(title)

    fig.suptitle(f'Gulf of Maine MHW events (>{pct}th pct) — {month_name} {year}', fontsize=11)
    path = '/path/to/data/paper_figs/gom_events_{}{:02d}.png'.format(year, month)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')

if __name__ == "__main__":
    # model_dir = '/path/to/data/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v6'
    # #all_preds_path = f'{model_dir}/all_preds.npz'
    # config.read(f'{model_dir}/config.ini')
    # loc = config['DATASET']['location']
    # model_dir = '/path/to/data/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v12'
    # save_val_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader, test_loader=test_loader, output_dir=model_dir)
    # find_mlhc_heatwave_years(loc=loc, month=7, save_dir=model_dir)

    # analyze_mhw(model_dir=model_dir, mhw_year=2008, mhw_month=7,
    #             all_preds_path=all_preds_path, all_start_year=1979, all_start_month=1)
    # import numpy as np
    # import matplotlib.pyplot as plt

    #Your weight data
    # models = np.array([
    #     [ 0.0383,  0.1476, -0.0953, -0.0574, -0.1465,  0.3118], # v1
    #     [-0.0103, -0.1007,  0.0110,  0.1862, -0.1373,  0.3492], # v2
    #     [-0.0377,  0.1309, -0.1254,  0.1145, -0.0518, -0.3326], # v3
    #     [-0.1613, -0.1455, -0.1664,  0.1050, -0.1329, -0.1685], # v4
    #     [ 0.0010,  0.1037, -0.0359, -0.1144,  0.0840,  0.3211]  # v5
    # ])

    # concepts = 4, free = 1
    #models = np.array([[-0.06738445907831192, 0.001842171186581254, 0.3259401023387909, -0.07592181861400604, -0.12493477016687393], [-0.02637508325278759, -0.0035374246072024107, -0.2135140299797058, -0.009366443380713463, -0.23865735530853271], [0.05817287042737007, 0.04584785923361778, 0.26516133546829224, 0.21162927150726318, 0.27583619952201843], [0.08272426575422287, 0.1320900022983551, 0.2300870567560196, 0.19920714199543, -0.21609172224998474], [0.1487811952829361, -0.1177094504237175, 0.07849641144275665, -0.04589887335896492, 0.2955295145511627]])
    # concepts = 4, free = 0
    #models = np.array([[0.17457669973373413, -1.838305115699768, -0.01666351966559887, 0.2898353040218353], [0.1574288159608841, -1.7596479654312134, 0.3195071816444397, -0.106661856174469], [0.0027659342158585787, -0.08824791014194489, 1.2545744180679321, -1.0346895456314087], [0.18134212493896484, -1.9464365243911743, 0.6096028685569763, -0.3804885447025299], [-0.006851653102785349, -0.04464983567595482, 1.2734904289245605, -1.0549668073654175]])
    # unsup
    # models = np.array([[-0.030259542167186737, 0.009104108437895775, 0.3854852020740509, -0.05901351198554039, -1.5056997426654561e-06], [-0.04361393302679062, -0.00040911146788857877, -0.3518856465816498, -0.01356587279587984, -0.1765439510345459], [0.01722194068133831, 0.08543573319911957, 0.34384095668792725, 0.3721291124820709, 0.09240081906318665], [0.1328747719526291, 0.2375536412000656, 0.3160589635372162, 0.36930859088897705, -0.08267282694578171], [0.18817049264907837, -0.2083168923854828, 0.008376712910830975, -0.0047642532736063, 0.21402356028556824]])
    # #labels = ['Buoyancy freq.', 'Shear', 'Heat flux ent.', 'MLD tend.']
    # # labels = ['von2', 'vos2', 'vohfe', 'mxl_tendency']
    # labels = ['Rep. 1', 'Rep. 2', 'Rep. 3', 'Rep. 4', 'Rep. 5']
    # # # labels = ['Buoyancy freq.', 'Shear', 'Heat flux', 'MLD tend.', 'Free']
    # model_ids = ['v1', 'v2', 'v3', 'v4', 'v5']

    # # 1. Take absolute values for "importance"
    # abs_models = np.abs(models)

    # # 2. Normalize so each model (bar) sums to 1.0 (100%)
    # norm_models = abs_models / abs_models.sum(axis=1, keepdims=True)

    # # 3. Plotting
    # #colors = ['#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#084594', '#FF7F50']
    # colors = ['#E5989B', '#FFB480', '#B5C99A', '#8ECAE6', '#957DAD', '#83C5BE']
    # fig, ax = plt.subplots(figsize=(4, 3))
    # bottom = np.zeros(len(model_ids))

    # for i, concept in enumerate(labels):
    #     ax.bar(model_ids, norm_models[:, i], bottom=bottom, 
    #         label=concept, color=colors[i], edgecolor='white', width=0.6)
    #     bottom += norm_models[:, i]

    # # Formatting
    # ax.set_ylabel('Relative absolute contribution')
    # ax.set_xlabel('Model ensemble member')
    # #ax.set_title('Prediction-only')
    # ax.set_ylim(0, 1.15)
    # ax.legend(loc='upper right', fontsize='x-small', framealpha=0.4)
    # ax.spines[['top', 'right']].set_visible(False)

    # plt.tight_layout()
    # fig.savefig('bar_unsup.pdf')
    # breakpoint()
            
    # model_dirs = ['/path/to/data/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v5',
    #     '/path/to/data/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v6',
    #     '/path/to/data/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v8']
    # plot_gom_events(model_dirs=model_dirs, nc_path=nc_path, year=2012, month=8) 
    
    # model_dir = '/path/to/data/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v10'
    # save_val_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader, test_loader=test_loader, output_dir=model_dir)
    # # find_mlhc_heatwave_years(loc=loc, month=7, save_dir=model_dir)

    # # analyze_mhw(model_dir=model_dir, mhw_year=2008, mhw_month=7,
    # #             all_preds_path=all_preds_path, all_start_year=1979, all_start_month=1)
   
    nc_path = '/path/to/scratch/temp_project/sosstsst_ORAS5_1m_201801_grid_T_02.nc'
    dirs_free1 = [f'/path/to/data/paper_cbm/concepts_4/free1/UNetCBM_adaptive_ep101_lr0.001_bs64_L1Loss_ZScore_v{v}' for v in range(1, 6)]
    all_preds_path = '/path/to/data/paper_cbm/concepts_4/free1/UNetCBM_adaptive_ep101_lr0.001_bs64_L1Loss_ZScore_v1/all_preds.npz'
    analyze_gom(model_dirs=dirs_free1, nc_path=nc_path, year=2000, month=8, member=0,
                val_start_year=2011, val_start_month=5, n_members=5,
                n_lookback=6, save_dir='/path/to/repo/paper_figs', compare_year=None,
                gom_lat=(39, 46), gom_lon=(-71, -62),
                domain_lat=(20, 66), domain_lon=(-80, 20),
                all_preds_path=all_preds_path, all_start_year=1979, all_start_month=1)
    
    #plot_gom_events(model_dirs=cbm_dirs, nc_path=nc_path, year=2012, month=8)

    # SHAP: input -> concepts for Aug 2012 opa0 (val index 75)
    # config.read(f'{model_dir}/config.ini')
    # nc_path = '/path/to/data/oras5/ORCA025/sosstsst/opa0/sosstsst_ORAS5_1m_201208_grid_T_02.nc'
    # for label, val_idx in [('aug2012', 75), ('aug2011', 15)]:
    #     gradient_shap_inputs(model_dir=model_dir,
    #                          output_dir=model_dir,
    #                          config_path=f'{model_dir}/config.ini',
    #                          val_sample_idx=val_idx, nc_path=nc_path,
    #                          gom_suffix=label)

    # model_dir   = '/path/to/data/runs_040826/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore'
    # config_path = f'{model_dir}/config.ini'
    # results    = np.load(os.path.join(model_dir, 'val_preds_lead0.npz'), allow_pickle=True)
    # ocean_mask = results['ocean_mask']
    # mhw_mask   = _get_sst_mhw_mask(loc, mhw_month=7, mhw_year=2012, ocean_mask=ocean_mask)
    # plot_mhw_inputs(model_dir=model_dir, config_path=config_path,
    #                 mhw_mask=mhw_mask, mhw_year=2012, compare_year=2011)
    # gradient_shap_free_concept(model_dir=model_dir, input_norm=None, val_loader=None,
    #                            free_concept_idx=0, n_baselines=20,
    #                            output_dir=model_dir, config_path=config_path)


