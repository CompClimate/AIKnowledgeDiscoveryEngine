import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import os
import xarray as xr
import numpy as np

def plot_var(var, member, year, month):
    if var == 'vomlhc':
        ds = xr.open_dataset(f'/quobyte/maikesgrp/sanah/target/vomlhc/{member}/vomlhc_{year}{month:02d}_T.nc')
        lon_bounds=(-80, 20),
        lat_bounds=(20, 66)
        mask = ((ds.nav_lon >= lon_bounds[0]) & (ds.nav_lon <= lon_bounds[1]) & (ds.nav_lat >= lat_bounds[0]) & (ds.nav_lat <= lat_bounds[1]))
        y_inds = mask.any(dim="x")
        x_inds = mask.any(dim="y")
        ds_na = ds.isel(y=y_inds, x=x_inds)
        fig, ax = plt.subplots()
        im = ax.imshow(ds_na.vomlhc.squeeze(), cmap='RdYlBu', origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'{str.upper(var)} {member}: {year}{month}')
        fig.savefig(f'figs/{var}_{member}_{year}{month}')
    else:
        ds = xr.open_zarr(f'/quobyte/maikesgrp/sanah/na_crop_latest/{member}/{var}_na.zarr')[var]
        date_str = f'{str(year)}-{month:02d}'
        ds_plot = ds.sel(time_counter = date_str).squeeze()
        fig, ax = plt.subplots()
        im = ax.imshow(ds_plot, cmap='RdYlBu', origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'{str.upper(var)} {member}: {year}{month}')
        fig.savefig(f'figs/{var}_{member}_{year}{month}')
    print(f'saved {var}')

def plot_var_eda(vars, member='opa0',
                 loc='/quobyte/maikesgrp/sanah/na_crop_latest',
                 smooth_sigma=None, clip_pct=(2, 98)):
    """One figure per variable with: seasonal cycle, 3 distributions (raw,
    clipped, log-y), preprocessed distribution (smooth+clip), std map,
    and 4 seasonal mean maps.

    smooth_sigma: gaussian sigma to apply before clipping (None = no smoothing)
    clip_pct: (low, high) percentiles for clipping, default (2, 98)
    """
    from scipy.ndimage import gaussian_filter
    import os
    os.makedirs('figs', exist_ok=True)

    mask_ds = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    ocean_mask = mask_ds['tmaskutil'].isel(t=0).isel(y=slice(0, 302), x=slice(0, 400)).values == 1

    # map zarr filename -> variable name inside the file
    var_name_map = {
        'vozocrtx_mld': 'vozocrtx',
        'vomecrty_mld': 'vomecrty',
    }

    for var in vars:
        print(f'Loading {var}...', flush=True)
        ds = xr.open_zarr(f'{loc}/{member}/{var}_na.zarr')
        varname = var_name_map.get(var, var)
        arr = ds[varname].isel(y=slice(0, 302), x=slice(0, 400)).values  # (T, Y, X)
        times = ds['time_counter'].values

        arr = arr.astype(float)
        arr[:, ~ocean_mask] = np.nan

        T = arr.shape[0]
        months = np.array([int(str(t)[:7].replace('-', '')[4:6]) for t in times])

        spatial_mean = np.nanmean(arr, axis=(1, 2))

        # Seasonal cycle: mean and std per calendar month
        monthly_mean = np.array([np.nanmean(spatial_mean[months == m]) for m in range(1, 13)])
        monthly_std  = np.array([np.nanstd(spatial_mean[months == m])  for m in range(1, 13)])

        std_map = np.nanstd(arr, axis=0)

        # Histogram values (subsample for speed)
        vals = arr[~np.isnan(arr)].ravel()
        if len(vals) > 500_000:
            vals = np.random.choice(vals, 500_000, replace=False)

        # Preprocessed: smooth then clip
        arr_proc = arr.copy()
        if smooth_sigma is not None:
            for t in range(arr_proc.shape[0]):
                nan_mask = np.isnan(arr_proc[t])
                arr_proc[t] = np.where(nan_mask, 0.0, arr_proc[t])
                arr_proc[t] = gaussian_filter(arr_proc[t], sigma=smooth_sigma)
                arr_proc[t][nan_mask] = np.nan
        plo = np.nanpercentile(arr_proc, clip_pct[0])
        phi = np.nanpercentile(arr_proc, clip_pct[1])
        arr_proc = np.clip(arr_proc, plo, phi)
        vals_proc = arr_proc[~np.isnan(arr_proc)].ravel()
        if len(vals_proc) > 500_000:
            vals_proc = np.random.choice(vals_proc, 500_000, replace=False)

        # Seasonal means (DJF, MAM, JJA, SON)
        season_months = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
                         'JJA': [6, 7, 8],  'SON': [9, 10, 11]}
        season_maps = {s: np.nanmean(arr[np.isin(months, m)], axis=0)
                       for s, m in season_months.items()}
        # Shared colorscale across seasons
        all_season_vals = np.concatenate([v[ocean_mask] for v in season_maps.values()])
        svmin = np.nanpercentile(all_season_vals, 2)
        svmax = np.nanpercentile(all_season_vals, 98)

        p1, p99 = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
        p2_v, p98_v = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
        vals_clipped = vals[(vals >= p2_v) & (vals <= p98_v)]

        fig, axes = plt.subplots(2, 5, figsize=(24, 8), layout='constrained')
        fig.suptitle(f'{var} — {member}', fontsize=13)

        # 1. Normalized distribution (what the model sees)
        norm_mean = np.nanmean(vals)
        norm_std  = np.nanstd(vals)
        vals_norm = (vals - norm_mean) / norm_std
        ax = axes[0, 0]
        ax.hist(vals_norm, bins=120, edgecolor='none', alpha=0.85, color='steelblue')
        for s, c in [(-3, 'red'), (-2, 'orange'), (-1, 'green'),
                      (1, 'green'),  (2, 'orange'),  (3, 'red')]:
            ax.axvline(s, color=c, linestyle='--', linewidth=0.8, alpha=0.7)
        frac_within1 = np.mean(np.abs(vals_norm) <= 1)
        frac_within3 = np.mean(np.abs(vals_norm) <= 3)
        ax.set_title(f'Normalized (±1σ: {frac_within1:.0%}, ±3σ: {frac_within3:.0%})')
        ax.set_xlabel('(val - mean) / std')

        # 2. Seasonal cycle
        ax = axes[0, 1]
        ax.plot(range(1, 13), monthly_mean, marker='o', color='darkorange')
        ax.fill_between(range(1, 13), monthly_mean - monthly_std,
                        monthly_mean + monthly_std, alpha=0.25, color='darkorange')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax.set_title('Seasonal cycle (±1 std)')

        # 3. Distribution (full range, linear)
        ax = axes[0, 2]
        ax.hist(vals, bins=120, edgecolor='none', alpha=0.85, color='slategray')
        ax.axvline(np.nanmean(vals), color='red',   linestyle='--', linewidth=1, label=f'mean={np.nanmean(vals):.3g}')
        ax.axvline(p1,              color='orange', linestyle=':',  linewidth=1, label=f'1st={p1:.3g}')
        ax.axvline(p99,             color='orange', linestyle=':',  linewidth=1, label=f'99th={p99:.3g}')
        ax.set_title('Distribution (full)')
        ax.legend(fontsize=7)

        # 4. Distribution clipped to 5–95th pct
        ax = axes[0, 3]
        ax.hist(vals_clipped, bins=120, edgecolor='none', alpha=0.85, color='slategray')
        ax.axvline(np.nanmean(vals), color='red', linestyle='--', linewidth=1,
                   label=f'mean={np.nanmean(vals):.3g}')
        ax.set_title('Distribution (2–98th pct)')
        ax.legend(fontsize=7)

        # 5. Distribution of log10-transformed values (positive only)
        ax = axes[0, 4]
        vals_pos = vals[vals > 0]
        if len(vals_pos) > 0:
            log_vals = np.log10(vals_pos)
            log_mean = np.nanmean(log_vals)
            log_p1   = np.nanpercentile(log_vals, 1)
            log_p99  = np.nanpercentile(log_vals, 99)
            ax.hist(log_vals, bins=120, edgecolor='none', alpha=0.85, color='slategray')
            ax.axvline(log_mean, color='red',   linestyle='--', linewidth=1, label=f'mean={log_mean:.3g}')
            ax.axvline(log_p1,  color='orange', linestyle=':',  linewidth=1, label=f'1st={log_p1:.3g}')
            ax.axvline(log_p99, color='orange', linestyle=':',  linewidth=1, label=f'99th={log_p99:.3g}')
            ax.set_xlabel('log10(value)')
            frac_pos = len(vals_pos) / len(vals)
            ax.set_title(f'Distribution (log10, {frac_pos:.0%} positive)')
        else:
            ax.text(0.5, 0.5, 'No positive values', transform=ax.transAxes, ha='center')
            ax.set_title('Distribution (log10)')
        ax.legend(fontsize=7)

        # 5. Std map
        ax = axes[1, 0]
        std_masked = np.ma.masked_where(~ocean_mask, std_map)
        im = ax.imshow(std_masked, cmap='viridis', origin='lower', aspect='equal')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Temporal std dev')
        ax.axis('off')

        # 6–9. Seasonal mean maps
        for i, (sname, smap) in enumerate(season_maps.items()):
            ax = axes[1, i + 1]
            masked = np.ma.masked_where(~ocean_mask, smap)
            im = ax.imshow(masked, cmap='RdYlBu_r', origin='lower',
                           vmin=svmin, vmax=svmax, aspect='equal')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(sname)
            ax.axis('off')

        save_name = f'figs/eda_{var}_{member}_norm.png'
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_name}')

def plot_member_comparison(vars, members=None,
                           loc='/quobyte/maikesgrp/sanah/na_crop_latest'):
    """One figure per variable comparing all members.
    3 panels: seasonal cycle, spatial mean time series, distribution.
    """
    import os
    os.makedirs('figs', exist_ok=True)

    if members is None:
        members = ['opa0', 'opa1', 'opa2', 'opa3', 'opa4']

    var_name_map = {'vozocrtx_mld': 'vozocrtx', 'vomecrty_mld': 'vomecrty'}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    mask_ds = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    ocean_mask = mask_ds['tmaskutil'].isel(t=0).isel(y=slice(0, 302), x=slice(0, 400)).values == 1

    for var in vars:
        print(f'{var}...', flush=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), layout='constrained')
        fig.suptitle(var, fontsize=13)

        for mi, member in enumerate(members):
            c = colors[mi % len(colors)]

            ds = xr.open_zarr(f'{loc}/{member}/{var}_na.zarr')
            varname = var_name_map.get(var, var)
            arr = ds[varname].isel(y=slice(0, 302), x=slice(0, 400)).values.astype(float)
            times = ds['time_counter'].values
            arr[:, ~ocean_mask] = np.nan

            months = np.array([int(str(t)[:7].replace('-', '')[4:6]) for t in times])
            spatial_mean = np.nanmean(arr, axis=(1, 2))

            # Seasonal cycle
            monthly_mean = np.array([np.nanmean(spatial_mean[months == m]) for m in range(1, 13)])
            axes[0].plot(range(1, 13), monthly_mean, marker='o', markersize=3,
                         linewidth=1, color=c, label=member)

            # Time series
            axes[1].plot(times, spatial_mean, linewidth=0.6, color=c, alpha=0.8, label=member)

            # Distribution
            vals = arr[~np.isnan(arr)].ravel()
            if len(vals) > 200_000:
                vals = np.random.choice(vals, 200_000, replace=False)
            axes[2].hist(vals, bins=80, edgecolor='none', alpha=0.4, color=c, label=member)

        axes[0].set_xticks(range(1, 13))
        axes[0].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        axes[0].set_title('Seasonal cycle')
        axes[0].legend(fontsize=7)

        axes[1].set_title('Spatial mean time series')
        axes[1].set_xlabel('Date')
        axes[1].legend(fontsize=7)

        axes[2].set_title('Distribution')
        axes[2].legend(fontsize=7)

        save_name = f'figs/member_comparison_{var}.png'
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_name}')


def plot_preprocessing_maps(vars, member='opa0',
                             loc='/quobyte/maikesgrp/sanah/na_crop_latest',
                             log_vars=None):
    """4x4 grid showing effect of preprocessing on seasonal climatology maps.

    Columns: DJF, MAM, JJA, SON
    Rows: raw (log10 for log_vars), clipped(2,98),
          smooth σ=2 + clipped, smooth σ=3 + clipped
    Colorscale is consistent within each row.
    """
    from scipy.ndimage import gaussian_filter
    import os
    os.makedirs('figs', exist_ok=True)

    if log_vars is None:
        log_vars = ['vori', 'von2', 'vos2']
    log_vars = []
    var_name_map = {'vozocrtx_mld': 'vozocrtx', 'vomecrty_mld': 'vomecrty'}

    mask_ds = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    ocean_mask = mask_ds['tmaskutil'].isel(t=0).isel(y=slice(0, 302), x=slice(0, 400)).values == 1

    season_months = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
                     'JJA': [6, 7, 8],  'SON': [9, 10, 11]}
    seasons = list(season_months.keys())

    row_labels = ['raw', 'clipped (2–98)', 'smooth σ=2 + clipped', 'smooth σ=3 + clipped']

    for var in vars:
        print(f'{var}...', flush=True)
        ds = xr.open_zarr(f'{loc}/{member}/{var}_na.zarr')
        varname = var_name_map.get(var, var)
        arr = ds[varname].isel(y=slice(0, 302), x=slice(0, 400)).values.astype(float)
        times = ds['time_counter'].values
        arr[:, ~ocean_mask] = np.nan

        months = np.array([int(str(t)[:7].replace('-', '')[4:6]) for t in times])

        # Build 4 preprocessed versions
        def seasonal_maps(a):
            return {s: np.nanmean(a[np.isin(months, m)], axis=0)
                    for s, m in season_months.items()}

        # Row 0: raw (log10 for log_vars)
        arr0 = arr.copy()
        if var in log_vars:
            arr0 = np.where(arr0 > 0, np.log10(arr0), np.nan)
        maps0 = seasonal_maps(arr0)

        # Row 1: clipped
        p2, p98 = np.nanpercentile(arr0[~np.isnan(arr0)], 2), np.nanpercentile(arr0[~np.isnan(arr0)], 98)
        arr1 = np.clip(arr0, p2, p98)
        maps1 = seasonal_maps(arr1)

        # Row 2: smooth σ=2 + clipped
        arr2 = arr0.copy()
        for t in range(arr2.shape[0]):
            nm = np.isnan(arr2[t])
            arr2[t] = np.where(nm, 0.0, arr2[t])
            arr2[t] = gaussian_filter(arr2[t], sigma=2)
            arr2[t][nm] = np.nan
        arr2 = np.clip(arr2, p2, p98)
        maps2 = seasonal_maps(arr2)

        # Row 3: smooth σ=3 + clipped
        arr3 = arr0.copy()
        for t in range(arr3.shape[0]):
            nm = np.isnan(arr3[t])
            arr3[t] = np.where(nm, 0.0, arr3[t])
            arr3[t] = gaussian_filter(arr3[t], sigma=3)
            arr3[t][nm] = np.nan
        arr3 = np.clip(arr3, p2, p98)
        maps3 = seasonal_maps(arr3)

        all_maps = [maps0, maps1, maps2, maps3]

        fig, axes = plt.subplots(4, 4, figsize=(16, 13), layout='constrained')
        log_tag = ' (log10)' if var in log_vars else ''
        fig.suptitle(f'{var}{log_tag} — {member}', fontsize=13)

        for ri, (row_maps, row_label) in enumerate(zip(all_maps, row_labels)):
            # Colorscale from this row's data
            row_vals = np.concatenate([v[ocean_mask] for v in row_maps.values()])
            rvmin = np.nanpercentile(row_vals, 2)
            rvmax = np.nanpercentile(row_vals, 98)

            for ci, season in enumerate(seasons):
                ax = axes[ri, ci]
                masked = np.ma.masked_where(~ocean_mask, row_maps[season])
                im = ax.imshow(masked, cmap='RdYlBu_r', origin='lower',
                               vmin=rvmin, vmax=rvmax, aspect='equal')
                if ci == 3:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if ri == 0:
                    ax.set_title(season, fontsize=11)
                if ci == 0:
                    ax.set_ylabel(row_label, fontsize=9)
                ax.axis('off')

        save_name = f'figs/preprocessing_{var}_{member}.png'
        fig.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_name}')


if __name__ == "__main__":
    vars = ['vozocrtx_ml', 'vomecrty_ml']
    plot_var_eda(vars)
    #plot_member_comparison(vars)  
    #plot_preprocessing_maps(['vori', 'von2', 'vos2'], member='opa0')