import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm 
import seawater as sw
import scipy as sp 
import os
import argparse
import multiprocessing as mp
from itertools import product
import gsw

def vertical_shear(year, month, member):

    # replace with appropriate paths
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_mv = xr.open_dataset(f'{path}/vomecrty/{member}/vomecrty_ORAS5_1m_{year}{month}_grid_V_02.nc')
    ds_zv = xr.open_dataset(f'{path}/vozocrtx/{member}/vozocrtx_ORAS5_1m_{year}{month}_grid_U_02.nc')
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')
    
    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')
    tmask = mesh['tmaskutil'][0, :, :].values   
    umask = mesh['umaskutil'][0, :, :].values   
    vmask = mesh['vmaskutil'][0, :, :].values  
    
    # get data
    mld = ds_mxl['somxl010'].values[0, :, :]  # [lat, lon]
    depths = ds_zv['depthu'].values  # [depth]
    zv = ds_zv['vozocrtx'].values[0, :, :, :]  # [depth, lat, lon]
    mv = ds_mv['vomecrty'].values[0, :, :, :] # [depth, lat, lon]

    # masking 
    mld[tmask == 0] = np.nan
    zv[:, umask == 0] = np.nan
    mv[:, vmask == 0] = np.nan

    # find bracketing indices for all grid points at once
    indices = np.searchsorted(depths, mld)  # [lat, lon]

    # clip to valid range
    indices = np.clip(indices, 1, len(depths) - 1)

    # lower and upper indices
    ld = indices - 1
    ud = indices

    # output array
    shear_zonal = np.zeros_like(mld, dtype=float)
    shear_meridional = np.zeros_like(mld, dtype=float)
    shear_vertical_sq = np.zeros_like(mld, dtype=float)

    # zv[ld, lat, lon], zv[ud, lat, lon]
    # mv[ld, lat, lon], mv[ud, lat, lon]
    # for all points
    lat_idx, lon_idx = np.ogrid[:mld.shape[0], :mld.shape[1]]

    zv_below = zv[ud, lat_idx, lon_idx]
    zv_above = zv[ld, lat_idx, lon_idx]

    mv_below = mv[ud, lat_idx, lon_idx]
    mv_above = mv[ld, lat_idx, lon_idx]

    # depth differences
    dz = depths[ud] - depths[ld]

    # shear
    shear_zonal = (zv_below - zv_above) / dz
    shear_meridional = (mv_below - mv_above) / dz

    # handle nan
    shear_zonal[np.isnan(mld)] = np.nan
    shear_meridional[np.isnan(mld)] = np.nan

    # vertical shear
    shear_vertical_sq = np.abs(shear_meridional)**2 + np.abs(shear_zonal)**2

    # probably add more info abt the year and month?
    shear_sq_da = xr.DataArray(
        shear_vertical_sq[np.newaxis, :, :],
        coords={
            'time_counter': ds_mxl['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_mxl['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_mxl['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='vertical_shear_squared')

    output_path = f'/quobyte/maikesgrp/sanah/concepts/{member}/vograds2'
    os.makedirs(output_path,  exist_ok=True)
    shear_sq_da.to_netcdf(f'{output_path}/vograds2_{year}{month}.nc')

def heat_flux(year, month, member):
    # lat = y (1021), lon = x (1442)
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_pt = xr.open_dataset(f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')
    
    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')
    tmask = mesh['tmaskutil'][0, :, :].values 

    # get data
    mld = ds_mxl['somxl010'].values[0, :, :]  # [lat, lon]
    depths = ds_pt['deptht'].values  # [depth]
    pt = ds_pt['votemper'].values[0, :, :, :]  # [depth, lat, lon]

    mld[tmask == 0] = np.nan
    pt[:, tmask == 0] = np.nan

    # find bracketing indices for all grid points at once
    indices = np.searchsorted(depths, mld)  # [lat, lon]

    # clip to valid range
    indices = np.clip(indices, 1, len(depths) - 1)

    # lower and upper indices
    ld = indices - 1
    ud = indices

    # output array
    heat_diff = np.zeros_like(mld, dtype=float)

    # pt[ld, lat, lon], pt[ud, lat, lon]
    # for all points
    lat_idx, lon_idx = np.ogrid[:mld.shape[0], :mld.shape[1]]

    pt_below = pt[ud, lat_idx, lon_idx]
    pt_above = pt[ld, lat_idx, lon_idx]

    # shear
    heat_entrainment = np.abs(pt_below - pt_above)

    # handle nan
    heat_entrainment[np.isnan(mld)] = np.nan

    heat_entrainment = heat_entrainment[np.newaxis, :, :]  # shape now (1, y, x)

    # probably add more info abt the year and month?
    heat_flux_da = xr.DataArray(
        heat_entrainment,  
        coords={
            'time_counter': ds_mxl['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_mxl['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_mxl['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='heat_flux_entrainment' 
    )

    output_path = f'/quobyte/maikesgrp/sanah/concepts/{member}/votempdiff'
    os.makedirs(output_path,  exist_ok=True)
    heat_flux_da.to_netcdf(f'{output_path}/votempdiff_{year}{month}.nc')

def rho(sal, temp, d, lat):
    p = sw.eos80.pres(d, lat)
    return sw.eos80.dens(sal, temp, p)

def brunt_vaisala(year, month, member):
    # lat = y (1021), lon = x (1442)
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_pt = xr.open_dataset(f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_sal = xr.open_dataset(f'{path}/vosaline/{member}/vosaline_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')

    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')
    tmask = mesh['tmaskutil'][0, :, :].values   

    # get data
    mld = ds_mxl['somxl010'].values[0, :, :]  # [lat, lon]
    depths = ds_pt['deptht'].values  # [depth]
    pt = ds_pt['votemper'].values[0, :, :, :]  # [depth, lat, lon]
    sal = ds_sal['vosaline'].values[0, :, :, :] # [depth, lat, lon]

    mld[tmask == 0] = np.nan
    pt[:, tmask == 0] = np.nan
    sal[:, tmask == 0] = np.nan

    # find bracketing indices for all grid points at once
    indices = np.searchsorted(depths, mld)  # [lat, lon]

    # clip to valid range
    indices = np.clip(indices, 1, len(depths) - 1)

    # lower and upper indices
    ld = indices - 1
    ud = indices

    # output array
    shear_zonal = np.zeros_like(mld, dtype=float)
    shear_meridional = np.zeros_like(mld, dtype=float)
    shear_vertical_sq = np.zeros_like(mld, dtype=float)

    # zv[ld, lat, lon], zv[ud, lat, lon]
    # mv[ld, lat, lon], mv[ud, lat, lon]
    # for all points
    lat_idx, lon_idx = np.ogrid[:mld.shape[0], :mld.shape[1]]

    pt_below = pt[ud, lat_idx, lon_idx]
    pt_above = pt[ld, lat_idx, lon_idx]

    sal_below = sal[ud, lat_idx, lon_idx]
    sal_above = sal[ld, lat_idx, lon_idx]
    
    # assuming we have some density function
    rho_below = rho(sal_below, pt_below, depths[ud], ds_pt['nav_lat'].values)
    rho_above = rho(sal_above, pt_above, depths[ld], ds_pt['nav_lat'].values)

    # depth differences
    dz = depths[ud] - depths[ld]

    # brunt-vaisala freq
    drhodz = (rho_below - rho_above) / dz
    bv = np.sqrt(9.81/rho_below * drhodz)
    bv = bv[np.newaxis, :, :]

    # probably add more info abt the year and month?
    bv_da = xr.DataArray(
        bv,  
        coords={
            'time_counter': ds_mxl['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_mxl['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_mxl['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='brunt-vaisala_frequency' 
    )

    output_path = f'/quobyte/maikesgrp/sanah/concepts/{member}/vobvfreq'
    os.makedirs(output_path,  exist_ok=True)
    bv_da.to_netcdf(f'{output_path}/vobvfreq_{year}{month}.nc')
    
def richardson_number(year, month, member):
    bv = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/{member}/vobvfreq/vobvfreq_{year}{month}.nc')['brunt-vaisala_frequency'].squeeze()
    s2 = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/{member}/vograds2/vograds2_{year}{month}.nc')['vertical_shear_squared'].squeeze()
    
    s2_min = 1e-16
    ri = np.where(s2 > s2_min, bv**2 / s2, np.nan)

    # Add back singleton time dimension
    ri = xr.DataArray(
        ri[np.newaxis, :, :],
        coords={
            'time_counter': ('time_counter', [bv['time_counter'].values]),
            'nav_lat': (('y', 'x'), bv['nav_lat'].values),
            'nav_lon': (('y', 'x'), bv['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='richardson_number'
    )

    output_path = f'/quobyte/maikesgrp/sanah/concepts/{member}/vorino'
    os.makedirs(output_path, exist_ok=True)
    ri.to_netcdf(f'{output_path}/vorino_{year}{month}.nc')

def wind_stress_curl(year, month, member):
    # lat = y (1021), lon = x (1442)
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_m = xr.open_dataset(f'{path}/sometauy/{member}/sometauy_ORAS5_1m_{year}{month}_grid_V_02.nc')
    ds_z = xr.open_dataset(f'{path}/sozotaux/{member}/sozotaux_ORAS5_1m_{year}{month}_grid_U_02.nc')
    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')
    zws = ds_z['sozotaux'].values[0, :, :]  # [depth, lat, lon]
    mws = ds_m['sometauy'].values[0, :, :] # [depth, lat, lon]
    umask = mesh['umaskutil'][0, :, :].values
    vmask = mesh['vmaskutil'][0, :, :].values
    tmask = mesh['tmaskutil'][0, :, :].values
    #zws[umask == 0] = np.nan
    #mws[vmask == 0] = np.nan
    
    zws_smooth = sp.signal.savgol_filter(zws, window_length=15, polyorder=3)
    mws_smooth = sp.signal.savgol_filter(mws, window_length=15, polyorder=3)
    
    # Reapply masks after smoothing
    #zws_smooth[umask == 0] = np.nan
    #mws_smooth[vmask == 0] = np.nan
    
    e1v = mesh['e1v'].values[0, :, :]  # zonal spacing at V points
    e2u = mesh['e2u'].values[0, :, :]  # meridional spacing at U points
    
    # meridional grad wrt lon 
    dty_dx = np.gradient(mws_smooth, axis=1) / e1v  # d(tau_y)/dx
    # zonal grad wrt lat 
    dtx_dy = np.gradient(zws_smooth, axis=0) / e2u  # d(tau_x)/dy
    
    # wind stress curl
    wsc = dty_dx - dtx_dy
    wsc[tmask == 0] = np.nan
    wsc = wsc[np.newaxis, :, :] 
    wsc_da = xr.DataArray(
        wsc,  
        coords={
            'time_counter': ds_m['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_m['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_m['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='wind_stress_curl' 
    )
    output_path = f'/quobyte/maikesgrp/sanah/concepts/{member}/sowsc'
    os.makedirs(output_path, exist_ok=True)
    wsc_da.to_netcdf(f'{output_path}/sowsc_{year}{month}.nc')

def ekman_pumping(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_pt = xr.open_dataset(f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_sal = xr.open_dataset(f'{path}/vosaline/{member}/vosaline_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_wsc = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/{member}/sowsc/sowsc_{year}{month}.nc')
    
    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')
    tmask = mesh['tmaskutil'][0, :, :].values   
    fmask = mesh['fmaskutil'][0, :, :].values
    
    # data
    mld = ds_mxl['somxl010'].values[0, :, :]  # [lat, lon]
    depths = ds_pt['deptht'].values  # [depth]
    pt = ds_pt['votemper'].values[0, :, :, :]  # [depth, lat, lon]
    sal = ds_sal['vosaline'].values[0, :, :, :]  # [depth, lat, lon]
    wsc = ds_wsc['wind_stress_curl'].values

    mld[tmask == 0] = np.nan
    pt[:, tmask == 0] = np.nan
    sal[:, tmask == 0] = np.nan
    
    # bracketing indices
    indices = np.searchsorted(depths, mld)
    indices = np.clip(indices, 1, len(depths) - 1)
    ld = indices - 1
    ud = indices
    
    # potential temp and salinity at MLD base
    lat_idx, lon_idx = np.ogrid[:mld.shape[0], :mld.shape[1]]
    pt_below = pt[ud, lat_idx, lon_idx]
    sal_below = sal[ud, lat_idx, lon_idx]  
    
    # density at MLD base
    rho_below = rho(sal_below, pt_below, depths[ud], ds_mxl['nav_lat'].values)
    
    # coriolis parameter
    ff = mesh['ff'].values[0, :, :]
    ff[fmask == 0] = np.nan
    
    # ekman pumping
    ekman = wsc / (rho_below * ff)
    
    ep_da = xr.DataArray(
        ekman,  
        coords={
            'time_counter': ds_mxl['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_mxl['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_mxl['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='ekman_pumping' 
    )

    output_path = f'/quobyte/maikesgrp/sanah/concepts/{member}/voep'
    os.makedirs(output_path,  exist_ok=True)
    ep_da.to_netcdf(f'{output_path}/voep_{year}{month}.nc')

# doesnt change over time bcos depends on coriolis 
# could potentially be input, not concept 
def rossby():
    mesh = xr.open_dataset('mesh/mesh_mask.nc')
    ff = mesh['ff'].values[0, :, :] # [y, x]
    fmask = mesh['fmaskutil'][0, :, :].values
    ff[fmask == 0] = np.nan
    e2f = mesh['e2f'].values[0, :, :] 
    # gradient wrt y 
    dfdy = np.gradient(ff, axis=0)/e2f
    beta_da = xr.DataArray(
        dfdy,
        coords={
            'nav_lat': (('y', 'x'), mesh['nav_lat'].values),
            'nav_lon': (('y', 'x'), mesh['nav_lon'].values),
        },
        dims=['y', 'x'],
        name='rossby_parameter'
    )

    output_path = f'/quobyte/maikesgrp/sanah/concepts/{member}/sorp'
    os.makedirs(output_path,  exist_ok=True)
    beta_da.to_netcdf(f'{output_path}/sorp.nc')

def mxl_tendency(year, month, member):
    base_path = f'/quobyte/maikesgrp/kkringel/oras5/ORCA025/somxl010/{member}'
    output_dir = f'/quobyte/maikesgrp/sanah/concepts/{member}/mxl_tendency'
    os.makedirs(output_dir, exist_ok=True)
    
    month_str = f"{month:02d}"
    year_str = str(year)
    curr_file = f"{base_path}/somxl010_ORAS5_1m_{year_str}{month_str}_grid_T_02.nc"
    
    if not os.path.exists(curr_file):
        print(f"Missing current: {curr_file}")
        return
    
    ds_curr = xr.open_dataset(curr_file)
    
    # Previous month
    if month == 1:
        prev_yr, prev_month = year - 1, 12
    else:
        prev_yr, prev_month = year, month - 1
    prev_file = f"{base_path}/somxl010_ORAS5_1m_{prev_yr}{prev_month:02d}_grid_T_02.nc"
    
    # Next month
    if month == 12:
        next_yr, next_month = year + 1, 1
    else:
        next_yr, next_month = year, month + 1
    next_file = f"{base_path}/somxl010_ORAS5_1m_{next_yr}{next_month:02d}_grid_T_02.nc"
    
    # Safely open neighbors
    ds_prev = None
    ds_next = None
    
    try:
        ds_prev = xr.open_dataset(prev_file)
    except FileNotFoundError:
        print(f"Missing prev: {prev_file}")
    
    try:
        ds_next = xr.open_dataset(next_file)
    except FileNotFoundError:
        print(f"Missing next: {next_file}")
    
    # Compute tendency
    if ds_prev is not None and ds_next is not None:
        diff = ds_next['somxl010'].squeeze() - ds_prev['somxl010'].squeeze()
        time_diff = (ds_next['time_counter'].values[0] - ds_prev['time_counter'].values[0]) / np.timedelta64(1, 'D')
    elif ds_prev is not None:
        diff = ds_curr['somxl010'].squeeze() - ds_prev['somxl010'].squeeze()
        time_diff = (ds_curr['time_counter'].values[0] - ds_prev['time_counter'].values[0]) / np.timedelta64(1, 'D')
    elif ds_next is not None:
        diff = ds_next['somxl010'].squeeze() - ds_curr['somxl010'].squeeze()
        time_diff = (ds_next['time_counter'].values[0] - ds_curr['time_counter'].values[0]) / np.timedelta64(1, 'D')
    else:
        print(f"No neighboring files found for {year}-{month_str}")
        return
    
    tendency = (diff / time_diff).rename('mxl_tend')
    tendency = tendency.assign_coords(time_counter=ds_curr['time_counter'].values)
    
    output_file = f"{output_dir}/mxl_tendency_{year_str}{month_str}.nc"
    tendency.to_netcdf(output_file)

def cons_temp(sal, temp, d, lat, lon):
    p = sw.eos80.pres(d, lat)
    SA = gsw.SA_from_SP(sal, p, lon, lat)
    return gsw.CT_from_pt(SA, temp)

def mlhc(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_pt = xr.open_dataset(f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_sal = xr.open_dataset(f'{path}/vosaline/{member}/vosaline_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')
    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')
    
    tmask = mesh['tmaskutil'][0, :, :].values
    
    # Data
    mld = ds_mxl['somxl010'].values[0, :, :]  # [lat, lon]
    depths = ds_pt['deptht'].values  # [depth]
    pt = ds_pt['votemper'].values[0, :, :, :]  # [depth, lat, lon]
    sal = ds_sal['vosaline'].values[0, :, :, :]  # [depth, lat, lon]
    
    mld[tmask == 0] = np.nan
    pt[:, tmask == 0] = np.nan
    sal[:, tmask == 0] = np.nan
    
    # Bracketing indices
    indices = np.searchsorted(depths, mld)
    indices = np.clip(indices, 1, len(depths) - 1)
    ld = indices - 1
    ud = indices
    
    # Potential temp and salinity at MLD base
    lat_idx, lon_idx = np.ogrid[:mld.shape[0], :mld.shape[1]]
    pt_below = pt[ud, lat_idx, lon_idx]
    sal_below = sal[ud, lat_idx, lon_idx]
    
    # Density at MLD base
    rho_below = rho(sal_below, pt_below, depths[ud], ds_mxl['nav_lat'].values)
    
    # Conservative temperature at MLD base
    ct = cons_temp(sal_below, pt_below, depths[ud], ds_mxl['nav_lon'].values, ds_mxl['nav_lat'].values)
    
    # Depth brackets
    h_below = depths[ud]
    h_above = depths[ld]
    
    # Specific heat
    #cp = gsw.gsw_cp0
    cp = 3991.86795711963
    
    # Integral: trapezoid rule between upper and lower brackets
    #mlhc = rho_below * cp * ct * (h_below - h_above)
    mlhc = ct
    mlhc[tmask == 0] = np.nan
    
    mlhc_da = xr.DataArray(
        mlhc[np.newaxis, :, :],
        coords={
            'time_counter': ds_mxl['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_mxl['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_mxl['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='mixed_layer_heat_content'
    )
    
    output_path = f'/quobyte/maikesgrp/sanah/target/{member}/vomlhc'
    os.makedirs(output_path, exist_ok=True)
    mlhc_da.to_netcdf(f'{output_path}/vomlhc_{year}{month}.nc')

def plot_concept(year, month, concept_path, concept_name, plot_title, concept_label, log=False):
    ds = xr.open_dataset(concept_path)
    concept = ds[concept_name].squeeze()
    lats = ds['nav_lat']
    lons = ds['nav_lon']

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    concept_masked = np.ma.masked_invalid(concept)

    print(np.isnan(concept.values[:, -1]).all())  # True → missing last column
    print(np.nanmin(ds['nav_lon'].values), np.nanmax(ds['nav_lon'].values))

    
    if log: 
        vmin = np.nanpercentile(concept, 2)
        vmax = np.nanpercentile(concept, 98)
        concept_plot = ax.pcolormesh(lons, lats, concept_masked, transform=ccrs.PlateCarree(),
                            cmap='coolwarm', shading='auto',
                            norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        #vmin = np.nanmin(concept)
        #vmax = np.nanmax(concept)
        vmin = np.nanpercentile(concept, 2)
        vmax = np.nanpercentile(concept, 98)
    
        concept_plot = ax.pcolormesh(lons, lats, concept_masked, transform=ccrs.PlateCarree(),
                            cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)

    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=1.0, zorder=2)
    ax.coastlines(zorder=3)

    plt.colorbar(concept_plot, ax=ax, orientation='vertical', label=concept_label)
    plt.title(plot_title)
    plt.savefig(f'{concept_name}_{year}{month}')

def run_task(args):
    yr, mn, mem = args
    try:
        vertical_shear(yr, mn, mem)
    except Exception as e:
        print(f"[ERROR] {yr}-{mn}-{mem}: {e}")

if __name__ == "__main__":
    # # ---- Parse command-line arguments ----
    # parser = argparse.ArgumentParser(description="Run vertical shear in parallel.")
    # parser.add_argument("--member", type=str, default=None,
    #                     help="Specific ensemble member to process (e.g., opa0)")
    # args = parser.parse_args()

    # # ---- Define parameter ranges ----
    # years = [str(y) for y in range(1979, 2019)]
    # months = [f"{m:02d}" for m in range(1, 13)]
    # members = ['opa0', 'opa1', 'opa2', 'opa3', 'opa4']

    # if args.member:
    #     members = [args.member]

    # tasks = list(product(years, months, members))
    # print(f"Total tasks: {len(tasks)} for members: {members}")

    # # ---- Set number of processes ----
    # nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    # print(f"Using {nproc} processes")

    # # ---- Parallel execution ----
    # with mp.Pool(nproc) as pool:
    #     pool.map(run_task, tasks)

    # years = [str(y) for y in range(1979, 2019)]
    # months = [f"{m:02d}" for m in range(1, 13)]
    # members = ['opa0', 'opa1', 'opa2', 'opa3', 'opa4']
    # years = ['1979']
    # months = ['01']
    # members = ['opa0']
    # for yr in years:
    #     for mn in months:
    #         for mem in members: 
    #             vertical_shear(yr, mn, mem)

    # vertical_shear('1979', '01', 'opa0')
    # plot_concept('1979', '01', '/quobyte/maikesgrp/sanah/concepts/opa0/vograds2/vograds2_197901.nc',
    # 'vertical_shear_squared', 'Vertical Shear Squared', '$S^2$')

    # vertical_shear('1979', '06', 'opa0')
    # plot_concept('1979', '06', '/quobyte/maikesgrp/sanah/concepts/opa0/vograds2/vograds2_197906.nc',
    # 'vertical_shear_squared', 'Vertical Shear Squared', '$S^2$')

    #brunt_vaisala('1979', '01', 'opa0')
    # plot_concept('1979', '01', '/quobyte/maikesgrp/sanah/concepts/opa0/vobvfreq/vobvfreq_197901.nc',
    # 'brunt-vaisala_frequency', 'Brunt-Vaisala Frequency', '$N (s^{-1})$')

    # brunt_vaisala('1979', '06', 'opa0')
    # plot_concept('1979', '06', '/quobyte/maikesgrp/sanah/concepts/opa0/vobvfreq/vobvfreq_197906.nc',
    # 'brunt-vaisala_frequency', 'Brunt-Vaisala Frequency', '$N (s^{-1})$')

    #richardson_number('1979', '01', 'opa0')
    # plot_concept('1979', '01', '/quobyte/maikesgrp/sanah/concepts/opa0/vorino/vorino_197901.nc',
    # 'richardson_number', 'Richardson Number', '$Ri$', True)

    #richardson_number('1979', '06', 'opa0')
    # plot_concept('1979', '06', '/quobyte/maikesgrp/sanah/concepts/opa0/vorino/vorino_197906.nc',
    # 'richardson_number', 'Richardson Number', '$Ri$', True)

    # wind_stress_curl('1979', '01', 'opa0')
    # plot_concept('1979', '01', '/quobyte/maikesgrp/sanah/concepts/opa0/sowsc/sowsc_197901.nc',
    # 'wind_stress_curl', 'Wind Stress Curl', '$N/m^3$')

    # wind_stress_curl('1979', '06', 'opa0')
    # plot_concept('1979', '06', '/quobyte/maikesgrp/sanah/concepts/opa0/sowsc/sowsc_197906.nc',
    # 'wind_stress_curl', 'Wind Stress Curl', '$N/m^3$')

    # ekman_pumping('1979', '01', 'opa0')
    # plot_concept('1979', '01', '/quobyte/maikesgrp/sanah/concepts/opa0/voep/voep_197901.nc',
    # 'ekman_pumping', 'Ekman Pumping', '$m/s$')

    # ekman_pumping('1979', '06', 'opa0')
    plot_concept('1979', '06', '/quobyte/maikesgrp/sanah/concepts/opa0/voep/voep_197906.nc',
    'ekman_pumping', 'Ekman Pumping', '$m/s$')

    # mlhc('1979', '01', 'opa0')
    # plot_concept('1979', '01', '/quobyte/maikesgrp/sanah/target/opa0/vomlhc/vomlhc_197901.nc',
    # 'mixed_layer_heat_content', 'MLHC', '$C$')



