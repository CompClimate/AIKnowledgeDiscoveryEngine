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
from visualize import concept_viz, target_viz
from scipy.ndimage import gaussian_filter
import xgcm

def mld_interface(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')
    #ds_zv = xr.open_dataset(f'{path}/vozocrtx/{member}/vozocrtx_ORAS5_1m_{year}{month}_grid_U_02.nc')
    ds_pt = ds_pt = xr.open_dataset(f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc')
    mld = ds_mxl['somxl010'].values[0, :, :]
    #depths = ds_zv['depthu'].values

    
    indices = np.searchsorted(depths, mld)
    indices = np.clip(indices, 1, len(depths) - 1)
    above_idx = indices - 1
    
    mld_interface_da = xr.DataArray(
        above_idx[np.newaxis, :, :],
        coords={
            'time_counter': ds_mxl['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_mxl['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_mxl['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='mxl_interface'
    )
    
    output_path = f'/quobyte/maikesgrp/sanah/concepts/mxl_interface/{member}'
    os.makedirs(output_path, exist_ok=True)
    mld_interface_da.to_netcdf(f'{output_path}/mxl_interface_{year}{month}.nc')

def vertical_shear(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_mv = xr.open_dataset(f'{path}/vomecrty/{member}/vomecrty_ORAS5_1m_{year}{month}_grid_V_02.nc')
    ds_zv = xr.open_dataset(f'{path}/vozocrtx/{member}/vozocrtx_ORAS5_1m_{year}{month}_grid_U_02.nc')
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')
    
    # Load pre-computed MLD interface
    ds_interface = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/mxl_interface/{member}/mxl_interface_{year}{month}.nc')
    above_idx = ds_interface['mxl_interface'].values[0, :, :]
    
    # get data
    depths = ds_zv['depthu'].values
    zv = ds_zv['vozocrtx'].values[0, :, :, :]
    mv = ds_mv['vomecrty'].values[0, :, :, :]
    
    below_idx = above_idx + 1
    
    # compute shear
    lat_idx, lon_idx = np.ogrid[:above_idx.shape[0], :above_idx.shape[1]]
    zv_below = zv[below_idx, lat_idx, lon_idx]
    zv_above = zv[above_idx, lat_idx, lon_idx]
    mv_below = mv[below_idx, lat_idx, lon_idx]
    mv_above = mv[above_idx, lat_idx, lon_idx]
    
    dz = depths[below_idx] - depths[above_idx]
    
    shear_zonal = (zv_below - zv_above) / dz
    shear_meridional = (mv_below - mv_above) / dz
    shear_vertical_sq = np.abs(shear_meridional)**2 + np.abs(shear_zonal)**2
    
    # Save
    shear_sq_da = xr.DataArray(
        shear_vertical_sq[np.newaxis, :, :],
        coords={
            'time_counter': ds_mv['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_mv['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_mv['nav_lon'].values)
        },
        dims=['time_counter', 'y', 'x'],
        name='vograds2'
    )
    
    output_path = f'/quobyte/maikesgrp/sanah/concepts/vograds2/{member}'
    os.makedirs(output_path, exist_ok=True)
    shear_sq_da.to_netcdf(f'{output_path}/vograds2_{year}{month}_T.nc')

def vertical_shear_updated(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_mv = xr.open_dataset(f'{path}/vomecrty/{member}/vomecrty_ORAS5_1m_{year}{month}_grid_V_02.nc')
    ds_zv = xr.open_dataset(f'{path}/vozocrtx/{member}/vozocrtx_ORAS5_1m_{year}{month}_grid_U_02.nc')
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')
    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')

    mvdz = ds_mv.vomecrty.differentiate(coord='depthv')
    zvdz = ds_zv.vozocrtx.differentiate(coord='depthu')

    mld = ds_mxl.somxl010

    # 2. Extract values at the nearest depth
    mvdz_at_mld = mvdz.sel(depthv=mld, method='nearest')
    zvdz_at_mld = zvdz.sel(depthu=mld, method='nearest')

    s2_at_mld = mvdz_at_mld**2 + zvdz_at_mld**2

    s2_da = xr.DataArray(
        s2_at_mld,
        coords={
            'time_counter': ds_mv['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_mv['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_mv['nav_lon'].values)
        },
        dims=['time_counter', 'y', 'x'],
        name='vos2'
    )
    
    output_path = f'/quobyte/maikesgrp/sanah/concepts/vos2/{member}'
    os.makedirs(output_path, exist_ok=True)
    s2_da.to_netcdf(f'{output_path}/vos2_{year}{month}_T.nc')
    

def heat_flux(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_pt = xr.open_dataset(f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc')
    
    ds_interface = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/mxl_interface/{member}/mxl_interface_{year}{month}.nc')
    above_idx = ds_interface['mxl_interface'].values[0, :, :]

    # get data
    depths = ds_pt['deptht'].values  # [depth]
    pt = ds_pt['votemper'].values[0, :, :, :]  # [depth, lat, lon]

    below_idx = above_idx + 1

    # pt[ld, lat, lon], pt[ud, lat, lon]
    # for all points
    lat_idx, lon_idx = np.ogrid[:above_idx.shape[0], :above_idx.shape[1]]
    pt_below = pt[below_idx, lat_idx, lon_idx]
    pt_above = pt[above_idx, lat_idx, lon_idx]

    # heat entrainment
    heat_entrainment = np.abs(pt_below - pt_above)

    # probably add more info abt the year and month?
    heat_flux_da = xr.DataArray(
        heat_entrainment[np.newaxis, :, :],  
        coords={
            'time_counter': ds_pt['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_pt['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_pt['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='votempdiff' 
    )

    output_path = f'/quobyte/maikesgrp/sanah/concepts/votempdiff/{member}'
    os.makedirs(output_path,  exist_ok=True)
    heat_flux_da.to_netcdf(f'{output_path}/votempdiff_{year}{month}_T.nc')

def rho(sal, temp, d, lat):
    p = sw.eos80.pres(d, lat)
    return sw.eos80.dens(sal, temp, p)

def brunt_vaisala_updated(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_pt = xr.open_dataset(f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_sal = xr.open_dataset(f'{path}/vosaline/{member}/vosaline_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')
    print('data loaded')
    rho_da = xr.apply_ufunc(rho, ds_sal.vosaline, ds_pt.votemper, ds_sal.deptht, ds_sal.nav_lat)
    print('rho calculated')
    mld = ds_mxl.somxl010
    drhodz = rho_da.differentiate(coord='deptht')
    print('drhodz calculated')
    rho_at_mld = drhodz.sel(deptht=mld, method='nearest')
    
    g = 9.7963 
    bv = g / 1026 * rho_at_mld

    bv_da = xr.DataArray(
        bv,  
        coords={
            'time_counter': ds_pt['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_pt['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_pt['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='von2' 
    )

    output_path = f'/quobyte/maikesgrp/sanah/concepts/von2/{member}'
    os.makedirs(output_path,  exist_ok=True)
    bv_da.to_netcdf(f'{output_path}/von2_{year}{month}_T.nc')

    

def brunt_vaisala(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_pt = xr.open_dataset(f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_sal = xr.open_dataset(f'{path}/vosaline/{member}/vosaline_ORAS5_1m_{year}{month}_grid_T_02.nc')
    
    ds_interface = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/mxl_interface/{member}/mxl_interface_{year}{month}.nc')
    above_idx = ds_interface['mxl_interface'].values[0, :, :]
    below_idx = above_idx + 1

    # get data
    depths = ds_pt['deptht'].values  # [depth]
    pt = ds_pt['votemper'].values[0, :, :, :]  # [depth, lat, lon]
    sal = ds_sal['vosaline'].values[0, :, :, :] # [depth, lat, lon]

    lat_idx, lon_idx = np.ogrid[:above_idx.shape[0], :above_idx.shape[1]]

    pt_below = pt[below_idx, lat_idx, lon_idx]
    pt_above = pt[above_idx, lat_idx, lon_idx]
    
    sal_below = sal[below_idx, lat_idx, lon_idx]
    sal_above = sal[above_idx, lat_idx, lon_idx]
    
    # assuming we have some density function
    rho_below = rho(sal_below, pt_below, depths[below_idx], ds_pt['nav_lat'].values)
    rho_above = rho(sal_above, pt_above, depths[above_idx], ds_pt['nav_lat'].values)

    # depth differences
    dz = depths[below_idx] - depths[above_idx]

    # brunt-vaisala freq
    drhodz = (rho_below - rho_above) / dz
    g = 9.7963 
    bv = g**2 / rho_below * drhodz

    # probably add more info abt the year and month?
    bv_da = xr.DataArray(
        bv[np.newaxis, :, :],  
        coords={
            'time_counter': ds_pt['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_pt['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_pt['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='von2' 
    )

    output_path = f'/quobyte/maikesgrp/sanah/concepts/von2/{member}'
    os.makedirs(output_path,  exist_ok=True)
    bv_da.to_netcdf(f'{output_path}/von2_{year}{month}_T.nc')
    
def richardson_number(year, month, member):
    # 1. Load data without squeezing so we keep the time dimension (1, y, x)
    n2_ds = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/von2/{member}/von2_{year}{month}_T.nc')
    s2_ds = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/vos2/{member}/vos2_{year}{month}_T.nc')
    
    n2 = n2_ds['von2']
    s2 = s2_ds['vos2']
    
    # 2. Perform the math (Xarray handles the coordinate alignment automatically)
    ri = n2 / s2

    # 3. Create DataArray - no need to manually reconstruct coords if you use 'ri' directly
    # 'ri' already inherited coords from n2/s2 during the math operation!
    ri.coords['nav_lat'] = n2_ds.nav_lat
    ri.coords['nav_lon'] = n2_ds.nav_lon
    ri_da = ri.rename('vori')
    
    # Optional: Handle division by zero/shear being very small
    # ri_da = ri_da.where(s2 > 1e-10, np.nan) 

    output_path = f'/quobyte/maikesgrp/sanah/concepts/vori/{member}'
    os.makedirs(output_path, exist_ok=True)
    
    # 4. Save the DataArray object, not the raw 'ri' result
    ri_da.to_netcdf(f'{output_path}/vori_{year}{month}_T.nc')

def wind_stress_curl(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_v = xr.open_dataset(f'{path}/sometauy/{member}/sometauy_ORAS5_1m_{year}{month}_grid_V_02.nc')
    ds_u = xr.open_dataset(f'{path}/sozotaux/{member}/sozotaux_ORAS5_1m_{year}{month}_grid_U_02.nc')
    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')

    ds_v = ds_v.rename({"y": "y_f", "x": "x_c"})
    ds_u = ds_u.rename({"y": "y_c", "x": "x_f"})

    ds = ds_v.copy()
    ds.coords['y_f'] = ds_v['y_f']
    ds.coords['x_c'] = ds_v['x_c']

    for var in ds_u.data_vars:
        ds[var] = ds_u[var]

    ds = ds.merge(
        mesh[
            [
                # metrics
                "e2u",
                "e1v",

            ]
        ]
    )

    ds['e2u'] = ds['e2u'].rename({"y": "y_c", "x": "x_f"})
    ds['e1v'] = ds['e1v'].rename({"y": "y_f", "x": "x_c"})

    metrics = {
        ('X',): ['e1v'], # X distances
        ('Y',): ['e2u'], # Y distances
    }

    grid = xgcm.Grid(ds, coords={'X':{'center': 'x_c', 'right': 'x_f'}, 'Y':{'center': 'y_c', 'right': 'y_f'}},
                    
                    metrics = metrics, periodic=False, autoparse_metadata=False)

    dtauydx = grid.derivative(ds['sometauy'], 'X', boundary='fill')
    dtauxdy = grid.derivative(ds['sozotaux'], 'Y', boundary='fill')
    curl = dtauydx - dtauxdy

    f_mask = mesh['fmaskutil']

    ocean = f_mask.values == 1

    curl_smooth = curl.copy()
    curl_smooth.values[ocean] = gaussian_filter(
        curl.values[ocean], 
        sigma=5
    )

    wsc_da = xr.DataArray(
        curl_smooth[:, :, :, 0],  
        coords={
        'time_counter': ds_v['time_counter'].values,
        'nav_lat': (('y', 'x'), ds_v['nav_lat'].values),
        'nav_lon': (('y', 'x'), ds_v['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='sowsc' 
    )
    output_path = f'/quobyte/maikesgrp/sanah/concepts/sowsc/{member}'
    os.makedirs(output_path,  exist_ok=True)
    wsc_da.to_netcdf(f'{output_path}/sowsc_{year}{month}_F.nc')

def ekman_pumping(year, month, member):
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'

    ds_pt = xr.open_dataset(
        f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc'
    )
    ds_sal = xr.open_dataset(
        f'{path}/vosaline/{member}/vosaline_ORAS5_1m_{year}{month}_grid_T_02.nc'
    )
    ds_wsc = xr.open_dataset(
        f'/quobyte/maikesgrp/sanah/concepts/sowsc/{member}/sowsc_{year}{month}_F.nc'
    )
    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')

    # mixed-layer interface indices
    ds_interface = xr.open_dataset(
        f'/quobyte/maikesgrp/sanah/concepts/mxl_interface/{member}/mxl_interface_{year}{month}.nc'
    )
    above_idx = ds_interface['mxl_interface'].values[0, :, :]
    below_idx = above_idx + 1

    # data
    depths = ds_pt['deptht'].values
    pt = ds_pt['votemper'].values[0, :, :, :]
    sal = ds_sal['vosaline'].values[0, :, :, :]
    wsc = ds_wsc['sowsc'].values

    # indices for vectorized selection
    lat_idx, lon_idx = np.ogrid[:above_idx.shape[0], :above_idx.shape[1]]

    # properties just below MLD base
    pt_below = pt[below_idx, lat_idx, lon_idx]
    sal_below = sal[below_idx, lat_idx, lon_idx]
    depth_below = depths[below_idx]

    # density at MLD base
    rho_below = rho(
        sal_below,
        pt_below,
        depth_below,
        ds_pt['nav_lat'].values
    )

    # coriolis parameter
    ff = mesh['ff'].values[0, :, :]

    # Ekman pumping velocity
    ekman = wsc / (rho_below * ff)
    
    ep_da = xr.DataArray(
        ekman,
        coords={
            'time_counter': ds_pt['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_pt['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_pt['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='voep'
    )

    output_path = f'/quobyte/maikesgrp/sanah/concepts/voep/{member}'
    os.makedirs(output_path, exist_ok=True)
    ep_da.to_netcdf(f'{output_path}/voep_{year}{month}_F.nc')


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

# def mxl_tendency(year, month, member):
#     base_path = f'/quobyte/maikesgrp/kkringel/oras5/ORCA025/somxl010/{member}'
#     output_dir = f'/quobyte/maikesgrp/sanah/concepts/{member}/mxl_tendency'
#     os.makedirs(output_dir, exist_ok=True)
    
#     month_str = f"{month:02d}"
#     year_str = str(year)
#     curr_file = f"{base_path}/somxl010_ORAS5_1m_{year_str}{month_str}_grid_T_02.nc"
    
#     if not os.path.exists(curr_file):
#         print(f"Missing current: {curr_file}")
#         return
    
#     ds_curr = xr.open_dataset(curr_file)
    
#     # Previous month
#     if month == 1:
#         prev_yr, prev_month = year - 1, 12
#     else:
#         prev_yr, prev_month = year, month - 1
#     prev_file = f"{base_path}/somxl010_ORAS5_1m_{prev_yr}{prev_month:02d}_grid_T_02.nc"
    
#     # Next month
#     if month == 12:
#         next_yr, next_month = year + 1, 1
#     else:
#         next_yr, next_month = year, month + 1
#     next_file = f"{base_path}/somxl010_ORAS5_1m_{next_yr}{next_month:02d}_grid_T_02.nc"
    
#     # Safely open neighbors
#     ds_prev = None
#     ds_next = None
    
#     try:
#         ds_prev = xr.open_dataset(prev_file)
#     except FileNotFoundError:
#         print(f"Missing prev: {prev_file}")
    
#     try:
#         ds_next = xr.open_dataset(next_file)
#     except FileNotFoundError:
#         print(f"Missing next: {next_file}")
    
#     # Compute tendency
#     if ds_prev is not None and ds_next is not None:
#         diff = ds_next['somxl010'].squeeze() - ds_prev['somxl010'].squeeze()
#         time_diff = (ds_next['time_counter'].values[0] - ds_prev['time_counter'].values[0]) / np.timedelta64(1, 'D')
#     elif ds_prev is not None:
#         diff = ds_curr['somxl010'].squeeze() - ds_prev['somxl010'].squeeze()
#         time_diff = (ds_curr['time_counter'].values[0] - ds_prev['time_counter'].values[0]) / np.timedelta64(1, 'D')
#     elif ds_next is not None:
#         diff = ds_next['somxl010'].squeeze() - ds_curr['somxl010'].squeeze()
#         time_diff = (ds_next['time_counter'].values[0] - ds_curr['time_counter'].values[0]) / np.timedelta64(1, 'D')
#     else:
#         print(f"No neighboring files found for {year}-{month_str}")
#         return
    
#     tendency = (diff / time_diff).rename('mxl_tend')
#     tendency = tendency.assign_coords(time_counter=ds_curr['time_counter'].values)
    
#     output_file = f"{output_dir}/mxl_tendency_{year_str}{month_str}.nc"
#     tendency.to_netcdf(output_file)

def mxl_tendency(year, month, member):

    base_path = f'/quobyte/maikesgrp/kkringel/oras5/ORCA025/somxl010/{member}'
    output_dir = f'/quobyte/maikesgrp/sanah/concepts/mxl_tendency/{member}/'
    os.makedirs(output_dir, exist_ok=True)
    
    curr_file = f"{base_path}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc"
    
    if not os.path.exists(curr_file):
        print(f"Missing current: {curr_file}")
        return
    
    ds_curr = xr.open_dataset(curr_file)
    
    # --- Convert to int for logic to find neighbors ---
    y_int = int(year)
    m_int = int(month)
    
    # Previous month logic
    if m_int == 1:
        prev_yr, prev_month = y_int - 1, 12
    else:
        prev_yr, prev_month = y_int, m_int - 1
    
    # Next month logic
    if m_int == 12:
        next_yr, next_month = y_int + 1, 1
    else:
        next_yr, next_month = y_int, m_int + 1
    
    # Format back to strings for file paths (ensuring 2-digit months)
    prev_file = f"{base_path}/somxl010_ORAS5_1m_{prev_yr}{prev_month:02d}_grid_T_02.nc"
    next_file = f"{base_path}/somxl010_ORAS5_1m_{next_yr}{next_month:02d}_grid_T_02.nc"
    
    ds_prev = xr.open_dataset(prev_file) if os.path.exists(prev_file) else None
    ds_next = xr.open_dataset(next_file) if os.path.exists(next_file) else None
    
    # --- Compute Tendency in m/s ---
    if ds_prev is not None and ds_next is not None:
        # Central difference (Next - Prev)
        diff = ds_next['somxl010'].squeeze() - ds_prev['somxl010'].squeeze()
        time_diff_seconds = (ds_next['time_counter'].values[0] - ds_prev['time_counter'].values[0]) / np.timedelta64(1, 's')
    elif ds_prev is not None:
        # Backward difference (Curr - Prev)
        diff = ds_curr['somxl010'].squeeze() - ds_prev['somxl010'].squeeze()
        time_diff_seconds = (ds_curr['time_counter'].values[0] - ds_prev['time_counter'].values[0]) / np.timedelta64(1, 's')
    elif ds_next is not None:
        # Forward difference (Next - Curr)
        diff = ds_next['somxl010'].squeeze() - ds_curr['somxl010'].squeeze()
        time_diff_seconds = (ds_next['time_counter'].values[0] - ds_curr['time_counter'].values[0]) / np.timedelta64(1, 's')
    else:
        return

    mxl_tend_ms = diff / time_diff_seconds

    # --- Create DataArray for saving ---
    tend_da = xr.DataArray(
        mxl_tend_ms.values[np.newaxis, :, :],
        coords={
            'time_counter': ds_curr['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_curr['nav_lat'].squeeze().values),
            'nav_lon': (('y', 'x'), ds_curr['nav_lon'].squeeze().values),
        },
        dims=['time_counter', 'y', 'x'],
        name='mxl_tend'
    )

    # Save using the original year/month strings
    output_file = f"{output_dir}/mxl_tendency_{year}{month}_T.nc"
    tend_da.to_netcdf(output_file)

# def sal_temp_conv(sal, temp, d, lon, lat):
#     p = sw.eos80.pres(d, lat)
#     SA = gsw.SA_from_SP(sal, p, lon, lat)
#     CT = gsw.CT_from_pt(SA, temp)
#     return SA, CT, p

def mlhc(year, month, member):
    rho=1026.0 
    cp=3990.0
    path = '/quobyte/maikesgrp/kkringel/oras5/ORCA025'
    ds_pt = xr.open_dataset(f'{path}/votemper/{member}/votemper_ORAS5_1m_{year}{month}_grid_T_02.nc')
    ds_mxl = xr.open_dataset(f'{path}/somxl010/{member}/somxl010_ORAS5_1m_{year}{month}_grid_T_02.nc')
    mesh = xr.open_dataset(f'{path}/mesh/mesh_mask.nc')
    ds = ds_pt.merge(
        mesh[
            [
                # metrics
                "e1t",
                "e2t",
                "e3t_0"
            ]
        ]
    )
    ds['e1t'] = ds['e1t'].rename({"t": "time_counter"})
    ds['e2t'] = ds['e2t'].rename({"t": "time_counter"})
    ds['e3t_0'] = ds['e3t_0'].rename({"z": "deptht", "t": "time_counter"})
    # 2. Simplified Masking Logic (The "Simple" Part)
    # Broadcast 1D deptht to the 3D shape of temperature
    temp = ds_pt.votemper 
    mld = ds_mxl.somxl010
    depth_broad, _ = xr.broadcast(ds.deptht, temp)
    
    # Create the mask based on the MLD (in meters)
    mask = depth_broad <= mld.fillna(0)

    # 3. Setup the xgcm Grid (The "Grid" Part)
    coords = {'X':{'center': 'x'}, 'Y':{'center': 'y'}, 'Z':{'center': 'deptht'}}
    metrics = {('X',): ['e1t'], ('Y',): ['e2t'], ('Z',): ['e3t_0']}
    grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=False, autoparse_metadata=False)

    # 4. Integrate using Grid Metrics
    pt_ml = temp.where(mask, 0)
    
    # Grid.integrate uses the 3D cell thicknesses (e3t_0) for perfect accuracy
    mlhc = rho * cp * grid.integrate(pt_ml, "Z")
    print('mlhc shape: ', mlhc.where(mld.notnull()).shape)
    mlhc_da = xr.DataArray(
        mlhc.where(mld.notnull()),
        coords={
            'time_counter': ds_pt['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_pt['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_pt['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='vomlhc'
    )
    output_path = f'/quobyte/maikesgrp/sanah/target/vomlhc/{member}'
    os.makedirs(output_path, exist_ok=True)
    mlhc_da.to_netcdf(f'{output_path}/vomlhc_{year}{month}_T.nc')



def plot_concept(year, month, concept_path, concept_name, plot_title, concept_label, log=False):
    ds = xr.open_dataset(concept_path)
    concept = ds[concept_name].values[0, :, :]
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

def plot_mxl(n, log=False):
    ds = xr.open_dataset('/quobyte/maikesgrp/kkringel/oras5/ORCA025/somxl010/opa0/somxl010_ORAS5_1m_197901_grid_T_02.nc')
    mesh = xr.open_dataset('/quobyte/maikesgrp/kkringel/oras5/ORCA025/mesh/mesh_mask.nc')
    
    concept = ds['somxl010'].squeeze()
    lats = ds['nav_lat']
    lons = ds['nav_lon']
    print('MAX:', np.nanmax(concept))
    print('MIN:', np.nanmin(concept))
    print('MEAN:', np.nanmean(concept))

    # Use full tmask at surface (time=0, depth=0)
    tmask = mesh['tmask'][0, 0, :, :].values

    # Check what this gives
    land_values = concept.values[tmask == 0]
    print("Land values with tmask:")
    print(f"  Min: {np.nanmin(land_values)}")
    print(f"  Max: {np.nanmax(land_values)}")
    print(f"  Count: {len(land_values)}")
    print(f"  NaN count: {np.isnan(land_values).sum()}")

    ocean_values = concept.values[tmask == 1]
    print("\nOcean values with tmask:")
    print(f"  Min: {np.nanmin(ocean_values)}")
    print(f"  Max: {np.nanmax(ocean_values)}")
    print(f"  Count: {len(ocean_values)}")
    
    # Explicitly set masked values to NaN
    concept_values = concept.values.copy()
    concept_values[tmask == 0] = np.nan
    concept_masked = xr.DataArray(concept_values, coords=concept.coords, dims=concept.dims)
    print("concept shape:", concept.shape)
    print("tmask shape:", tmask.shape)
    print("NaN count after masking:", np.isnan(concept_values).sum())
    print("Total points:", concept_values.size)
    # Before masking
    print("Before mask - sample values:", concept.values[0:10, 0:10])
    print("Mask values:", tmask[0:10, 0:10])

    # After masking
    concept_values[tmask == 0] = np.nan
    print("After mask - sample values:", concept_values[0:10, 0:10])
    fig, ax= plt.subplots(figsize=(12, 6))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    
    
    vmin = np.nanpercentile(concept_masked, 2)
    vmax = np.nanpercentile(concept_masked, 98)
    
    if log:
        concept_plot = ax.imshow(concept_masked.values,
                                     cmap='coolwarm', norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower')
    else:
        concept_plot = ax.imshow(concept_masked.values,
                                     cmap='coolwarm', vmin=vmin, vmax=vmax, origin='lower')


    plt.colorbar(concept_plot, ax=ax, orientation='vertical')
    plt.title('$S^2$')
    plt.savefig('mxl')

def entrainment(year, month, member):
    # 1. Open datasets
    ds_tend = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/mxl_tendency/{member}/mxl_tendency_{year}{month}_T.nc')
    ds_diff = xr.open_dataset(f'/quobyte/maikesgrp/sanah/concepts/votempdiff/{member}/votempdiff_{year}{month}_T.nc')
    
    rho = 1026.0
    cp = 3990.0
    
    # 2. Extract the specific variables for math
    # Use .values or ensure they are both DataArrays to avoid Dataset-alignment errors
    # Also: Apply the 'deepening only' logic (mxl_tend > 0)
    tendency_ms = ds_tend['mxl_tend']
    we = xr.where(tendency_ms > 0, tendency_ms, 0)
    
    # Calculate Heat Entrainment (W/m^2)
    he = rho * cp * we * ds_diff['votempdiff']
    
    # 3. Create DataArray
    he_da = xr.DataArray(
        he.values, # Use .values to ensure we are passing the data matrix
        coords={
            'time_counter': ds_diff['time_counter'].values,
            'nav_lat': (('y', 'x'), ds_diff['nav_lat'].values),
            'nav_lon': (('y', 'x'), ds_diff['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='vohfe'
    )
    
    # 4. Save
    output_path = f'/quobyte/maikesgrp/sanah/concepts/vohfe/{member}'
    os.makedirs(output_path, exist_ok=True)
    
    # Corrected variable name from mlhc_da to he_da
    he_da.to_netcdf(f'{output_path}/vohfe_{year}{month}_T.nc')

def run_task(args):
    yr, mn, mem = args
    try:
        entrainment(yr, mn, mem)
    except Exception as e:
        print(f"[ERROR] {yr}-{mn}-{mem}: {e}")

if __name__ == "__main__":
    # ---- Parse command-line arguments ----
    parser = argparse.ArgumentParser(description="Run MLD interface calculations in parallel.")
    parser.add_argument("--member", type=str, default=None,
                        help="Specific ensemble member to process (e.g., opa0)")
    parser.add_argument("--year", type=int, default=None,
                        help="Specific year to run")

    args = parser.parse_args()
    
    # Will iterate over year, member pairs -> 40*5 = 200 iterations

    # ---- Define parameter ranges ----
    years = [str(y) for y in range(1979, 2019)]
    # years = [1979]
    months = [f"{m:02d}" for m in range(1, 13)]
    members = ['opa0', 'opa1', 'opa2', 'opa3', 'opa4']

    if args.member:
        members = [args.member]
    if args.year:
        years = [args.year]

    tasks = list(product(years, months, members))
    print(f"Total tasks: {len(tasks)} for members: {members} and year: {years}")

    # ---- Set number of processes ----
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", min(len(tasks),mp.cpu_count())))
    print(f"Using {nproc} processes")

    # ---- Parallel execution ----
    with mp.Pool(nproc) as pool:
        pool.map(run_task, tasks)





    

    
    





    
    

