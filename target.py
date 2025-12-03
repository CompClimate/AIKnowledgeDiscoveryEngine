import xarray as xr
import numpy as np
import seawater as sw
import gsw
import os
from concepts import plot_concept

def rho(sal, temp, d, lat):
    p = sw.eos80.pres(d, lat)
    return sw.eos80.dens(sal, temp, p)

def cons_temp(sal, temp, d, lat, lon):
    p = sw.eos80.pres(d, lat)
    SA = gsw.SA_from_SP(sal, p, lat, lon)
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
    pt_above = pt[ld, lat_idx, lon_idx]
    sal_above = sal[ld, lat_idx, lon_idx]
    
    # Density at MLD base
    rho_below = rho(sal_below, pt_below, depths[ud], ds_mxl['nav_lat'].values)
    rho_above = rho(sal_above, pt_above, depths[ld], ds_mxl['nav_lat'].values)
    
    # Conservative temperature at MLD base
    ct_below = cons_temp(sal_below, pt_below, depths[ud], ds_mxl['nav_lon'].values, ds_mxl['nav_lat'].values)
    ct_above = cons_temp(sal_above, pt_above, depths[ld], ds_mxl['nav_lon'].values, ds_mxl['nav_lat'].values)
    #print(np.isnan(ct[:, -1]).all(), np.isnan(ct[:, 0]).all())

    # Depth brackets
    h_below = depths[ud]
    h_above = depths[ld]
    
    # Specific heat
    #cp = gsw.gsw_cp0
    cp = 3991.86795711963
    
    # Integral
    rho_interp = (rho_below + rho_above) / 2
    ct_interp = (ct_below + ct_above) / 2
    mlhc = rho_interp * cp * ct_interp * (h_below - h_above)
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

if __name__ == "__main__":
    mlhc('1979', '06', 'opa0')
    plot_concept('1979', '06', '/quobyte/maikesgrp/sanah/target/opa0/vomlhc/vomlhc_197906.nc',
    'mixed_layer_heat_content', 'MLHC', '$J/m^{-2}$')
