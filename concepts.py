import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm 
import seawater as sw
import scipy as sp 

def vertical_shear(year, month):

    # replace with appropriate paths
    ds_mv = xr.open_dataset(f'vomecrty_control_monthly_highres_3D_{year}{month}_CONS_v0.1.nc')
    ds_zv = xr.open_dataset(f'vozocrtx_control_monthly_highres_3D_{year}{month}_CONS_v0.1.nc')
    ds_mxl = xr.open_dataset(f'somxl010_control_monthly_highres_2D_{year}{month}_CONS_v0.1.nc')

    # get data
    mld = ds_mxl['somxl010'].values[0, :, :]  # [lat, lon]
    depths = ds_zv['depthu'].values  # [depth]
    zv = ds_zv['vozocrtx'].values[0, :, :, :]  # [depth, lat, lon]
    mv = ds_mv['vomecrty'].values[0, :, :, :] # [depth, lat, lon]

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


    shear_sq_da.to_netcdf(f'vograds2_{year}{month}.nc')

def heat_flux(year, month):
    # lat = y (1021), lon = x (1442)
    # replace with appropriate paths
    ds_pt = xr.open_dataset(f'votemper_control_monthly_highres_3D_{year}{month}_CONS_v0.1.nc')
    ds_mxl = xr.open_dataset(f'somxl010_control_monthly_highres_2D_{year}{month}_CONS_v0.1.nc')

    # get data
    mld = ds_mxl['somxl010'].values[0, :, :]  # [lat, lon]
    depths = ds_pt['deptht'].values  # [depth]
    pt = ds_pt['votemper'].values[0, :, :, :]  # [depth, lat, lon]

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

    print('Entrainment of heat flux:', heat_entrainment[~np.isnan(heat_entrainment)])

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

    heat_flux_da.to_netcdf(f'votempdiff_{year}{month}.nc')

def rho(sal, temp, d, lat):
    p = sw.eos80.pres(d, lat)
    return sw.eos80.dens(sal, temp, p)

def brunt_vaisala(year, month):
    # lat = y (1021), lon = x (1442)
    # replace with appropriate paths
    ds_pt = xr.open_dataset(f'votemper_control_monthly_highres_3D_{year}{month}1_CONS_v0.1.nc')
    ds_sal = xr.open_dataset(f'vosaline_control_monthly_highres_3D_{year}{month}_CONS_v0.1.nc')
    ds_mxl = xr.open_dataset(f'somxl010_control_monthly_highres_2D_{year}{month}_CONS_v0.1.nc')
    # get data
    mld = ds_mxl['somxl010'].values[0, :, :]  # [lat, lon]
    depths = ds_pt['deptht'].values  # [depth]
    pt = ds_pt['votemper'].values[0, :, :, :]  # [depth, lat, lon]
    sal = ds_sal['vosaline'].values[0, :, :, :] # [depth, lat, lon]

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
    print(bv)
    print('BV:', bv[~np.isnan(bv)])

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

    bv_da.to_netcdf(f'vobvfreq_{year}{month}.nc')
    
def richardson_number(year, month):
    # load brunt-vaisala frequency and vertical shear at mld base
    bv = xr.open_dataset(f'vobvfreq_{year}{month}.nc')['brunt-vaisala_frequency'].squeeze()
    s2 = xr.open_dataset(f'vograds2_{year}{month}.nc')['vertical_shear_squared'].squeeze()
    
    # Set minimum shear threshold to avoid division by near-zero values
    s2_min = 1e-8
    
    # Calculate Ri, masking where shear is below threshold
    ri = np.where(s2 > s2_min, bv**2 / s2, np.nan)
    
    ri_da = xr.DataArray(
        ri,  
        coords={
            'time_counter': bv['time_counter'].values,
            'nav_lat': (('y', 'x'), bv['nav_lat'].values),
            'nav_lon': (('y', 'x'), bv['nav_lon'].values),
        },
        dims=['time_counter', 'y', 'x'],
        name='richardson_number' 
    )
    ri_da.to_netcdf(f'vorino_{year}{month}.nc')

def wind_stress_curl(year, month):
    # lat = y (1021), lon = x (1442)
    ds_m = xr.open_dataset(f'sometauy_control_monthly_highres_2D_{year}{month}_CONS_v0.1.nc')
    ds_z = xr.open_dataset(f'sozotaux_control_monthly_highres_2D_{year}{month}_CONS_v0.1.nc')
    mesh = xr.open_dataset('mesh/mesh_mask.nc')
    zws = ds_z['sozotaux'].values[0, :, :]  # [depth, lat, lon]
    mws = ds_m['sometauy'].values[0, :, :] # [depth, lat, lon]
    vmask = mesh['vmask'].values[0, 0, :, :]
    umask = mesh['umask'].values[0, 0, :, :]
    mws_masked = np.ma.masked_where(vmask == 0, mws)
    zws_masked = np.ma.masked_where(umask == 0, zws)
    zws_smooth = sp.signal.savgol_filter(zws_masked, window_length=15, polyorder=3)
    mws_smooth = sp.signal.savgol_filter(mws_masked, window_length=15, polyorder=3)
    e1v = mesh['e1v'].values[0, :, :]  # zonal spacing at V points
    e2u = mesh['e2u'].values[0, :, :]  # meridional spacing at U points
    # meridional grad wrt lon 
    dty_dx = np.gradient(mws_smooth, axis=1) / e1v  # d(tau_y)/dx
    # zonal grad wrt lat 
    dtx_dy = np.gradient(zws_smooth, axis=0) / e2u  # d(tau_x)/dy
    
    # wind stress curl
    wsc = dty_dx - dtx_dy

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

    wsc_da.to_netcdf(f'sowsc_{year}{month}_smooth.nc') 

def ekman_pumping(year, month):
    ds_pt = xr.open_dataset(f'votemper_control_monthly_highres_3D_{year}{month}_CONS_v0.1.nc')
    ds_sal = xr.open_dataset(f'vosaline_control_monthly_highres_3D_{year}{month}_CONS_v0.1.nc')
    ds_mxl = xr.open_dataset(f'somxl010_control_monthly_highres_2D_{year}{month}_CONS_v0.1.nc')
    ds_wsc = xr.open_dataset(f'sowsc_{year}{month}.nc')
    mesh = xr.open_dataset('mesh/mesh_mask.nc')
    
    # data
    mld = ds_mxl['somxl010'].values[0, :, :]  # [lat, lon]
    depths = ds_pt['deptht'].values  # [depth]
    pt = ds_pt['votemper'].values[0, :, :, :]  # [depth, lat, lon]
    sal = ds_sal['vosaline'].values[0, :, :, :]  # [depth, lat, lon]
    wsc = ds_wsc['wind_stress_curl'].values
    
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
    ep_da.to_netcdf(f'voep_{year}{month}.nc')

def rossby(year, month):
    mesh = xr.open_dataset('mesh/mesh_mask.nc')
    ff = mesh['ff'].values[0, :, :] # [y, x]
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
    beta_da.to_netcdf(f'sorp_{year}{month}.nc')

def plot_concept(concept_path, concept_name, plot_title, concept_label, log=False):
    ds = xr.open_dataset(concept_path)
    concept = ds[concept_name].squeeze()
    lats = ds['nav_lat']
    lons = ds['nav_lon']

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    concept_masked = np.ma.masked_invalid(concept)
    
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
    plt.show()

year = '1979'
month = '01'

#wind_stress_curl(year, month)



# wsc = ds['wind_stress_curl'].values
# wsc_smooth = sp.signal.savgol_filter(wsc, window_length=19, polyorder=3)
# wsc_da = xr.DataArray(
#     wsc_smooth,
#     coords={
#         'nav_lat': (('y', 'x'), ds['nav_lat'].values),
#         'nav_lon': (('y', 'x'), ds['nav_lon'].values),
#     },
#     dims=['y', 'x'],
#     name='wind_stress_curl'
# )
# wsc_da.to_netcdf('sowsc_197901_smoother.nc')

#plot_concept('vobvfreq_197901.nc', 'brunt-vaisala_frequency', '', '', log=False)

vertical_shear(year, month)
ds_sal = xr.open_dataset('vograds2_197901.nc')
print(ds_sal)