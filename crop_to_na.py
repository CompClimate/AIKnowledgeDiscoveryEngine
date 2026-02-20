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
from scipy.ndimage import gaussian_filter
import xgcm
import calendar
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='zarr')

def crop_concept_to_zarr(
    member: str,
    concept: str,
    cell: str,
    years=range(1979, 2019),
    lon_bounds=(-80, 20),
    lat_bounds=(20, 66),
):
    # Output directory
    out_dir = f"/quobyte/maikesgrp/sanah/na_crop_latest/{member}"
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, f"{concept}_na.zarr")

    first_file = True

    for year in years:
        for month_idx in range(1, 13):
            ym = f"{year}{month_idx:02d}"
            infile = (
                f"/quobyte/maikesgrp/sanah/concepts/{concept}/{member}/"
                f"{concept}_{ym}_{cell}.nc"
            )

            print(f"Processing {infile}", flush=True)

            try:
                ds = xr.open_dataset(infile, engine="netcdf4")
            except Exception as e:
                print(f"Failed to open {infile}: {e}")
                continue  # skip this file

            # Crop North Atlantic using 2D curvilinear coordinates
            mask = (
                (ds.nav_lon >= lon_bounds[0]) & (ds.nav_lon <= lon_bounds[1]) &
                (ds.nav_lat >= lat_bounds[0]) & (ds.nav_lat <= lat_bounds[1])
            )

            y_inds = mask.any(dim="x")
            x_inds = mask.any(dim="y")

            ds_na = ds.isel(y=y_inds, x=x_inds)

            # Chunk for ML
            ds_na = ds_na.chunk({"time_counter": 1})

            # Write to Zarr
            if first_file:
                ds_na.to_zarr(zarr_path, mode="w", consolidated=True)
                first_file = False
            else:
                ds_na.to_zarr(
                    zarr_path,
                    mode="a",
                    append_dim="time_counter",
                    consolidated=True,
                )

            ds.close()

    print(f"Finished processing {concept} for {member}. Saved to {zarr_path}")

def crop_input_to_zarr(
    member: str,
    concept: str,
    cell: str,
    years=range(1979, 2019),
    lon_bounds=(-80, 20),
    lat_bounds=(20, 66),
):
    # Output directory
    out_dir = f"/quobyte/maikesgrp/sanah/na_crop_latest/{member}"
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, f"{concept}_na.zarr")

    first_file = True

    for year in years:
        for month_idx in range(1, 13):
            ym = f"{year}{month_idx:02d}"
            infile = (
                f'/quobyte/maikesgrp/kkringel/oras5/ORCA025/{concept}/{member}/{concept}_ORAS5_1m_{ym}_grid_{cell}_02.nc'
            )

            print(f"Processing {infile}", flush=True)

            try:
                ds = xr.open_dataset(infile, engine="netcdf4")
            except Exception as e:
                print(f"Failed to open {infile}: {e}")
                continue  # skip this file

            # Crop North Atlantic using 2D curvilinear coordinates
            mask = ((ds.nav_lon >= lon_bounds[0]) & (ds.nav_lon <= lon_bounds[1]) & (ds.nav_lat >= lat_bounds[0]) & (ds.nav_lat <= lat_bounds[1]))

            y_inds = mask.any(dim="x")
            x_inds = mask.any(dim="y")

            ds_na = ds.isel(y=y_inds, x=x_inds)

            # Chunk for ML
            ds_na = ds_na.chunk({"time_counter": 1})

            # Write to Zarr
            if first_file:
                ds_na.to_zarr(zarr_path, mode="w", consolidated=True)
                first_file = False
            else:
                ds_na.to_zarr(
                    zarr_path,
                    mode="a",
                    append_dim="time_counter",
                    consolidated=True,
                )

            ds.close()

    print(f"Finished processing {concept} for {member}. Saved to {zarr_path}")



def crop_velocity_at_mld_to_zarr(
    member, vel_var, depth_coord, grid_cell,
    years=range(1979, 2019),
    lon_bounds=(-80, 20), lat_bounds=(20, 66),
):
    """Crop 3D velocity field at MLD depth to NA zarr.
    Selects the nearest depth level to MLD for each grid point,
    then crops to North Atlantic and saves as zarr.
    """
    out_dir = f"/quobyte/maikesgrp/sanah/na_crop_latest/{member}"
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, f"{vel_var}_mld_na.zarr")
    first_file = True

    for year in years:
        for month_idx in range(1, 13):
            ym = f"{year}{month_idx:02d}"
            vel_file = f'/quobyte/maikesgrp/kkringel/oras5/ORCA025/{vel_var}/{member}/{vel_var}_ORAS5_1m_{ym}_grid_{grid_cell}_02.nc'
            mld_file = f'/quobyte/maikesgrp/kkringel/oras5/ORCA025/somxl010/{member}/somxl010_ORAS5_1m_{ym}_grid_T_02.nc'

            print(f"Processing {vel_var} {ym} {member}", flush=True)

            try:
                ds_vel = xr.open_dataset(vel_file, engine="netcdf4")
                ds_mxl = xr.open_dataset(mld_file, engine="netcdf4")
            except Exception as e:
                print(f"Failed to open {ym}: {e}")
                continue

            # Select at MLD with numpy
            mld_vals = ds_mxl['somxl010'].values[0]  # (y, x)
            depths = ds_vel[depth_coord].values
            depth_idx = np.abs(depths[:, None, None] - mld_vals[None, :, :]).argmin(axis=0)
            lat_idx, lon_idx = np.ogrid[:mld_vals.shape[0], :mld_vals.shape[1]]
            vel_at_mld = ds_vel[vel_var].values[0, depth_idx, lat_idx, lon_idx]  # (y, x)

            # Wrap as dataset
            ds_out = xr.Dataset({
                vel_var: xr.DataArray(
                    vel_at_mld[np.newaxis],
                    dims=['time_counter', 'y', 'x'],
                    coords={
                        'time_counter': ds_vel['time_counter'].values,
                        'nav_lat': (('y', 'x'), ds_vel['nav_lat'].values),
                        'nav_lon': (('y', 'x'), ds_vel['nav_lon'].values),
                    },
                )
            })

            # Crop North Atlantic
            mask = (
                (ds_out.nav_lon >= lon_bounds[0]) & (ds_out.nav_lon <= lon_bounds[1]) &
                (ds_out.nav_lat >= lat_bounds[0]) & (ds_out.nav_lat <= lat_bounds[1])
            )
            ds_na = ds_out.isel(y=mask.any(dim="x"), x=mask.any(dim="y"))
            ds_na = ds_na.chunk({"time_counter": 1})

            if first_file:
                ds_na.to_zarr(zarr_path, mode="w", consolidated=True)
                first_file = False
            else:
                ds_na.to_zarr(zarr_path, mode="a", append_dim="time_counter", consolidated=True)

            ds_vel.close()
            ds_mxl.close()

    print(f"Finished {vel_var} at MLD for {member}. Saved to {zarr_path}")


if __name__ == "__main__":

    cell = 'T'
    member = sys.argv[1]
    concept = sys.argv[2]

    if concept in ['sowsc', 'voep', 'vovort']:
        cell = 'F'

    grids = {'sometauy': 'V', 'sozotaux': 'U', 'sosaline': 'T', 'sosstsst': 'T', 'sohefldo': 'T', 
            'somxl010': 'T', 'vomecrty_ml': 'V', 'vozocrtx_ml': 'U'}
    
    crop_concept_to_zarr(member, concept, grids[concept], years=range(1979, 2019))