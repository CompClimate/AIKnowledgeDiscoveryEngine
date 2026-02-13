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
import pandas as pd
import seaborn as sns
import sys

def crop_concept_to_zarr(
    member: str,
    concept: str,
    cell: str,
    years=range(1980, 2019),
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
                f"/quobyte/maikesgrp/sanah/target/{concept}/{member}/"
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



if __name__ == "__main__":

    cell = 'T'
    member = sys.argv[1]
    concept = sys.argv[2]

    if concept in ['sowsc', 'voep']:
        cell = 'F'

    grids = {'sometauy': 'V', 'sozotaux': 'U', 'sosaline': 'T', 'sosstsst': 'T', 'sohefldo': 'T', 
            'somxl010': 'T'}
    
    crop_input_to_zarr(member, concept, grids[concept], years=range(1979, 2019))