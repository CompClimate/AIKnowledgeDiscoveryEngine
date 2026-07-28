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
from scipy.ndimage import gaussian_filter
#import xgcm
import calendar
import pandas as pd
import sys

def crop_concept_to_zarr(
    member: str,
    concept: str,
    cell: str,
    years=range(1980, 2019),
    lon_bounds=(-145, -110),
    lat_bounds=(20, 55),
):
    # Output directory
    out_dir = f"/path/to/data/nep_crop/{member}"
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, f"{concept}_na.zarr")

    first_file = True

    for year in years:
        for month_idx in range(1, 13):
            ym = f"{year}{month_idx:02d}"
            infile = (
                f"/path/to/data/concepts/{concept}/{member}/"
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
    out_dir = f"/path/to/data/pacific_crop/{member}"
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, f"{concept}_p.zarr")

    first_file = True
    
    for year in years:
        for month_idx in range(1, 13):
            ym = f"{year}{month_idx:02d}"
            infile = (
                f'/path/to/data/oras5/{concept}/{member}/{concept}_ORAS5_1m_{ym}_grid_{cell}_02.nc'
            )

            print(f"Processing {infile}", flush=True)

            try:
                ds = xr.open_dataset(infile, engine="netcdf4")
            except Exception as e:
                print(f"Failed to open {infile}: {e}")
                continue  # skip this file

            # Crop North Atlantic using 2D curvilinear coordinates
            mask = ((ds.nav_lat >= -60) & (ds.nav_lat <= 60) & ((ds.nav_lon >= 120) | (ds.nav_lon <= -75)))

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

def crop_and_detrend(
      member: str,
      concept: str,
      cell: str,
      years=range(1979, 2019),
      lon_bounds=(-145, -110),
      lat_bounds=(20, 55),
      input_dir=None,
      concept_dir=None,
  ):
      # variable name inside the netCDF may differ from concept name
      VAR_REMAP = {'mxl_tendency': 'mxl_tend'}
      var_key = VAR_REMAP.get(concept, concept)

      out_dir = f"/path/to/data/nep_detrended/{member}"
      os.makedirs(out_dir, exist_ok=True)
      zarr_path = os.path.join(out_dir, f"{concept}_p.zarr")

      # Step 1: crop and append all timesteps
      first_file = True
      for year in years:
          for month_idx in range(1, 13):
              ym = f"{year}{month_idx:02d}"
              if input_dir:
                  infile = f"{input_dir}/{concept}/{member}/{concept}_ORAS5_1m_{ym}_grid_{cell}_02.nc"
              else:
                  infile = f"{concept_dir}/{concept}/{member}/{concept}_{ym}_{cell}.nc"

              print(f"Processing {infile}", flush=True)
              try:
                  ds = xr.open_dataset(infile, engine="netcdf4")
              except Exception as e:
                  print(f"Failed to open {infile}: {e}")
                  continue

              mask = (
                  (ds.nav_lon >= lon_bounds[0]) & (ds.nav_lon <= lon_bounds[1]) &
                  (ds.nav_lat >= lat_bounds[0]) & (ds.nav_lat <= lat_bounds[1])
              )
              y_inds = mask.any(dim="x")
              x_inds = mask.any(dim="y")
              ds_na = ds.isel(y=y_inds, x=x_inds).chunk({"time_counter": 1})

              # standardize variable name to concept
              if var_key != concept:
                  ds_na = ds_na.rename({var_key: concept})

              if first_file:
                  ds_na.to_zarr(zarr_path, mode="w", consolidated=True)
                  first_file = False
              else:
                  ds_na.to_zarr(zarr_path, mode="a", append_dim="time_counter", consolidated=True)
              ds.close()

      # Step 2: detrend the full time series
      print(f"Detrending {concept} for {member}...", flush=True)
      ds_full = xr.open_zarr(zarr_path)
      arr = ds_full[concept].values  # (time, y, x)
      nan_mask = np.isnan(arr)
      arr[nan_mask] = 0.0
      arr_detrend = sp.signal.detrend(arr, axis=0)
      arr_detrend[nan_mask] = np.nan

      ds_out = xr.Dataset(
          {concept: (ds_full[concept].dims, arr_detrend)},
          coords=ds_full[concept].coords
      )
      ds_out.to_zarr(zarr_path, mode='w', consolidated=True)
      print(f"Done {concept} {member}", flush=True)

def global_detrend(
    member: str,
    concept: str,
    cell: str,
    years=range(1979, 2019),
    input_dir=None,
    concept_dir=None,
    out_base='/path/to/data/global_detrended',
):
    VAR_REMAP = {'mxl_tendency': 'mxl_tend'}
    var_key = VAR_REMAP.get(concept, concept)

    out_dir = f"{out_base}/{member}"
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, f"{concept}_g.zarr")

    # Step 1: append all timesteps with no spatial crop
    first_file = True
    for year in years:
        for month_idx in range(1, 13):
            ym = f"{year}{month_idx:02d}"
            if input_dir:
                infile = f"{input_dir}/{concept}/{member}/{concept}_ORAS5_1m_{ym}_grid_{cell}_02.nc"
            else:
                infile = f"{concept_dir}/{concept}/{member}/{concept}_{ym}_{cell}.nc"

            print(f"Processing {infile}", flush=True)
            try:
                ds = xr.open_dataset(infile)
            except Exception as e:
                print(f"Failed to open {infile}: {e}")
                continue

            ds = ds.chunk({"time_counter": 1})

            if var_key != concept:
                ds = ds.rename({var_key: concept})

            if first_file:
                ds.to_zarr(zarr_path, mode="w", consolidated=True)
                first_file = False
            else:
                ds.to_zarr(zarr_path, mode="a", append_dim="time_counter", consolidated=True)
            ds.close()

    # Step 2: detrend the full time series
    print(f"Detrending {concept} for {member}...", flush=True)
    ds_full = xr.open_zarr(zarr_path)
    arr = ds_full[concept].values  # (time, y, x)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 0.0
    arr_detrend = sp.signal.detrend(arr, axis=0)
    arr_detrend[nan_mask] = np.nan

    ds_out = xr.Dataset(
        {concept: (ds_full[concept].dims, arr_detrend)},
        coords=ds_full[concept].coords
    )
    ds_out.to_zarr(zarr_path, mode='w', consolidated=True)
    print(f"Done {concept} {member}", flush=True)


def global_tmask(
    mesh_path='/path/to/data/oras5/mesh/mesh_mask.nc',
    out_base='/path/to/data/global_detrended',
):
    ds = xr.open_dataset(mesh_path)
    ds[['tmaskutil', 'nav_lat', 'nav_lon']].to_zarr(
        os.path.join(out_base, 'tmask_g.zarr'), mode='w', consolidated=True
    )
    ds.close()
    print(f'Saved tmask_g.zarr to {out_base}', flush=True)


if __name__ == "__main__":
    grids = {'sometauy': 'V', 'sozotaux': 'U', 'sosaline': 'T', 'sosstsst': 'T', 'sohefldo': 'T',
             'somxl010': 'T', 'sossheig': 'T', 'vomecrty_ml': 'V', 'vozocrtx_ml': 'U', 'vos2': 'T', 'von2': 'T', 'vohfe': 'T',
             'vosaldiff': 'T', 'votempdiff': 'T', 'sowsc': 'F', 'mxl_tendency': 'T'}

    INPUT_DIR   = '/path/to/data/oras5'
    CONCEPT_DIR = '/path/to/data/concepts'
    MEMBERS     = ['opa1']

    # --- global ocean: tmask ---
    global_tmask()

    # --- global ocean: features ---
    features = ['somxl010', 'sossheig', 'sosaline', 'sosstsst']
    # features = ['sohefldo']
    # for member in MEMBERS:
    #     for feat in features:
    #         global_detrend(member, feat, grids[feat], input_dir=INPUT_DIR)

    # --- global ocean: concepts ---
    # concepts = ['von2', 'vos2', 'vohfe', 'mxl_tendency', 'sowsc', 'vozocrtx_ml', 'vomecrty_ml', 'vosaldiff', 'votempdiff']
    # for member in MEMBERS:
    #     for concept in concepts:
    #         global_detrend(member, concept, grids[concept], concept_dir=CONCEPT_DIR)

    # --- global ocean: label ---
    for member in MEMBERS:
        global_detrend(member, 'vomlhc', 'T', concept_dir=CONCEPT_DIR)

    # --- NEP (previous runs) ---
    # for member in MEMBERS:
    #     crop_and_detrend(member, 'vomlhc', 'T',
    #                 concept_dir=CONCEPT_DIR)
    