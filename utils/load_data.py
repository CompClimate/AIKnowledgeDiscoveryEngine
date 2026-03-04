from torch.utils.data import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xarray as xr
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from utils.get_config import try_cast, parse_section, config

xr.set_options(use_new_combine_kwarg_defaults=True)

class EmulatorDataset(Dataset):
    def __init__(self):
        self.features = try_cast(config['DATASET']['features'])
        self.concepts = try_cast(config['DATASET']['concepts'])
        self.labels = try_cast(config['DATASET']['labels'])
        self.opas = try_cast(config['DATASET']['members'])
        self.loc = config['DATASET']['location']
        self.window = config.getint('DATASET', 'context_window')
        self.offset = try_cast(config['DATASET']['offset'])
        self.start = config['DATASET']['start']
        self.end = config['DATASET']['end']
        self.file_details = try_cast(config['DATASET.FILEDETAILS']['inputs'])
        self.concept_details = try_cast(config['DATASET.FILEDETAILS']['concepts'])

        self.lazy_data = {}
        for feat in self.features:
            data = []
            for opa in self.opas:
                dp = xr.open_zarr(f"{self.loc}/{opa}/{feat}_na.zarr")
                dp = dp.expand_dims(opa=[opa])
                data.append(dp)
            ds = xr.concat(data, dim="opa")
            ds = ds.rename({'time_counter': 'time'})
            ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))
            self.lazy_data[feat] = ds.load()
            print(f'  loaded {feat}', flush=True)

        self.lazy_concepts = {}
        for concept in self.concepts:
            data = []
            for opa in self.opas:
                dp = xr.open_zarr(f"{self.loc}/{opa}/{concept}_na.zarr")
                dp = dp.expand_dims(opa=[opa])
                data.append(dp)
            ds = xr.concat(data, dim="opa")
            ds = ds.rename({'time_counter': 'time'})
            ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))
            self.lazy_concepts[concept] = ds.load()
            print(f'  loaded {concept}', flush=True)

        self.lazy_labels = {}
        for label in self.labels:
            data = []
            for opa in self.opas:
                dp = xr.open_zarr(f"{self.loc}/{opa}/{label}_na.zarr")
                dp = dp.expand_dims(opa=[opa])
                data.append(dp)
            ds = xr.concat(data, dim="opa")
            ds = ds.rename({'time_counter': 'time'})
            ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))
            self.lazy_labels[label] = ds.load()
            print(f'  loaded {label}', flush=True)
        self.smooth_concepts = try_cast(config.get('DATASET', 'smooth_concepts', fallback='[]'))
        self.smooth_inputs = try_cast(config.get('DATASET', 'smooth_inputs', fallback='[]'))
        self.smooth_sigma = config.getfloat('DATASET', 'smooth_sigma', fallback=2)
        self._materialized = False

    def _smooth(self, arr):
        """Apply Gaussian smoothing to each 2D spatial slice."""
        for m in range(arr.shape[0]):
            for t in range(arr.shape[1]):
                arr[m, t] = gaussian_filter(np.nan_to_num(arr[m, t], nan=0.0), sigma=self.smooth_sigma)
        return arr

    def _clip(self, arr, name):
        """Clip to [2, 98] percentile."""
        p2 = np.nanpercentile(arr, 2)
        p98 = np.nanpercentile(arr, 98)
        arr = np.clip(arr, p2, p98)
        print(f'  {name}: clipped to [{p2:.3g}, {p98:.3g}]')
        return arr

    def materialize(self):
        """Extract numpy arrays from xarray for fast __getitem__."""
        self._input_arrays = {}
        for feat in self.features:
            ds = self.lazy_data[feat]
            arr = ds.sel(y=slice(0, 302), x=slice(0, 400)).to_array().values
            arr = arr.squeeze(0)  # (n_members, n_timesteps, 302, 400)
            if feat in self.smooth_inputs:
                arr = self._smooth(arr)
                print(f'  {feat}: smoothed with sigma={self.smooth_sigma}')
            arr = self._clip(arr, feat)
            self._input_arrays[feat] = arr
        print('materialized inputs')

        self._concept_arrays = {}
        for concept in self.concepts:
            ds = self.lazy_concepts[concept]
            arr = ds.sel(y=slice(0, 302), x=slice(0, 400)).to_array().values
            arr = arr.squeeze(0)
            if concept == 'vori':
                arr = np.where(arr > 0, np.log10(arr), np.nan)
                print(f'  vori: log10 transform, range=[{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]')
            arr = self._smooth(arr)
            print(f'  {concept}: smoothed with sigma={self.smooth_sigma}')
            arr = self._clip(arr, concept)
            self._concept_arrays[concept] = arr
        print('materialized concepts')

        self._label_arrays = {}
        for label in self.labels:
            ds = self.lazy_labels[label]
            arr = ds.sel(y=slice(0, 302), x=slice(0, 400)).to_array().values
            self._label_arrays[label] = arr.squeeze(0)
        print('materialized labels')
        # Free xarray data to avoid holding double the memory
        self.lazy_data = {}
        self.lazy_concepts = {}
        self.lazy_labels = {}
        self._materialized = True

    def __len__(self):
        return (len(self.date_range()) - self.window - max(self.offset) + 1) * len(self.opas)

    def __getitem__(self, idx):
        member = idx % len(self.opas)
        time = idx // len(self.opas)
        if self._materialized:
            return self._getitem_fast(member, time)
        data = self.get_input_window(member, time)
        label = self.get_label(member, time)
        concept = self.get_concepts(member, time)
        return data, concept, label

    def _getitem_fast(self, member, time):
        X = np.stack([self._input_arrays[f][member, time:time+self.window]
                      for f in self.features])
        cidx = [time + self.window - 1 + lead for lead in self.offset]
        C = np.stack([self._concept_arrays[c][member, cidx]
                      for c in self.concepts])
        L = np.stack([self._label_arrays[l][member, cidx]
                      for l in self.labels])
        return torch.from_numpy(X).float(), torch.from_numpy(C).float(), torch.from_numpy(L).float()
    
    def date_range(self):
        fmt = "%Y-%m" if '-' in self.start else "%Y%m"
        start = datetime.strptime(self.start, fmt)
        end = datetime.strptime(self.end, fmt)

        date_list = []
        cur_date = start
        while cur_date <= end:
            date_list.append(cur_date.strftime("%Y%m"))
            cur_date += relativedelta(months=1)
        return date_list
    
    # using .values makes this not lazy TODO Fix
    def get_input_window(self, member, time):
        X_vars = []
        for feat in self.features:
            var_slice = self.lazy_data[feat].isel(time=slice(time, time+self.window), opa=member).to_array(dim='variable')
            var_slice = var_slice.sel(y=slice(0, 302),x=slice(0, 400))
            X_vars.append(var_slice.values)
        X_vals = np.concatenate(X_vars)
        return torch.from_numpy(X_vals).float()

    def get_concepts(self, member, time):
        c_vars = []
        concept_idx = [time+self.window-1+lead for lead in self.offset]
        for concept in self.concepts:
            concept_slice = self.lazy_concepts[concept].isel(time=concept_idx, opa=member).to_array(dim='variable')
            concept_slice = concept_slice.sel(y=slice(0, 302), x=slice(0,400))
            c_vars.append(concept_slice)
        c_vals = xr.concat(c_vars, dim='variable')
        c_vals = c_vals.transpose("variable", "time", "y", "x")
        return torch.from_numpy(c_vals.values).float()
    
    def get_label(self, member, time):
        l_vars = []
        label_idx = [time+self.window-1+lead for lead in self.offset]
        for label in self.labels:
            ds = self.lazy_labels[label].isel(time=label_idx, opa=member).to_array(dim='variable')
            ds = ds.sel(y=slice(0, 302), x=slice(0,400))
            l_vars.append(ds)
        l_vals = xr.concat(l_vars, dim="variable")
        l_vals = l_vals.transpose("variable", "time", "y", "x")
        return torch.from_numpy(l_vals.values).float()