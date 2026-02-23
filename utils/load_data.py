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
        self.features_to_clip = try_cast(config['DATASET']['features_to_clip'])
        self.concepts_to_clip = try_cast(config['DATASET']['concepts_to_clip'])
        self.opas = try_cast(config['DATASET']['members'])
        self.loc = config['DATASET']['location']
        self.window = config.getint('DATASET', 'context_window')
        self.offset = try_cast(config['DATASET']['offset'])
        self.start = config['DATASET']['start']
        self.end = config['DATASET']['end']
        self.file_details = try_cast(config['DATASET.FILEDETAILS']['inputs'])
        self.concept_details = try_cast(config['DATASET.FILEDETAILS']['concepts'])

        self.np_data = {}
        for feat in self.features:
            data = []
            for opa in self.opas:
                dp = xr.open_zarr(f"{self.loc}/{opa}/{feat}_na.zarr")
                dp = dp.expand_dims(opa=[opa])
                dp = dp.sel(y=slice(0, 302), x=slice(0,400))
                dp = dp.sortby('time_counter')
                dp = dp.sel(time_counter=slice(self.start, self.end))
                data.append(dp)
            ds = xr.concat(data, dim="opa")
            ds = ds.assign_coords(time=np.arange(ds.sizes["time_counter"]))
            self.np_data[feat] = ds.to_array().values.squeeze(0)

        self.np_concepts = {}
        for concept in self.concepts:
            data = []
            for opa in self.opas:
                dp = xr.open_zarr(f"{self.loc}/{opa}/{concept}_na.zarr")
                dp = dp.expand_dims(opa=[opa])
                dp = dp.sel(y=slice(0, 302), x=slice(0,400))
                dp = dp.sortby('time_counter')
                dp = dp.sel(time_counter=slice(self.start, self.end))
                data.append(dp)
            ds = xr.concat(data, dim="opa")
            ds = ds.assign_coords(time=np.arange(ds.sizes["time_counter"]))
            self.np_concepts[concept] = ds.to_array().values.squeeze(0)

        self.np_labels = {}
        for label in self.labels:
            data = []
            for opa in self.opas:
                dp = xr.open_zarr(f"{self.loc}/{opa}/{label}_na.zarr")
                dp = dp.expand_dims(opa=[opa])
                dp = dp.sel(y=slice(0, 302), x=slice(0,400))
                dp = dp.sel(time_counter=slice(self.start, self.end))
                data.append(dp)
            ds = xr.concat(data, dim="opa")
            ds = ds.assign_coords(time=np.arange(ds.sizes["time_counter"]))
            self.np_labels[label] = ds.to_array().values.squeeze(0)

        self.preprocessing()

    def preprocessing(self):
        log_features = {'somxl010'}
        log_concepts = {'vori', 'von2', 'vos2'}  # always >= 0 (negatives → NaN, clipped away)
        symlog_concepts = {'vohfe'}               # can be negative — use symlog
        smooth_features = try_cast(config['DATASET']['smooth_features'])
        smooth_concepts = try_cast(config['DATASET']['smooth_concepts'])
        sigma = config.getfloat('DATASET', 'smooth_sigma')

        for feat in self.features_to_clip:
            if feat not in self.np_data:
                continue
            arr = self.np_data[feat]
            if feat in log_features:
                arr = np.log10(np.where(arr > 0, arr, np.nan))
            p2 = np.nanpercentile(arr, 2)
            p98 = np.nanpercentile(arr, 98)
            arr = np.clip(arr, p2, p98)
            print(f'  {feat}: {"log10 + " if feat in log_features else ""}clipped to [{p2:.3g}, {p98:.3g}]')
            if feat in smooth_features:
                arr = self._smooth(arr, sigma)
                print(f'  {feat}: smoothed sigma={sigma}')
            self.np_data[feat] = arr
        print('preprocessed inputs')

        for concept in self.concepts_to_clip:
            if concept not in self.np_concepts:
                continue
            arr = self.np_concepts[concept]
            if concept in log_concepts:
                arr = np.log10(np.where(arr > 0, arr, np.nan))
                p2 = np.nanpercentile(arr, 2)
                p98 = np.nanpercentile(arr, 98)
                arr = np.clip(arr, p2, p98)
                print(f'  {concept}: log10 + clipped to [{p2:.3g}, {p98:.3g}]')
            elif concept in symlog_concepts:
                arr = np.sign(arr) * np.log10(1 + np.abs(arr))
                p2 = np.nanpercentile(arr, 2)
                p98 = np.nanpercentile(arr, 98)
                arr = np.clip(arr, p2, p98)
                print(f'  {concept}: symlog + clipped to [{p2:.3g}, {p98:.3g}]')
            else:
                p2 = np.nanpercentile(arr, 2)
                p98 = np.nanpercentile(arr, 98)
                arr = np.clip(arr, p2, p98)
                print(f'  {concept}: clipped to [{p2:.3g}, {p98:.3g}]')
            if concept in smooth_concepts:
                arr = self._smooth(arr, sigma)
                print(f'  {concept}: smoothed sigma={sigma}')
            self.np_concepts[concept] = arr
        print('preprocessed concepts')

    def _smooth(self, arr, sigma):
        """Apply gaussian smoothing over spatial dims (y, x) per member per timestep."""
        out = arr.copy()
        for m in range(arr.shape[0]):
            for t in range(arr.shape[1]):
                nan_mask = np.isnan(out[m, t])
                out[m, t] = np.where(nan_mask, 0.0, out[m, t])
                out[m, t] = gaussian_filter(out[m, t], sigma=sigma)
                out[m, t][nan_mask] = np.nan
        return out

    def __len__(self):
        return (len(self.date_range()) - self.window - max(self.offset) + 1) * len(self.opas)

    def __getitem__(self, idx):
        member = idx % len(self.opas)
        time = idx // len(self.opas)
        data = self.get_input_window(member, time)
        label = self.get_label(member, time)
        concept = self.get_concepts(member, time)
        return data, concept, label
    
    def date_range(self):
        start = datetime.strptime(self.start, "%Y-%m")
        end = datetime.strptime(self.end, "%Y-%m")

        date_list = []
        cur_date = start
        while cur_date <= end:
            date_list.append(cur_date.strftime("%Y%m"))
            cur_date += relativedelta(months=1)
        return date_list
    
    def get_input_window(self, member, time):
        X_vars = []
        feature_idx = slice(time, time+self.window)
        for feat in self.features:
            var_slice = self.np_data[feat][member][feature_idx]
            X_vars.append(var_slice)
        X_vals = np.stack(X_vars)
        return torch.from_numpy(X_vals).float()

    def get_concepts(self, member, time):
        climate_modes = try_cast(config['DATASET.LAG']['climate_modes'])
        lag = config.getint('DATASET.LAG', 'lag')
        c_vars = []
        concept_idx = [time+self.window-1+lead for lead in self.offset]
        for concept in self.concepts:
            if concept in climate_modes:
                concept_idx = [time-lag for time in concept_idx]
            concept_slice = self.np_concepts[concept][member][concept_idx]
            c_vars.append(concept_slice)
        c_vals = np.stack(c_vars)
        return torch.from_numpy(c_vals).float()
    
    def get_label(self, member, time):
        l_vars = []
        label_idx = [time+self.window-1+lead for lead in self.offset]
        for label in self.labels:
            label_slice = self.np_labels[label][member][label_idx]
            l_vars.append(label_slice)
        l_vals = np.stack(l_vars)
        return torch.from_numpy(l_vals).float()