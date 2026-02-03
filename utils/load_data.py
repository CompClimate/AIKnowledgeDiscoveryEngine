from torch.utils.data import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xarray as xr
import numpy as np
import torch
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

        print('init')
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
            self.lazy_data[feat] = ds
        
        print('got input')
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
            self.lazy_concepts[concept] = ds

        print('got concepts')
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
            self.lazy_labels[label] = ds
        print('got label')

    def __len__(self):
        return len(self.date_range()) - self.window - max(self.offset) + 1
    
    def __getitem__(self, idx):
        member = idx % len(self.opas)
        time = idx // len(self.opas)
        data = self.get_input_window(member, time)
        label = self.get_label(member, time)
        concept = self.get_concepts(member, time)
        return data, concept, label
    
    def date_range(self):
        start = datetime.strptime(self.start, "%Y%m")
        end = datetime.strptime(self.end, "%Y%m")

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
        X_vals = np.concat(X_vars)
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