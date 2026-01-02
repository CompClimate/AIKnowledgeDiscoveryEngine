from torch.utils.data import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xarray as xr
import numpy as np
import torch

class EmulatorDataset(Dataset):
    def __init__(self):
        self.features = ['iicethic', 'iicevelu', 'sohefldo']
        self.concepts = ['voep', 'von2']
        self.labels = ['sowsc']
        self.opas = ['opa0', 'opa1', 'opa2', 'opa3', 'opa4']
        self.window = 6
        self.offset = [1, 3, 6]
        self.start = '197901'
        self.end = '198501'
        self.file_details = {'iicethic': {'type': 'icemod', 'where': ''}, 'iicevelu': {'type': 'icemod', 'where': ''}, 'iicevelv': {'type': 'icemod', 'where': ''}, 'ileadfra': {'type': 'icemod', 'where': ''}, 'so20chgt': {'type': 'grid', 'where': '_T'}, 'sohefldo': {'type': 'grid', 'where': '_T'}, 'sohtc300': {'type': 'grid', 'where': '_T'}, 'sohtc700': {'type': 'grid', 'where': '_T'}, 'sohtcbtm': {'type': 'grid', 'where': '_T'}, 'sometauy': {'type': 'grid', 'where': '_V'}, 'somxl010': {'type': 'grid', 'where': '_T'}, 'sosaline': {'type': 'grid', 'where': '_T'}, 'sossheig': {'type': 'grid', 'where': '_T'}, 'sosstsst': {'type': 'grid', 'where': '_T'}, 'sowaflup': {'type': 'grid', 'where': '_T'}, 'sozotaux': {'type': 'grid', 'where': '_U'}, 'vomecrty': {'type': 'grid', 'where': '_V'}, 'vosaline': {'type': 'grid', 'where': '_T'}, 'votemper': {'type': 'grid', 'where': '_T'}, 'vozocrtx': {'type': 'grid', 'where': '_U'}}
        self.concept_details = {'von2': 'T', 'voep': 'F', 'sowsc': 'F'}

        self.dates = self.date_range()
        self.lazy_data = {}
        for feat in self.features:
            details = self.file_details[feat]
            files = [f"/quobyte/maikesgrp/kkringel/oras5/ORCA025/{feat}/opa0/{feat}_ORAS5_1m_{date}_{details['type']}{details['where']}_02.nc" for date in self.dates]
            ds = xr.open_mfdataset(files, combine='by_coords') 
            ds = ds.rename({'time_counter': 'time'})
            ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))
            self.lazy_data[feat] = ds
        
        self.lazy_concepts = {}
        for concept in self.concepts:
            files = [f"/quobyte/maikesgrp/sanah/concepts/{concept}/opa0/{concept}_{date}_{self.concept_details[concept]}.nc" for date in self.dates]
            ds = xr.open_mfdataset(files, combine='by_coords') 
            ds = ds.rename({'time_counter': 'time'})
            ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))
            self.lazy_concepts[concept] = ds

        self.lazy_labels = {}
        for label in self.labels:
            files = [f"/quobyte/maikesgrp/sanah/concepts/{label}/opa0/{label}_{date}_{self.concept_details[label]}.nc" for date in self.dates]
            ds = xr.open_mfdataset(files, combine='by_coords') 
            ds = ds.rename({'time_counter': 'time'})
            ds = ds.assign_coords(time=np.arange(ds.sizes["time"]))
            self.lazy_labels[label] = ds
    

    def __len__(self):
        return len(self.date_range()) - self.window - max(self.offset) + 1
    
    def __getitem__(self, idx):
        data = self.get_input_window(idx)
        label = self.get_label(idx)
        concept = self.get_concepts(idx)

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
    def get_input_window(self, idx):
        X_vars = []
        for feat in self.features:
            var_slice = self.lazy_data[feat].isel(time=slice(idx, idx+self.window)).to_array(dim='variable')
            X_vars.append(var_slice.values)
        X_vals = np.concat(X_vars)
        return torch.from_numpy(X_vals).float()

    def get_concepts(self, idx):
        c_vars = []
        concept_idx = [idx+self.window-1+lead for lead in self.offset]
        for concept in self.concepts:
            concept_slice = self.lazy_concepts[concept].isel(time=concept_idx).to_array(dim='variable')
            c_vars.append(concept_slice)
        c_vals = xr.concat(c_vars, dim='variable')
        c_vals = c_vals.transpose("time", "y", "x", "variable")
        return torch.from_numpy(c_vals.values).float()
    
    def get_label(self, idx):
        l_vars = []
        label_idx = [idx+self.window-1+lead for lead in self.offset]
        for label in self.labels:
            ds = self.lazy_labels[label].isel(time=label_idx).to_array(dim='variable')
            l_vars.append(ds)
        l_vals = xr.concat(l_vars, dim="variable")
        l_vals = l_vals.transpose("time", "y", "x", "variable")
        return torch.from_numpy(l_vals.values).float()