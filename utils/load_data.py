from torch.utils.data import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
import xarray as xr
import torch

class EmulatorDataset(Dataset):
    def __init__(self):
        self.features = ['iicethic', 'iicevelu']
        self.concepts = ['sowsc']
        self.label = 'sowsc'
        self.opas = ['opa0', 'opa1', 'opa2', 'opa3', 'opa4']
        self.window = 6
        self.offset = [1, 3, 6]
        self.start = '197901'
        self.end = '198001'
        self.file_details = {'iicethic': {'type': 'icemod', 'where': ''}, 'iicevelu': {'type': 'icemod', 'where': ''}, 'iicevelv': {'type': 'icemod', 'where': ''}, 'ileadfra': {'type': 'icemod', 'where': ''}, 'so20chgt': {'type': 'grid', 'where': '_T'}, 'sohefldo': {'type': 'grid', 'where': '_T'}, 'sohtc300': {'type': 'grid', 'where': '_T'}, 'sohtc700': {'type': 'grid', 'where': '_T'}, 'sohtcbtm': {'type': 'grid', 'where': '_T'}, 'sometauy': {'type': 'grid', 'where': '_V'}, 'somxl010': {'type': 'grid', 'where': '_T'}, 'sosaline': {'type': 'grid', 'where': '_T'}, 'sossheig': {'type': 'grid', 'where': '_T'}, 'sosstsst': {'type': 'grid', 'where': '_T'}, 'sowaflup': {'type': 'grid', 'where': '_T'}, 'sozotaux': {'type': 'grid', 'where': '_U'}, 'vomecrty': {'type': 'grid', 'where': '_V'}, 'vosaline': {'type': 'grid', 'where': '_T'}, 'votemper': {'type': 'grid', 'where': '_T'}, 'vozocrtx': {'type': 'grid', 'where': '_U'}}

    def __len__(self):
        return len(self.date_range()) - self.window - max(self.offset) + 1
    
    def __getitem__(self, idx):
        data = self.get_input_window(idx)
        label = self.get_label(idx)
        concept = self.get_concepts(idx)
        # X = torch.tensor(data, dtype=torch.float32)
        # y = torch.tensor(label, dtype=torch.float32)

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
    
    def get_input_window(self, idx):
        X_vars = []
        window_dates = self.date_range()[idx:idx+self.window]
        for feat in self.features:
            details = self.file_details[feat]
            files = [f"/quobyte/maikesgrp/kkringel/oras5/ORCA025/{feat}/opa0/{feat}_ORAS5_1m_{date}_{details['type']}{details['where']}_02.nc" for date in window_dates]
            ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', join='outer')    #creates indexed time dimension whihc is better for ML?
            X_vars.append(ds.isel(time_counter = 0, drop=True))
        return xr.concat(X_vars, dim = 'feature')
    
    def get_concepts(self, idx):
        c_vars = []
        concept_dates = [self.date_range()[idx+self.window-1+lead] for lead in self.offset]
        for concept in self.concepts:
            concept_dates = [self.date_range()[idx+self.window-1+lead] for lead in self.offset]
            files = [f"/quobyte/maikesgrp/sanah/concepts/{concept}/opa0/{concept}_{date}_F.nc" for date in concept_dates]
            ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', join='outer')    #creates indexed time dimension whihc is better for ML?
            c_vars.append(ds.isel(time_counter = 0, drop=True))
        return xr.concat(c_vars, dim = 'feature')
    
    def get_label(self, idx):
        label_dates = [self.date_range()[idx+self.window-1+lead] for lead in self.offset]
        files = [f"/quobyte/maikesgrp/sanah/concepts/{self.label}/opa0/{self.label}_{date}_F.nc" for date in label_dates]
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', join='outer')
        ds = ds.isel(time_counter=0, drop=True)
        return ds
    