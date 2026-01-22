import utils.load_data as load_data
from utils.get_config import config, try_cast
from datetime import datetime
from dateutil.relativedelta import relativedelta
import torch
import xarray as xr

class Normalize():
    def __init__(self, features):
        self.count = torch.zeros(features)
        self.mean = torch.zeros(features)
        self.M2 = torch.zeros(features)

    def update(self, idx, x):
        batch = x[:, idx, :, :, :]
        valid = ~torch.isnan(batch)
        batch = batch[valid]
        batch_mean = batch.mean()
        batch_count = batch.numel()

        delta = batch_mean - self.mean[idx]
        total = self.count[idx] + batch_count
        self.mean[idx] += delta * batch_count / total
        
        variance = ((batch - batch_mean) ** 2).sum()
        self.M2[idx] += variance + delta**2 * self.count[idx] * batch_count / total
        
        self.count[idx] = total
    
    def finalize(self):
        var = self.M2 / torch.clamp(self.count-1, min=1)
        self.std = torch.sqrt(var)

    def normalize(self, x):
        return (x - self.mean[None, :, None, None, None]) / self.std[None, :, None, None, None]

    def denormalize(self, x):
        return x * self.std[None, :, None, None, None] + self.mean[None, :, None, None, None]

        