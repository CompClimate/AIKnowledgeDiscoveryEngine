import utils.load_data as load_data
from utils.get_config import config, try_cast
from datetime import datetime
from dateutil.relativedelta import relativedelta
import torch
import xarray as xr
import numpy as np

class ZScoreNormalize():
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = torch.from_numpy(np.nanmean(x, axis=(1, 2, 3, 4))).float()
        self.std = np.nanstd(x, axis=(1, 2, 3, 4))        
        self.std = torch.from_numpy(np.clip(self.std, a_min=1e-8, a_max=None)).float()

    def normalize(self, x):
        x = (x - self.mean[None, :, None, None, None]) / self.std[None, :, None, None, None]
        return torch.nan_to_num(x, nan=0.0)

    def denormalize(self, x):
        return x * self.std[None, :, None, None, None] + self.mean[None, :, None, None, None]
    
class MinMaxNormalize():
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x):
        self.min = torch.from_numpy(np.nanmin(x, axis=(1, 2, 3, 4))).float()
        self.max = torch.from_numpy(np.nanmax(x, axis=(1, 2, 3, 4))).float()

    def normalize(self, x):
        bottom = torch.clamp(self.max[None, :, None, None, None] - self.min[None, :, None, None, None], 1e-8)
        return (x - self.min[None, :, None, None, None]) / bottom

    def denormalize(self, x):
        return x * (self.max[None, :, None, None, None] - self.min[None, :, None, None, None]) + self.min[None, :, None, None, None]


        