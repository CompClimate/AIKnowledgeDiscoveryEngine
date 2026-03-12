# calls to load_data
# then deals with splitting data
# if we want to parallelize in the future that would happen here
print('in the file', flush=True)
from utils.load_data import EmulatorDataset
print('1', flush=True)
from torch.utils.data import Subset
print('1', flush=True)
from torch.utils.data import DataLoader
print('1', flush=True)
from utils.get_config import config, try_cast
print('1', flush=True)
from utils.compute_stats import ZScoreNormalize, MinMaxNormalize
print('1', flush=True)
import torch
print('1', flush=True)
import numpy as np
print('1', flush=True)
import matplotlib.pyplot as plt
print('1', flush=True)
import time
print('1', flush=True)
import os
print('imported everything', flush=True)

def plot_data_histograms(X_vals, c_vals, output_dir='figs'):
    """Plot histograms of all features, concepts, and climate modes after normalization."""
    #os.makedirs(output_dir, exist_ok=True)
    features      = try_cast(config['DATASET']['features'])
    concepts      = try_cast(config['DATASET']['concepts'])
    climate_modes = try_cast(config['DATASET.LAG']['climate_modes'])
    colors = {'feature': 'tab:blue', 'climate_mode': 'tab:orange', 'concept': 'tab:green'}

    all_vars = (
        [(v, 'feature', X_vals[i])      for i, v in enumerate(features)] +
        [(v, 'climate_mode' if v in climate_modes else 'concept', c_vals[i])
         for i, v in enumerate(concepts)]
    )

    for var, kind, arr_raw in all_vars:
        arr = arr_raw.flatten()
        arr = arr[~np.isnan(arr)]

        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        ax.hist(arr, bins=100, color=colors[kind], alpha=0.8, edgecolor='none')
        ax.set_title(f'{var}  [{kind.replace("_", " ")}]')
        ax.set_xlabel('Normalized value')
        ax.set_ylabel('Count')
        ax.annotate(f'n={len(arr):,}\nmin={arr.min():.3g}\nmax={arr.max():.3g}\nμ={arr.mean():.3g}\nσ={arr.std():.3g}',
                    xy=(0.98, 0.97), xycoords='axes fraction', fontsize=8,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))

        save_name = f'figs/hist_{var}.png'
        fig.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {save_name}')

def get_dataset():
    print('in get dataset', flush=True)
    start_time = time.time()
    dataset = EmulatorDataset()
    print('dataset initialized', flush=True)

    n = len(dataset)
    print(f'dataset length: {n}', flush=True)
    features = try_cast(config['DATASET']['features'])
    concepts = try_cast(config['DATASET']['concepts'])
    labels = try_cast(config['DATASET']['labels'])
    n_members = len(try_cast(config['DATASET']['members']))
    n_times = n // n_members

    # Split by time steps so no month appears in multiple sets
    train_time_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n_times)
    val_time_end = train_time_end + int(config.getfloat('MODEL.HYPERPARAMETERS', 'test_frac') * n_times)
    train_end = train_time_end * n_members
    val_end = val_time_end * n_members

    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n))

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    print('subsetting done', flush=True)

    norm_type = config.get('TRAINING', 'norm_type', fallback='MinMax')
    NormClass = ZScoreNormalize if norm_type == 'ZScore' else MinMaxNormalize
    input_norm = NormClass()
    concept_norm = NormClass()
    output_norm = NormClass()

    X_vars = []
    for feat in features:
        print(feat)
        var_slice = dataset.np_data[feat]
        print(var_slice.shape)
        X_vars.append(var_slice)
    X_vals = np.stack(X_vars)

    c_vars = []
    for concept in concepts:
        concept_slice = dataset.np_concepts[concept]
        c_vars.append(concept_slice)
    c_vals = np.stack(c_vars)

    l_vars = []
    for label in labels:
        label_slice = dataset.np_labels[label]
        l_vars.append(label_slice)
    l_vals = np.stack(l_vars)

    input_norm.fit(X_vals[:, :, :train_time_end])
    concept_norm.fit(c_vals[:, :, :train_time_end])
    output_norm.fit(l_vals[:, :, :train_time_end])

    #plot_data_histograms(X_vals, c_vals)

    batch_size =  config.getint('DATASET', 'batch_size') 
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    end_time = time.time()
    print(f'done in {end_time - start_time}', flush=True)

    return input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader

if __name__ == "__main__":
    get_dataset()