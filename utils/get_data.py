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
        print(concept)
        concept_slice = dataset.np_concepts[concept]
        print(concept_slice.shape)
        c_vars.append(concept_slice)
    c_vals = np.stack(c_vars)

    l_vars = []
    for label in labels:
        print(label)
        label_slice = dataset.np_labels[label]
        print(label_slice.shape)
        l_vars.append(label_slice)
    l_vals = np.stack(l_vars)

    norm_cache = '/quobyte/maikesgrp/sanah/norm_stats.npz'
    features_key = '_'.join(sorted(features)) + f'_w{config.getint("DATASET", "context_window")}'
    concepts_key = '_'.join(sorted(concepts)) + f'_w{config.getint("DATASET", "context_window")}'
    if os.path.exists(norm_cache):
        cache = np.load(norm_cache, allow_pickle=True)
        if cache['features_key'].item() == features_key and cache['concepts_key'].item() == concepts_key:
            print('loading norm stats from cache', flush=True)
            input_norm.mean   = torch.from_numpy(cache['input_mean']).float()
            input_norm.std    = torch.from_numpy(cache['input_std']).float()
            concept_norm.mean = torch.from_numpy(cache['concept_mean']).float()
            concept_norm.std  = torch.from_numpy(cache['concept_std']).float()
            output_norm.mean  = torch.from_numpy(cache['output_mean']).float()
            output_norm.std   = torch.from_numpy(cache['output_std']).float()
        else:
            print('norm cache mismatch, refitting norms', flush=True)
            input_norm.fit(X_vals[:, :, :train_time_end])
            concept_norm.fit(c_vals[:, :, :train_time_end])
            output_norm.fit(l_vals[:, :, :train_time_end])
            np.savez(norm_cache,
                     input_mean=input_norm.mean.numpy(), input_std=input_norm.std.numpy(),
                     concept_mean=concept_norm.mean.numpy(), concept_std=concept_norm.std.numpy(),
                     output_mean=output_norm.mean.numpy(), output_std=output_norm.std.numpy(),
                     features_key=features_key, concepts_key=concepts_key)
    else:
        print('no norm cache, fitting and saving', flush=True)
        input_norm.fit(X_vals[:, :, :train_time_end])
        print('fit input')
        concept_norm.fit(c_vals[:, :, :train_time_end])
        print('fit concept')
        output_norm.fit(l_vals[:, :, :train_time_end])
        print('fit label')
        np.savez(norm_cache,
                 input_mean=input_norm.mean.numpy(), input_std=input_norm.std.numpy(),
                 concept_mean=concept_norm.mean.numpy(), concept_std=concept_norm.std.numpy(),
                 output_mean=output_norm.mean.numpy(), output_std=output_norm.std.numpy(),
                 features_key=features_key, concepts_key=concepts_key)
        print('norm stats saved', flush=True)

    batch_size =  config.getint('DATASET', 'batch_size') 
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    end_time = time.time()
    print(f'done in {end_time - start_time}', flush=True)

    return input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader

if __name__ == "__main__":
    get_dataset()