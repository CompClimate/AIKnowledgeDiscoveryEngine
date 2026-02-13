# calls to load_data
# then deals with splitting data
# if we want to parallelize in the future that would happen here
print('in the file', flush=True)
from utils.load_data import EmulatorDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from utils.get_config import config, try_cast
from utils.compute_stats import Normalize
import torch
import time
import os

def get_dataset():
    start_time = time.time()
    dataset = EmulatorDataset()
    print('dataset initialized', flush=True)
    n = len(dataset)
    print(f'dataset length: {n}', flush=True)
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

    stats_loader = DataLoader(train_set, batch_size = 1, num_workers = 0, shuffle = False)
    input_norm = Normalize(len(try_cast(config['DATASET']['features']))) 
    concept_norm = Normalize(len(try_cast(config['DATASET']['concepts'])))
    output_norm = Normalize(len(try_cast(config['DATASET']['labels'])))  
    for i, (batch, concept_y, y) in enumerate(stats_loader):
        B, C, T, Y, X = batch.shape
        for c in range(C):               
            input_norm.update(c, batch)           
        B, F, T, Y, X = concept_y.shape
        for f in range(F):
            concept_norm.update(f, concept_y)
        B, F, T, Y, X= y.shape
        for f in range(F):
            output_norm.update(f, y)
    
    input_norm.finalize()
    concept_norm.finalize()
    output_norm.finalize()
    print('normalized', flush=True)

    batch_size =  config.getint('DATASET', 'batch_size') 
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)
    end_time = time.time()
    print(f'done in {end_time - start_time}', flush=True)

    return input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Run this once
    input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()

    # This gets the path to /home/sansuri/
    home_dir = os.path.expanduser("~") 
    save_path = os.path.join(home_dir, 'normalization_stats.pt')

    # Save the normalization objects
    torch.save({
        'input': input_norm,
        'concept': concept_norm,
        'output': output_norm
    }, save_path)
    
    print(f"Stats saved successfully to: {save_path}")