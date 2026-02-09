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


# 1. Import get_config
import utils.get_config as get_config

# 2. Define path and FORCE the read immediately
home_dir = os.path.expanduser("~")
config_path = os.path.join(home_dir, 'AIKnowledgeDiscoveryEngine/utils/config.ini')

if os.path.exists(config_path):
    get_config.config.read(config_path)
    print(f"Config loaded successfully from {config_path}", flush=True)
else:
    print(f"ERROR: Config not found at {config_path}", flush=True)

# 3. NOW import the dataset (it will now see the populated config)
from utils.load_data import EmulatorDataset
from torch.utils.data import Subset, DataLoader

print('imports done', flush=True)

def get_dataset():
    start_time = time.time()
    dataset = EmulatorDataset()
    print('dataset initialized', flush=True)
    n = len(dataset)
    train_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n)
    val_end = train_end + int(config.getfloat('MODEL.HYPERPARAMETERS', 'test_frac') * n)

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
        print('in the loop', flush=True)
        B, C, T, Y, X = batch.shape
        for c in range(C):               
            input_norm.update(c, batch)           
        B, F, T, Y, X = concept_y.shape
        for f in range(F):
            concept_norm.update(f, concept_y)
        B, F, T, Y, X= y.shape
        for f in range(F):
            output_norm.update(f, y)
    
    print('out of loop', flush=True)
    input_norm.finalize()
    concept_norm.finalize()
    output_norm.finalize()

    batch_size =  config.getint('DATASET', 'batch_size') 
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)
    end_time = time.time()
    print(f'done in {end_time - start_time}', flush=True)

    return input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader

def get_dataset_preload():
    from torch.utils.data import TensorDataset
    start_time = time.time()
    dataset = EmulatorDataset()
    print('dataset initialized', flush=True)
    n = len(dataset)

    # Preload all data into memory
    all_inputs, all_concepts, all_labels = [], [], []
    for i in range(n):
        x, c, y = dataset[i]
        all_inputs.append(x)
        all_concepts.append(c)
        all_labels.append(y)
        if i % 50 == 0:
            print(f'preloading {i}/{n}', flush=True)
    all_inputs = torch.stack(all_inputs)
    all_concepts = torch.stack(all_concepts)
    all_labels = torch.stack(all_labels)
    print(f'preloaded all data: inputs={all_inputs.shape}, concepts={all_concepts.shape}, labels={all_labels.shape}', flush=True)

    # Split
    train_end = int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n)
    val_end = train_end + int(config.getfloat('MODEL.HYPERPARAMETERS', 'test_frac') * n)

    train_set = TensorDataset(all_inputs[:train_end], all_concepts[:train_end], all_labels[:train_end])
    val_set = TensorDataset(all_inputs[train_end:val_end], all_concepts[train_end:val_end], all_labels[train_end:val_end])
    test_set = TensorDataset(all_inputs[val_end:], all_concepts[val_end:], all_labels[val_end:])

    # Compute normalization stats from training set
    input_norm = Normalize(len(try_cast(config['DATASET']['features'])))
    concept_norm = Normalize(len(try_cast(config['DATASET']['concepts'])))
    output_norm = Normalize(len(try_cast(config['DATASET']['labels'])))
    for i in range(len(train_set)):
        batch, concept_y, y = train_set[i]
        batch = batch.unsqueeze(0)
        concept_y = concept_y.unsqueeze(0)
        y = y.unsqueeze(0)
        for c in range(batch.shape[1]):
            input_norm.update(c, batch)
        for f in range(concept_y.shape[1]):
            concept_norm.update(f, concept_y)
        for f in range(y.shape[1]):
            output_norm.update(f, y)
    input_norm.finalize()
    concept_norm.finalize()
    output_norm.finalize()
    print('normalization stats computed', flush=True)

    batch_size = config.getint('DATASET', 'batch_size')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
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