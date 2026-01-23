# calls to load_data
# then deals with splitting data
# if we want to parallelize in the future that would happen here
from utils.load_data import EmulatorDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from utils.get_config import config, try_cast
from utils.compute_stats import Normalize

def get_dataset():
    dataset = EmulatorDataset()

    n = len(dataset)
    print(n)
    members = len(dataset.opas)
    train_end = round(int(config.getfloat('MODEL.HYPERPARAMETERS', 'train_frac') * n)/members)*members
    val_end = train_end + round(int(config.getfloat('MODEL.HYPERPARAMETERS', 'test_frac') * n)/members)*members

    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n))

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    stats_loader = DataLoader(train_set, batch_size = 1, num_workers = 0, shuffle = False)
    input_norm = Normalize(len(try_cast(config['DATASET']['features']))) 
    concept_norm = Normalize(len(try_cast(config['DATASET']['concepts'])))
    output_norm = Normalize(len(try_cast(config['DATASET']['labels'])))    
    for batch, concept_y, y in stats_loader:    
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

    train_loader = DataLoader(train_set, batch_size = 5, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = 5, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = 5, shuffle = True)

    return input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader
    