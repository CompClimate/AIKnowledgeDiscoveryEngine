# calls to load_data
# then deals with splitting data
# if we want to parallelize in the future that would happen here
from utils.load_data import EmulatorDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader

def get_dataset():
    dataset = EmulatorDataset()

    n = len(dataset)
    train_end = int(0.7 * n)
    val_end = train_end + int(0.15 * n)

    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n))

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size = 1, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = 1, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True)

    return train_loader, val_loader, test_loader
    