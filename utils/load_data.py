# to load in from nc files general preprocessing, may be combined with get_data later if this is a simple file
# load data may be what calls concepts?
from torch.utils.data import Dataset

class EmulatorDataset(Dataset):
    