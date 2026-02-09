#from utils.get_config import get_config
from utils.train import train, eval
from utils.visualization import visualize
from utils.get_data import get_dataset, get_dataset_preload
import time

def run():
    test = False
    start = time.time()
    input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset_preload()
    get_data_done = time.time()
    print('get data took ', get_data_done - start)
    model, train_losses, val_losses = train(input_norm, concept_norm, output_norm, train_loader, val_loader)
    train_done = time.time()
    print('training took ', train_done - get_data_done)
    test_losses = None
    if test:
        test_losses = eval(input_norm, concept_norm, output_norm, model, test_loader)
        test_done = time.time()
        print('testing done ', test_done - train_done)
    visualize(train_losses, val_losses, test_losses)

if __name__ == '__main__':
    run()