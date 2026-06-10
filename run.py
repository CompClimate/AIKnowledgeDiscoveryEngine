from utils.train import train, eval, make_output_dir
from utils.visualization import visualize, plot_sample, plot_sample_pred_only
from utils.get_data import get_dataset
from utils.get_config import config, try_cast
import torch
import time
from inference import save_val_preds, plot_pearsonr
import os

import traceback
_real_print = print
def print(*args, **kwargs):
    if any("cuda" in str(a).lower() or "cpu" in str(a).lower() for a in args):
        traceback.print_stack()
    _real_print(*args, **kwargs)

def run():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(DEVICE)
    start = time.time()
    input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    for i in range(config.getint('TRAINING', 'members')): 
        print('ensemble member: ', i+1)
        get_data_done = time.time()
        model, train_losses, val_losses, model_dir = train(input_norm, concept_norm, output_norm, train_loader, val_loader)
        train_done = time.time()
        print('training took ', train_done - get_data_done)
        plot_test = config.getboolean('OUTPUT', 'plot_test', fallback=False)
        if plot_test:
            test_results = eval(input_norm, concept_norm, output_norm, model, test_loader)
            from utils.visualization import find_output_dir
            output_dir = find_output_dir()
            losses_path = f'{output_dir}/detailed_losses.pt'
            detailed = torch.load(losses_path, weights_only=False)
            detailed.update(test_results)
            torch.save(detailed, losses_path)
            test_done = time.time()
            print('testing done ', test_done - train_done)
    visualize()
    save_val_preds(input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader)
    plot_sample(input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader)
    plot_pearsonr()
    plot_sample_pred_only(input_norm=input_norm, val_loader=val_loader)

if __name__ == '__main__':
    run()

