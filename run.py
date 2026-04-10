from utils.train import train, eval, make_output_dir
from utils.visualization import visualize, plot_sample
from utils.get_data import get_dataset
from utils.get_config import config, try_cast
import torch
import time
from inference import save_val_preds, plot_pearsonr
import os

def run():
    input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    for i in range(6): 
        print('ensemble member: ', i+1)
        start = time.time()
        get_data_done = time.time()
        print('get data took ', get_data_done - start)

        # concept_names = try_cast(config['DATASET']['concepts'])
        # for i, concept in enumerate(concept_names):
        #     print(f'\n=== Training with concept {concept} ({i+1}/{len(concept_names)}) ===', flush=True)
        output_dir=f'/quobyte/maikesgrp/mlhc_cbm/detrended/UNetCBM_lam0.5_ep101_lr0.001_bs64_L1Loss_ZScore_v{i+7}'
        os.makedirs(output_dir, exist_ok=True)
        model, train_losses, val_losses, model_dir = train(input_norm, concept_norm, output_norm, train_loader, val_loader, 
        output_dir=output_dir)
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
        #visualize()
        #save_val_preds(input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader)
        #plot_sample(input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader)
        #plot_pearsonr()
        #plot_sample_pred_only(input_norm=input_norm, val_loader=val_loader)


if __name__ == '__main__':
    run()
    #model_dir = '/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.85_ep76_lr0.001_bs64_L1Loss_ZScore_v2'
    #input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    #visualize(model_dir)
    #save_val_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader)
    #plot_sample(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader)
    #plot_pearsonr(model_dir)

