from utils.train import train, eval, make_output_dir
from utils.visualization import visualize, plot_sample
from utils.get_data import get_dataset
from utils.get_config import config, try_cast
import torch
import time
from inference import save_val_preds, save_all_preds
import os
import numpy as np

def run():
    nc_path = '/path/to/data/oras5/somxl010/opa0/somxl010_ORAS5_1m_199812_grid_T_02.nc'
    input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    from torch.utils.data import DataLoader 
    full_loader = DataLoader(train_loader.dataset.dataset, batch_size=8, shuffle=False, num_workers=4)  
    for i in range(1): 
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        np.random.seed(i)
        print('ensemble member: ', i+1)
        start = time.time()
        get_data_done = time.time()
        print('get data took ', get_data_done - start)

        # n_free_concepts = 1, concept_lambda = 0.5 and adaptive
        config.set('MODEL.HYPERPARAMETERS', 'n_free_concepts', '1')
        config.set('TRAINING', 'concept_lambda', '0.5')
        output_dir=f'/path/to/data/global_cbm/UNetCBM_adaptive_ep101_lr0.001_bs64_L1Loss_ZScore_v{i+1}'
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/config.ini', 'w') as f:
            config.write(f)
        model, train_losses, val_losses, model_dir = train(input_norm, concept_norm, output_norm, train_loader, val_loader, 
        output_dir=output_dir)
        train_done = time.time()
        print('training took ', train_done - get_data_done)
        visualize(output_dir)
        save_all_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm,                
                        output_norm=output_norm, output_dir=model_dir, full_loader=full_loader)
        model_acc([model_dir], nc_path=nc_path, domain_lat=(-90, 90), domain_lon=(-180, 180), save_dir=model_dir)
        # save_val_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm, 
        #                 output_norm=output_norm, val_loader=val_loader, test_loader=test_loader, output_dir=model_dir) 

        # # n_free_concepts = 0, concept_lambda = 0.5 and adaptive
        # config.set('MODEL.HYPERPARAMETERS', 'n_free_concepts', '0')
        # output_dir=f'/path/to/data/paper_cbm/concepts_4/free0/UNetCBM_adaptive_ep101_lr0.001_bs64_L1Loss_ZScore_v{i+1}'
        # os.makedirs(output_dir, exist_ok=True)
        # with open(f'{output_dir}/config.ini', 'w') as f:
        #     config.write(f)
        # model, train_losses, val_losses, model_dir = train(input_norm, concept_norm, output_norm, train_loader, val_loader, 
        # output_dir=output_dir)
        # train_done = time.time()
        # print('training took ', train_done - get_data_done)
        # visualize(output_dir)
        # save_all_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm,                
        #                 output_norm=output_norm, output_dir=model_dir, full_loader=full_loader)
        # save_val_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm, 
        #                output_norm=output_norm, val_loader=val_loader, test_loader=test_loader, output_dir=model_dir)  

        # # concept_lambda = 0, n_free_concepts = 1
        # config.set('MODEL.HYPERPARAMETERS', 'n_free_concepts', '1')
        # config.set('TRAINING', 'concept_lambda', '0.0')
        # output_dir=f'/path/to/data/paper_cbm/concepts_4/unsup/UNetCBM_lam0_ep101_lr0.001_bs64_L1Loss_ZScore_v{i+1}'
        # os.makedirs(output_dir, exist_ok=True)
        # with open(f'{output_dir}/config.ini', 'w') as f:
        #     config.write(f)
        # model, train_losses, val_losses, model_dir = train(input_norm, concept_norm, output_norm, train_loader, val_loader, 
        # output_dir=output_dir)
        # train_done = time.time()
        # print('training took ', train_done - get_data_done)
        # visualize(output_dir)
        # save_all_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm,                
        #                 output_norm=output_norm, output_dir=model_dir, full_loader=full_loader)
        # save_val_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm, 
        #                output_norm=output_norm, val_loader=val_loader, test_loader=test_loader, output_dir=model_dir) 

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
    #model_dir = '/path/to/data/runs/UNetCBM_lam0.85_ep76_lr0.001_bs64_L1Loss_ZScore_v2'
    #input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    #visualize(model_dir)
    #save_val_preds(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader)
    #plot_sample(model_dir=model_dir, input_norm=input_norm, concept_norm=concept_norm, output_norm=output_norm, val_loader=val_loader)
    #plot_pearsonr(model_dir)

