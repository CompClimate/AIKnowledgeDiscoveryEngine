# trains models
# returns training loss and val loss

from model import UNetCBM
from utils.get_data import get_dataset
#from utils.compute_stats import compute_mean_std, FeatureStandardize
import torch
import importlib
from utils.get_config import parse_section, config
import utils.get_config as get_config
import xarray as xr

mesh = xr.open_dataset('/quobyte/maikesgrp/kkringel/oras5/ORCA025/mesh/mesh_mask.nc')
mask = mesh['tmaskutil'].isel(t=0).values  # [y, x]
mask = torch.tensor(mask, dtype=torch.float32)[None, None, :, :, None]
mask = mask.permute(0, 1, 4, 2, 3) 

def train(input_norm, concept_norm, output_norm, train_loader, val_loader):
    # Training loop
    n_epochs = config.getint('TRAINING', 'epochs')
    concept_loss_fn = config['TRAINING']['concept_loss_fn']
    concept_loss_fn = getattr(torch.nn, concept_loss_fn)()
    out_loss_fn = config['TRAINING']['out_loss_fn']
    out_loss_fn = getattr(torch.nn, out_loss_fn)()
    concept_lambda = config.getfloat('TRAINING', 'concept_lambda')

    model_type = model_type = config['MODEL']['type']
    model = get_config.get_model()
    optimizer = get_config.get_optimizer(model)
    scheduler = get_config.get_scheduler(optimizer)
    scheduler_name = config['SCHEDULER']['type']   

    #allow for restart?
    start_epoch = 0

    losses = []
    val_losses = []
    for epoch in range(start_epoch, n_epochs):
        model.train(True)
        train_loss_accum = None
        n_snaps = 0
        for batch, concept_y, y in train_loader:
            batch = torch.nan_to_num(input_norm.normalize(batch), nan=0.0)
            concept_y = torch.nan_to_num(concept_norm.normalize(concept_y), nan=0.0)
            y = torch.nan_to_num(output_norm.normalize(y), nan=0.0)
            
            pred, concept_pred = model(batch)
            pred = pred*mask
            concept_pred = concept_pred*mask

            loss = out_loss_fn(pred, y) + (concept_lambda * concept_loss_fn(concept_pred, concept_y))
            n_snaps += 1
            if train_loss_accum:
              train_loss_accum = train_loss_accum.add(loss)
            else:
              train_loss_accum = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_mean = train_loss_accum / n_snaps
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            n_snaps = 0
            for val_batch, val_concept_y, val_y in val_loader:
                val_batch = torch.nan_to_num(input_norm.normalize(val_batch), nan=0.0)
                val_concept_y = torch.nan_to_num(concept_norm.normalize(val_concept_y), nan=0.0)
                val_y = torch.nan_to_num(output_norm.normalize(val_y), nan=0.0)

                pred, concept_pred = model(val_batch)
                pred = pred*mask
                concept_pred = concept_pred*mask

                val_loss += out_loss_fn(pred, val_y) + (concept_lambda * concept_loss_fn(concept_pred, val_concept_y))
                n_snaps += 1
            val_loss_mean = val_loss / n_snaps 
            if scheduler_name == 'ReduceLROnPlateau':
                # Step the scheduler based on validation loss
                scheduler.step(val_loss_mean)
            else:
                # Step the scheduler every epoch (for schedulers like StepLR)
                scheduler.step()
        output = config['OUTPUT']['dir']
        if epoch % 5 == 0:
            print(f"epoch: {epoch}; loss: {loss_mean:.5f}; val_loss: {val_loss_mean:.5f}")
            print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        losses.append(loss_mean.item())
        val_losses.append(val_loss_mean.item())
        if epoch % config.getint('OUTPUT', 'n_epochs_between_checkpoints') == 0:
            #update to have model save with more detail
            save_model = f"{model_type}"        
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{output}/{save_model}_epoch{epoch}.pt")
            torch.save(losses, f"{output}/losses.pt")
            torch.save(val_losses, f"{output}/val_losses.pt")
     #change to be BESTMODEL
    return model, losses, val_losses

def eval(input_norm, concept_norm, output_norm, model, test_loader):
    concept_loss_fn = config['TRAINING']['concept_loss_fn']
    concept_loss_fn = getattr(torch.nn, concept_loss_fn)()
    out_loss_fn = config['TRAINING']['out_loss_fn']
    out_loss_fn = getattr(torch.nn, out_loss_fn)()
    concept_lambda = config.getfloat('TRAINING', 'concept_lambda')

    test_loss = 0
    with torch.nograd():
        for test_batch, concept_y, y in test_loader:
            test_batch = torch.nan_to_num(input_norm.normalize(test_batch), nan=0.0)
            concept_y = torch.nan_to_num(concept_norm.normalize(concept_y), nan=0.0)
            y = torch.nan_to_num(output_norm.normalize(y), nan=0.0)

            pred, concept_pred = model(test_batch)
            pred = pred*mask
            concept_pred = concept_pred*mask
            test_loss += out_loss_fn(pred, y) + (concept_lambda * concept_loss_fn(concept_pred, concept_y))     
            n_snaps += 1          
        test_loss /= n_snaps
        print(f'test_loss:{test_loss}')
    return test_loss