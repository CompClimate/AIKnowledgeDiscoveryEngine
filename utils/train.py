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
import os

loc = config['DATASET']['location']
mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
mesh = mesh['tmaskutil'].isel(t=0)  # [y, x]
mask = mesh.sel(y=slice(0, 302), x=slice(0,400)).values
mask = torch.tensor(mask, dtype=torch.float32)[None, None, None, :, :]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model.to(DEVICE)
    optimizer = get_config.get_optimizer(model)
    scheduler = get_config.get_scheduler(optimizer)
    scheduler_name = config['SCHEDULER']['type']   

    #allow for restart?
    start_epoch = 0

    losses = []
    val_losses = []
    pred_losses = []
    concept_losses = []
    val_pred_losses = []
    val_concept_losses = []
    for epoch in range(start_epoch, n_epochs):
        model.train(True)
        train_loss_accum = None
        train_pred_loss_accum = None
        train_concept_loss_accum = None
        n_snaps = 0
        for batch, concept_y, y in train_loader:
            batch = torch.nan_to_num(input_norm.normalize(batch), nan=0.0)
            concept_y = torch.nan_to_num(concept_norm.normalize(concept_y), nan=0.0)
            y = torch.nan_to_num(output_norm.normalize(y), nan=0.0)
            batch, concept_y, y = batch.to(DEVICE), concept_y.to(DEVICE), y.to(DEVICE)
            
            pred, concept_pred = model(batch)
            pred = pred*mask
            concept_pred = concept_pred*mask

            pred_loss = out_loss_fn(pred, y)
            concept_loss = concept_loss_fn(concept_pred, concept_y)
            loss = ((1-concept_lambda) * pred_loss) + (concept_lambda * concept_loss)
            n_snaps += 1
            if train_loss_accum:
              train_loss_accum = train_loss_accum.add(loss)
              train_pred_loss_accum = train_pred_loss_accum.add(pred_loss)
              train_concept_loss_accum = train_concept_loss_accum.add(concept_loss)
            else:
              train_loss_accum = loss
              train_pred_loss_accum = pred_loss
              train_concept_loss_accum = concept_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_mean = train_loss_accum / n_snaps
        pred_loss_mean = train_pred_loss_accum / n_snaps
        concept_loss_mean = train_concept_loss_accum / n_snaps

        
        val_loss = 0
        val_pred_loss = 0
        val_concept_loss = 0
        model.eval()
        with torch.no_grad():
            n_snaps = 0
            for val_batch, val_concept_y, val_y in val_loader:
                val_batch = torch.nan_to_num(input_norm.normalize(val_batch), nan=0.0)
                val_batch.to(DEVICE)
                val_concept_y = torch.nan_to_num(concept_norm.normalize(val_concept_y), nan=0.0)
                val_y = torch.nan_to_num(output_norm.normalize(val_y), nan=0.0)

                pred, concept_pred = model(val_batch)
                pred = pred*mask
                concept_pred = concept_pred*mask

                val_pred_loss += out_loss_fn(pred, val_y)
                val_concept_loss += concept_loss_fn(concept_pred, val_concept_y)
                val_loss += ((1-concept_lambda) * out_loss_fn(pred, val_y)) + (concept_lambda * concept_loss_fn(concept_pred, val_concept_y))
                n_snaps += 1
            val_loss_mean = val_loss / n_snaps 
            val_pred_loss_mean = val_pred_loss / n_snaps
            val_concept_loss_mean = val_concept_loss / n_snaps
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
        pred_losses.append(pred_loss_mean.item())
        concept_losses.append(concept_loss_mean.item())
        val_pred_losses.append(val_pred_loss_mean.item())
        val_concept_losses.append(val_concept_loss_mean.item())
        if epoch % config.getint('OUTPUT', 'n_epochs_between_checkpoints') == 0:
            #update to have model save with more detail
            save_model = f"{model_type}"        
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{output}/{save_model}_epoch{epoch}.pt")
            torch.save(losses, f"{output}/losses.pt")
            torch.save(val_losses, f"{output}/val_losses.pt")
            torch.save(pred_losses, f"{output}/pred_losses.pt")
            torch.save(concept_losses, f"{output}/concept_losses.pt")
            torch.save(val_pred_losses, f"{output}/val_pred_losses.pt")
            torch.save(val_concept_losses, f"{output}/val_concept_losses.pt")
     #change to be BESTMODEL
    return model, [losses, pred_losses, concept_losses,], [val_losses,  val_pred_losses, val_concept_losses]

def eval(input_norm, concept_norm, output_norm, model, test_loader):
    concept_loss_fn = config['TRAINING']['concept_loss_fn']
    concept_loss_fn = getattr(torch.nn, concept_loss_fn)()
    out_loss_fn = config['TRAINING']['out_loss_fn']
    out_loss_fn = getattr(torch.nn, out_loss_fn)()
    concept_lambda = config.getfloat('TRAINING', 'concept_lambda')

    test_loss = 0
    pred_loss = 0
    concept_loss = 0
    with torch.nograd():
        for test_batch, concept_y, y in test_loader:
            test_batch = torch.nan_to_num(input_norm.normalize(test_batch), nan=0.0)
            concept_y = torch.nan_to_num(concept_norm.normalize(concept_y), nan=0.0)
            y = torch.nan_to_num(output_norm.normalize(y), nan=0.0)
            test_batch, concept_y, y = test_batch.to(DEVICE), concept_y.to(DEVICE), y.to(DEVICE)

            pred, concept_pred = model(test_batch)
            pred = pred*mask
            concept_pred = concept_pred*mask
            pred_loss += out_loss_fn(pred, y)
            concept_loss += concept_loss_fn(concept_pred, concept_y)
            test_loss += ((1-concept_lambda) * out_loss_fn(pred, y)) + (concept_lambda * concept_loss_fn(concept_pred, concept_y))     
            n_snaps += 1          
        test_loss /= n_snaps
        pred_loss /= n_snaps
        concept_loss /= n_snaps
        print(f'test_loss:{test_loss}')
        print(f'pred_loss:{pred_loss}')
        print(f'concept_loss:{concept_loss}')
    return [test_loss, pred_loss, concept_loss]