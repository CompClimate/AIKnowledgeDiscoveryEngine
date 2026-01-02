# trains models
# returns training loss and val loss

from model import UNetCBM
from utils.get_data import get_dataset
#from utils.compute_stats import compute_feature_mean_std, FeatureStandardize
import torch
import importlib
from get_config import parse_section, config

def train(train_loader, val_loader):
    # Training loop
    n_epochs = config.getint('TRAINING', 'epochs')
    concept_loss_fn = config['TRAINING']['concept_loss_fn']
    concept_loss_fn = getattr(torch.nn, concept_loss_fn)()
    out_loss_fn = config['TRAINING']['out_loss_fn']
    out_loss_fn = getattr(torch.nn, out_loss_fn)()
    concept_lambda = config.getfloat('TRAINING', 'concept_lambda')

    model_type = config['MODEL']['type']
    model_module = config['MODEL']['definition']
    module = importlib.import_module(model_module)
    ModelClass = getattr(module, model_type)
    #nonlinearity?
    model = ModelClass(
        n_features=config.getint('MODEL.HYPERPARAMETERS', 'n_features'),
        n_concepts=config.getint('MODEL.HYPERPARAMETERS', 'n_concepts'),
        hidden_dim=config.getint('MODEL.HYPERPARAMETERS', 'hidden_dim'),
        output_dim=config.getint('MODEL.HYPERPARAMETERS', 'output_dim')
    )

    optimizer_name = config['OPTIMIZER']['type']
    optimizer_module = config['OPTIMIZER']['definition']     
    module = importlib.import_module(optimizer_module)
    optimizer_class = getattr(module, optimizer_name)
    optimizer_params = parse_section(config, 'OPTIMIZER.HYPERPARAMETERS')
    optimizer_params = optimizer_class(model.parameters(), **optimizer_params)
    optimizer = optimizer_class(optimizer_params)

    #allow for restart?
    start_epoch = 0

    losses = []
    val_losses = []
    for epoch in range(start_epoch, n_epochs):
        model.train(True)
        loss_final = 0
        train_loss_accum = None
        n_snaps = 0
        for batch, concept_y, y in train_loader:
            #batch = feature_standardizer(batch)
            print('training')
            pred, concept_pred = model(batch)
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
                #val_batch = feature_standardizer(val_batch)
                pred, concept_pred = model(val_batch)
                val_loss += out_loss_fn(pred, val_y) + (concept_lambda * concept_loss_fn(concept_pred, val_concept_y))
                n_snaps += 1
            val_loss_mean = val_loss / n_snaps
            module_name = config['SCHEDULER']['definition']  
            scheduler_name = config['SCHEDULER']['type']  
            module = importlib.import_module(module_name)
            scheduler_class = getattr(module, scheduler_name)
            scheduler_params = parse_section(config, 'SCHEDULER.HYPERPARAMETERS')
            scheduler = scheduler_class(optimizer, **scheduler_params) 
            scheduler.step()
        output = config['OUTPUT']['dir']
        if epoch % config.getint('OUTPUT', 'n_epochs_between_checkpoints') == 0:
            print(f"epoch: {epoch}; loss: {loss_mean:.5f}; val_loss: {val_loss_mean:.5f}")
            print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        losses.append(loss_mean.item())
        val_losses.append(val_loss_mean.item())
        if epoch % config.getint('OUTPUT', 'n_epochs_between_checkpoints') == 0:
            #update to have model save with more detail
            save_model = f"{model_type}"        
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{output}/{save_model}_epoch{epoch}.pt")
            torch.save(losses, f"{output}/losses.pt")
            torch.save(val_losses, f"{output}/val_losses.pt")
    return model #change to be BESTMODEL

def eval(model, test_loader):
    concept_loss_fn = config['TRAINING']['concept_loss_fn']
    concept_loss_fn = getattr(torch.nn, concept_loss_fn)()
    out_loss_fn = config['TRAINING']['out_loss_fn']
    out_loss_fn = getattr(torch.nn, out_loss_fn)()
    concept_lambda = config.getfloat('TRAINING', 'concept_lambda')

    test_loss = 0
    with torch.nograd():
        for test_batch, concept_y, y in test_loader:
            #test_batch = feature_standardizer(test_batch)
            n_snaps += 1
            pred, concept_pred = model(test_batch)
            test_loss += out_loss_fn(pred, y) + (concept_lambda * concept_loss_fn(concept_pred, concept_y))               
        test_loss /= n_snaps
