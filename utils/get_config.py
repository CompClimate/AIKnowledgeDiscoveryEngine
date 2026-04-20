#imports need to be added
import argparse
import ast
import configparser
import importlib
import inspect

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--config_file', required=True, help='Path to config file')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_file)

def try_cast(value):
    # Tries int, then float, then returns string
    for cast in (int, float, ast.literal_eval):
        try:
            return cast(value)
        except ValueError:
            continue
    return value

def parse_section(section):
    """Parses a section of the config and attempts to cast values to appropriate types."""
    return {k: try_cast(v) for k, v in config[section].items()}

def get_model():
    model_type = config['MODEL']['type']
    model_module = config['MODEL']['definition']
    module = importlib.import_module(model_module)
    ModelClass = getattr(module, model_type)
    sig = inspect.signature(ModelClass.__init__)
    params = set(sig.parameters.keys()) - {'self'}

    kwargs = {
        'n_features': len(try_cast(config['DATASET']['features']))* config.getint('DATASET', 'context_window'), 
        'n_concepts': len(try_cast(config['DATASET']['concepts'])),
        'output_dim': len(try_cast(config['DATASET']['offset'])),
        'dropout': config.getfloat('MODEL.HYPERPARAMETERS', 'dropout')
    }
    if 'num_layers' in params:
        kwargs['num_layers'] = config.getint('MODEL.HYPERPARAMETERS', 'depth')
    if 'hidden_dim' in params:
        kwargs['hidden_dim'] = config.getint('MODEL.HYPERPARAMETERS', 'width')
    if 'channels_list' in params and config.has_option('MODEL.HYPERPARAMETERS', 'channels_list'):
        kwargs['channels_list'] = try_cast(config['MODEL.HYPERPARAMETERS']['channels_list'])
    if 'spatial_patch_size' in params:
        kwargs['spatial_patch_size'] = config.getint('MODEL.HYPERPARAMETERS', 'spatial_patch_size')
    if 'temporal_patch_size' in params:
        kwargs['temporal_patch_size'] = config.getint('MODEL.HYPERPARAMETERS', 'temporal_patch_size')
    if 'd_model' in params:
        kwargs['d_model'] = config.getint('MODEL.HYPERPARAMETERS', 'd_model')
    if 'num_heads' in params:
        kwargs['num_heads'] = config.getint('MODEL.HYPERPARAMETERS', 'num_heads')
    if 'd_ff' in params:
        kwargs['d_ff'] = config.getint('MODEL.HYPERPARAMETERS', 'd_ff')
    

    model = ModelClass(**kwargs)
    return model

def get_optimizer(model):
    optimizer_name = config['OPTIMIZER']['type']
    optimizer_module = config['OPTIMIZER']['definition']     
    module = importlib.import_module(optimizer_module)
    optimizer_class = getattr(module, optimizer_name)
    optimizer_params = parse_section('OPTIMIZER.HYPERPARAMETERS')
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    return optimizer

def get_scheduler(optimizer):
    module_name = config['SCHEDULER']['definition']  
    scheduler_name = config['SCHEDULER']['type']  
    module = importlib.import_module(module_name)
    scheduler_class = getattr(module, scheduler_name)
    scheduler_params = parse_section('SCHEDULER.HYPERPARAMETERS')
    scheduler = scheduler_class(optimizer, **scheduler_params)
    return scheduler