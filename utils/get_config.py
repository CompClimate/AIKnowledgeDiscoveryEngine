#imports need to be added
import argparse
import ast
import configparser
import importlib

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

def parse_section(config, section):
    """Parses a section of the config and attempts to cast values to appropriate types."""
    return {k: try_cast(v) for k, v in config[section].items()}

def get_model():
    model_type = config['MODEL']['type']
    model_module = config['MODEL']['definition']
    module = importlib.import_module(model_module)
    ModelClass = getattr(module, model_type)
    #nonlinearity?
    model = ModelClass(
        n_features=len(try_cast(config['DATASET']['features']))*config.getint('DATASET', 'context_window'),
        n_concepts=len(try_cast(config['DATASET']['concepts'])),
        hidden_dim=config.getint('MODEL.HYPERPARAMETERS', 'width'),
        output_dim=len(try_cast(config['DATASET']['offset']))
    )
    return model

def get_optimizer(model):
    optimizer_name = config['OPTIMIZER']['type']
    optimizer_module = config['OPTIMIZER']['definition']     
    module = importlib.import_module(optimizer_module)
    optimizer_class = getattr(module, optimizer_name)
    optimizer_params = parse_section(config, 'OPTIMIZER.HYPERPARAMETERS')
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    return optimizer

def get_scheduler(optimizer):
    module_name = config['SCHEDULER']['definition']  
    scheduler_name = config['SCHEDULER']['type']  
    module = importlib.import_module(module_name)
    scheduler_class = getattr(module, scheduler_name)
    scheduler_params = parse_section(config, 'SCHEDULER.HYPERPARAMETERS')
    scheduler = scheduler_class(optimizer, **scheduler_params)
    return scheduler
