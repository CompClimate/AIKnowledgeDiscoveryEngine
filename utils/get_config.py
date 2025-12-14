#imports need to be added
import torch
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def get_config():