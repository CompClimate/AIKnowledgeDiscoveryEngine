from utils.get_config import get_config
from utils.train import train
from utils.visualization import visualize

def run():
    get_config()
    train()
    visualize()

if __name__ == '__main__':
    run()