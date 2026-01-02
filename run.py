#from utils.get_config import get_config
from utils.train import train, eval
#from utils.visualization import visualize
from utils.get_data import get_dataset

def run():
    #get_config()
    train_loader, val_loader, test_loader = get_dataset()
    model = train(train_loader, val_loader)
    eval(model, test_loader)

if __name__ == '__main__':
    run()