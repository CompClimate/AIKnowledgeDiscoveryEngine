from utils.train import train, eval, make_output_dir
from utils.visualization import visualize, plot_sample, plot_sample_pred_only
from utils.get_data import get_dataset
from utils.get_config import config
import torch
import time
from inference import run_inference, threshold_analysis

def run():
    start = time.time()
    input_norm, concept_norm, output_norm, train_loader, val_loader, test_loader = get_dataset()
    get_data_done = time.time()
    print('get data took ', get_data_done - start)
    model, train_losses, val_losses = train(input_norm, concept_norm, output_norm, train_loader, val_loader)
    train_done = time.time()
    print('training took ', train_done - get_data_done)

    plot_test = config.getboolean('OUTPUT', 'plot_test', fallback=False)
    if plot_test:
        test_results = eval(input_norm, concept_norm, output_norm, model, test_loader)
        # Append test results to detailed_losses.pt
        from utils.visualization import find_output_dir
        output_dir = find_output_dir()
        losses_path = f'{output_dir}/detailed_losses.pt'
        detailed = torch.load(losses_path, weights_only=False)
        detailed.update(test_results)
        torch.save(detailed, losses_path)
        test_done = time.time()
        print('testing done ', test_done - train_done)
    visualize()
    plot_sample(input_norm=input_norm, concept_norm=concept_norm, val_loader=val_loader)
    plot_sample_pred_only(input_norm=input_norm, val_loader=val_loader, thresholds=[0.5, 0.4, 0.3, 0.25])
    run_inference()


if __name__ == '__main__':
    run()