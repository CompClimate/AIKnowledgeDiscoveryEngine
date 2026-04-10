import os
import torch
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

# Import custom project utilities
from utils import get_config
from utils.compute_stats import Normalize
from utils.load_data import EmulatorDataset

def evaluate_model(model, loader, input_norm, mask_2d):
    """Calculates Accuracy and Recall across the entire dataset (all 3 lead times)."""
    model.eval()
    total_correct = 0
    total_pixels = 0
    total_true_positives = 0
    total_actual_positives = 0
    
    mask_flat = mask_2d.flatten() # 120,800
    
    with torch.no_grad():
        for data, _, target in loader:
            norm_data = input_norm.normalize(data.float())
            output, *_ = model(norm_data)
            
            # Flatten predictions and targets (Size: 362,400)
            pred_binary = (output > 0).float().cpu().numpy().flatten()
            target_binary = target.cpu().numpy().flatten()
            
            # Tile mask to match lead times (repeats mask 3 times)
            num_repeats = len(target_binary) // len(mask_flat)
            full_mask = np.tile(mask_flat, num_repeats)
            
            # Filter only ocean points
            ocean_preds = pred_binary[full_mask == 1]
            ocean_targets = target_binary[full_mask == 1]
            
            # Aggregate stats
            total_correct += (ocean_preds == ocean_targets).sum()
            total_pixels += len(ocean_targets)
            total_true_positives += ((ocean_preds == 1) & (ocean_targets == 1)).sum()
            total_actual_positives += (ocean_targets == 1).sum()
            
    acc = total_correct / total_pixels
    rec = total_true_positives / total_actual_positives if total_actual_positives > 0 else 0
    return acc, rec

# --- 1. SETUP ---
config = get_config.config
home_dir = os.path.expanduser("~")
loc = config['DATASET']['location']

# Load Normalization
stats_path = os.path.join(home_dir, 'normalization_stats.pt')
torch.serialization.add_safe_globals([Normalize])
norms = torch.load(stats_path, weights_only=False)
input_norm = norms['input']

# Load Mask and define Land Mask for plotting
mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
land_mask = (mask_2d == 0)

# Load Dataset
dataset = EmulatorDataset()
val_idx = list(range(int(0.8 * len(dataset)), int(0.9 * len(dataset))))
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)

# --- 2. GLOBAL EVALUATION ---
epochs_to_check = [0, 5, 10, 15]
metrics_results = {}

print(f"{'Epoch':<10} | {'Val Acc':<12} | {'Val Recall':<12}")
print("-" * 40)

for epoch in epochs_to_check:
    model = get_config.get_model()
    path = f'/home/kkringel/temp/20260202_testingMLP/bs_1/PointwiseCBM_epoch{epoch}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    acc, rec = evaluate_model(model, val_loader, input_norm, mask_2d)
    metrics_results[epoch] = (acc, rec)
    print(f"{epoch:<10} | {acc:<12.2%} | {rec:<12.2%}")

# --- 3. VISUALIZATION ---
data, _, target_true = next(iter(val_loader))
fig, axes = plt.subplots(1, len(epochs_to_check) + 1, figsize=(25, 5))

# Plot GT (Lead time 0)
axes[0].set_facecolor('white')
gt_img = np.ma.masked_where(land_mask, target_true[0, 0, 0, :, :].cpu().numpy())
axes[0].imshow(gt_img, vmin=0, vmax=1, cmap='viridis')
axes[0].set_title("GROUND TRUTH\nLead 0")

for i, epoch in enumerate(epochs_to_check):
    model = get_config.get_model()
    path = f'/home/kkringel/temp/20260202_testingMLP/bs_1/PointwiseCBM_epoch{epoch}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        output, *_ = model(input_norm.normalize(data.float()))
        # Shape is (1, 1, 3, 302, 400) -> we take Lead 0
        pred_2d = (output[0, 0, 0, :, :] > 0).float().cpu().numpy()
        
    ax = axes[i+1]
    ax.set_facecolor('white')
    masked_pred = np.ma.masked_where(land_mask, pred_2d)
    ax.imshow(masked_pred, vmin=0, vmax=1, cmap='viridis')
    
    acc, rec = metrics_results[epoch]
    ax.set_title(f"EPOCH {epoch}\nAcc: {acc:.1%}\nRec: {rec:.1%}")

plt.savefig('pointwise_cbm_eval.png', dpi=300, bbox_inches='tight')
print("\nSuccess! Results saved to pointwise_cbm_eval.png")