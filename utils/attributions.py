from captum.attr import GradientShap
from utils.get_data import get_dataset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import xarray as xr
from utils.get_config import config, try_cast
import utils.get_config as get_config
from utils.visualization import find_output_dir


# --- Wrappers ---

class ConceptWrapper(nn.Module):
    """Wraps model to return spatial-mean concept values per lead time.
    Output shape: (B, n_concepts * output_dim)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, concepts = self.model(x)  # (B, n_concepts, output_dim, Y, X)
        return concepts.mean(dim=(-2, -1)).reshape(x.shape[0], -1)


class OutputWrapper(nn.Module):
    """Wraps model to return spatial-mean prediction per lead time.
    Output shape: (B, output_dim)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, _ = self.model(x)  # (B, 1, output_dim, Y, X)
        return output.mean(dim=(-2, -1)).squeeze(1)


# --- Helper functions ---

def _load_model(model_dir, config_path=None):
    """Load model from the latest checkpoint in model_dir."""
    if config_path is not None:
        config.read(config_path)
        print(f'Loaded config from {config_path}', flush=True)
    else:
        saved_config = f'{model_dir}/config.ini'
        if os.path.exists(saved_config):
            config.read(saved_config)
            print(f'Loaded config from {saved_config}', flush=True)

    model_type = config['MODEL']['type']
    model = get_config.get_model()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    pattern = f'{model_dir}/{model_type}_epoch*.pt'
    checkpoints = sorted(glob.glob(pattern),
                         key=lambda p: int(p.split('epoch')[-1].split('.')[0]))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found matching {pattern}')
    latest = checkpoints[-1]
    print(f'Loading checkpoint: {latest}', flush=True)

    checkpoint = torch.load(latest, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, DEVICE


def _get_val_sample(input_norm, val_loader, val_sample_idx, DEVICE):
    """Get a single normalized validation sample on device."""
    val_dataset = val_loader.dataset
    data, concept_true, target_true = val_dataset[val_sample_idx]
    data = data.unsqueeze(0)  # (1, V, T, Y, X)
    data_norm = torch.nan_to_num(input_norm.normalize(data), nan=0.0).to(DEVICE)
    return data_norm, concept_true, target_true


# --- Main attribution function ---

def gradient_shap_inputs(model_dir=None, input_norm=None, concept_norm=None,
                         val_loader=None, val_sample_idx=None, n_baselines=20,
                         output_dir=None, config_path=None):
    """GradientSHAP attribution: pixel-level spatial maps of input importance.

    Saves spatial heatmaps per variable per lead time for both
    input → prediction and input → concepts.
    """
    if model_dir is None:
        model_dir = find_output_dir()
    if output_dir is None:
        output_dir = model_dir

    model, DEVICE = _load_model(model_dir, config_path=config_path)

    if val_sample_idx is None:
        val_sample_idx = config.getint('VISUALIZATION', 'val_sample_idx', fallback=1)
    if input_norm is None or concept_norm is None or val_loader is None:
        input_norm, concept_norm, _, _, val_loader, _ = get_dataset()

    features = try_cast(config['DATASET']['features'])
    concepts = try_cast(config['DATASET']['concepts'])
    offsets = try_cast(config['DATASET']['offset'])
    n_features = len(features)
    n_concepts = len(concepts)
    n_leads = len(offsets)

    data_norm, _, _ = _get_val_sample(input_norm, val_loader, val_sample_idx, DEVICE)

    # Build baselines from other validation samples
    baseline_list = []
    val_dataset = val_loader.dataset
    for i in range(min(n_baselines, len(val_dataset))):
        if i == val_sample_idx:
            continue
        b, _, _ = val_dataset[i]
        b = b.unsqueeze(0)
        b = torch.nan_to_num(input_norm.normalize(b), nan=0.0)
        baseline_list.append(b)
    baselines = torch.cat(baseline_list, dim=0).to(DEVICE)  # (n_baselines, V, T, Y, X)
    print(f'Built {baselines.shape[0]} baselines', flush=True)

    # Load land mask
    loc = config['DATASET']['location']
    mesh = xr.open_zarr(f'{loc}/tmask_crop.zarr')
    mask_2d = mesh['tmaskutil'].isel(t=0, y=slice(0, 302), x=slice(0, 400)).values
    land_mask = (mask_2d == 0)

    # --- GradientSHAP: inputs → prediction ---
    print('Computing GradientSHAP input → prediction...', flush=True)
    output_wrapper = OutputWrapper(model)
    gs = GradientShap(output_wrapper)

    for li in range(n_leads):
        attr = gs.attribute(data_norm, baselines=baselines, target=li)
        attr_np = attr.detach().cpu().numpy()[0]  # (V, T, Y, X)
        spatial_attr = attr_np.mean(axis=1)  # (V, Y, X) — signed, mean over time steps

        fig, axes = plt.subplots(1, n_features, figsize=(n_features * 4, 3.5),
                                 layout='constrained')
        for vi in range(n_features):
            ax = axes[vi]
            vals = np.ma.masked_where(land_mask, spatial_attr[vi])
            im = ax.imshow(vals, cmap='RdBu_r', aspect='equal', origin='lower')
            ax.set_title(features[vi])
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Attribution')
        fig.suptitle(f'GradientSHAP: Input → Prediction (Lead {offsets[li]}mo)')
        save_path = f'{output_dir}/gshap_input_pred_spatial_lead{offsets[li]}.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_path}', flush=True)

    # --- GradientSHAP: inputs → concepts ---
    print('Computing GradientSHAP input → concepts...', flush=True)
    concept_wrapper = ConceptWrapper(model)
    gs_concept = GradientShap(concept_wrapper)

    for li in range(n_leads):
        fig, axes = plt.subplots(n_features, n_concepts,
                                 figsize=(n_concepts * 4, n_features * 3),
                                 layout='constrained')
        for ci in range(n_concepts):
            target_idx = ci * n_leads + li
            attr = gs_concept.attribute(data_norm, baselines=baselines, target=target_idx)
            attr_np = attr.detach().cpu().numpy()[0]  # (V, T, Y, X)
            spatial_attr = attr_np.mean(axis=1)  # (V, Y, X) — signed, mean over time steps

            for vi in range(n_features):
                ax = axes[vi, ci] if n_features > 1 else axes[ci]
                vals = np.ma.masked_where(land_mask, spatial_attr[vi])
                im = ax.imshow(vals, cmap='RdBu_r', aspect='equal', origin='lower')
                if vi == 0:
                    ax.set_title(concepts[ci])
                if ci == 0:
                    ax.set_ylabel(features[vi])
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f'GradientSHAP: Input → Concepts (Lead {offsets[li]}mo)')
        save_path = f'{output_dir}/gshap_input_concept_spatial_lead{offsets[li]}.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {save_path}', flush=True)


if __name__ == "__main__":
    MODEL_DIR   = '/quobyte/maikesgrp/mlhc_cbm/runs/UNetCBM_lam0.5_ep50_lr0.001_bs64_BCELoss_MinMax'
    gradient_shap_inputs(model_dir=MODEL_DIR)
