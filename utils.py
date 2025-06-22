"""
🧠 Utility Functions for Document Memory Guidance

This module provides visualization and utility functions for the document
memory guidance experiments, including loss tracking, document representation
visualization, and experiment setup.

Author: Bumjin Park
"""

import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import torch
import os 
import json 
import imageio
from skimage import transform
from PIL import Image
from omegaconf import OmegaConf 
import datetime 
import random

# ==========================================
# 🎨 VISUALIZATION SETTINGS
# ==========================================

sns.set_style("whitegrid")

# ==========================================
# 📊 VISUALIZATION FUNCTIONS
# ==========================================

def save_summary(lm, tokenizer, flags, step):
    """
    Save comprehensive training summary including losses, document representations,
    and logit lens visualizations.
    
    Args:
        lm: Language model with memory module
        tokenizer: Tokenizer for decoding
        flags: Experiment configuration flags
        step: Current training step
    """
    save_dir = os.path.join(flags.save_dir, "summaries")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 📈 Save loss plots
    _save_loss_plots(flags, save_dir, step)
    
    # 📊 Save document representations
    _save_document_representations(lm, save_dir, step)
    
    # 🔍 Save logit lens analysis
    _save_logit_lens(lm, tokenizer, flags, save_dir, step)
    
    # 🎬 Create animated GIFs
    _create_animated_summaries(flags, save_dir)

def _save_loss_plots(flags, save_dir, step):
    """Save training loss visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=100)
    fig.suptitle(f"Training Losses - Step: {step}")
    
    axes[0].set_title("Total Loss")
    axes[1].set_title("LM Loss (Positive)")
    axes[2].set_title("LM Loss (Negative)")
    
    axes[0].plot(flags.training_info.losses, lw=2, c='blue')
    axes[1].plot(flags.training_info.losses_positive, lw=2, c='green')
    axes[2].plot(flags.training_info.losses_negative, lw=2, c='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'step{step}_losses.png'), bbox_inches='tight')
    plt.close()

def _save_document_representations(lm, save_dir, step):
    """Save document representation heatmap."""
    fig, axes = plt.subplots(1, 1, figsize=(10, 4), dpi=100)
    fig.suptitle(f"Document Representations - Step: {step}")
    axes.set_title("Document Memory Selections")
    
    # Extract document representations
    key_stacked = [] 
    with torch.no_grad():
        for k, param in lm.memory_module.keys.items():
            key_rep = lm.memory_module.compute_key(param.data.clone().unsqueeze(0))
            key_stacked.append(key_rep.squeeze(0).detach().cpu().numpy())
    
    key_stacked = np.stack(key_stacked, axis=0)
    sns.heatmap(key_stacked, cmap='seismic', 
                center=0, vmax=1, vmin=-1, linewidths=0.5, linecolor='black', ax=axes)
    
    axes.set_ylabel("Documents")
    axes.set_xlabel("Vector Dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'step{step}_docreps.png'), bbox_inches='tight')
    plt.close()

def _save_logit_lens(lm, tokenizer, flags, save_dir, step):
    """Save logit lens analysis for document memory."""
    logits_stacked = []
    topk_stacked = {}
    K = 20
    
    with torch.no_grad():
        for doc in range(len(lm.memory_module.keys)):
            # Get memory representation
            key = torch.tensor(lm.memory_module.keys[doc].data.clone().detach().cpu().numpy()).float().unsqueeze(0)
            key = key.to(lm.memory_module.down_proj.weight.device)
            mem = lm.memory_module.down_proj(key)
            
            # Apply logit lens
            head_layer = None 
            if "pythia" in flags.lm_name:
                head_layer = lm.lm.embed_out
            elif "llama2" in flags.lm_name:
                head_layer = lm.lm.lm_head
            
            mem = mem.to(head_layer.weight.device)
            logits = head_layer(mem).squeeze(0)
            logits_stacked.append(logits.cpu().detach().numpy())
            
            # Get top-k tokens
            topk = torch.topk(logits, K, dim=0).indices.detach().cpu()
            topk_stacked[doc] = []
            for d in topk:
                topk_stacked[doc].append(tokenizer.decode([d]))
    
    logits_stacked = np.stack(logits_stacked, axis=0)
    
    # Visualize logits
    fig, axes = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
    fig.suptitle(f"Logit Lens Analysis - Step: {step}")
    sns.heatmap(logits_stacked, cmap='Blues', ax=axes)
    axes.set_title("Token Logits")
    axes.set_ylabel("Documents")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'step{step}_logits.png'), bbox_inches='tight')
    plt.close()
    
    # Save top-k tokens
    with open(os.path.join(save_dir, f'step{step}_topk.json'), 'w') as outfile:
        for k, v in topk_stacked.items():
            json.dump({k: v}, outfile)
            outfile.write("\n")
    
    with open(os.path.join(flags.save_dir, 'topk.json'), 'w') as outfile:
        for k, v in topk_stacked.items():
            json.dump({k: v}, outfile)
            outfile.write("\n")

def _create_animated_summaries(flags, save_dir):
    """Create animated GIFs from saved images."""
    images_docrep = []
    images_losses = []
    images_logits = []
    
    summary_path = os.path.join(flags.save_dir, 'summaries')
    
    for filename in sorted(os.listdir(summary_path), key=lambda x: int(x.split("_")[0][4:])):
        if 'docreps.png' in filename:        
            images_docrep.append(
                Image.fromarray((transform.resize(imageio.imread(os.path.join(summary_path, filename)), (400, 1000), mode='constant')*255).astype(np.uint8))
            )
        if 'losses.png' in filename:        
            images_losses.append(
                Image.fromarray((transform.resize(imageio.imread(os.path.join(summary_path, filename)), (300, 900), mode='constant')*255).astype(np.uint8))
            )
        if 'logits.png' in filename:        
            images_logits.append(
                Image.fromarray((transform.resize(imageio.imread(os.path.join(summary_path, filename)), (300, 1200), mode='constant')*255).astype(np.uint8))
            )
    
    # Save animated GIFs
    imageio.mimsave(os.path.join(flags.save_dir, 'docreps.gif'), images_docrep, fps=1)
    imageio.mimsave(os.path.join(flags.save_dir, 'logits.gif'), images_logits, fps=1)
    imageio.mimsave(os.path.join(flags.save_dir, 'losses.gif'), images_losses, fps=1)
    plt.close()

# ==========================================
# 🔧 EXPERIMENT SETUP FUNCTIONS
# ==========================================

def prepare_script(args, file_name="run", update_and_make_save_dir=True):
    """
    Prepare experiment configuration and setup.
    
    Args:
        args: Command line arguments
        file_name: Base name for experiment directory
        update_and_make_save_dir: Whether to create new save directory
    
    Returns:
        OmegaConf: Configuration object with all experiment settings
    """
    flags = OmegaConf.create({})
    flags.done = False
    flags.datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")
    
    # Set all arguments as flags
    for k, v in vars(args).items():
        print(f"📋 {k}: {v}")
        setattr(flags, k, v)

    # 🎲 Set random seeds for reproducibility
    _set_random_seeds(args.seed)

    # 📁 Create experiment directories
    if update_and_make_save_dir:
        flags.save_dir = _create_experiment_directory(args.save_dir, file_name)
        OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))
    
    return flags

def _set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

def _create_experiment_directory(base_dir, file_name):
    """Create unique experiment directory."""
    train_idx = 0
    while os.path.exists(os.path.join(base_dir, f'{file_name}_{train_idx}')):
        train_idx += 1 
    
    save_dir = os.path.join(base_dir, f'{file_name}_{train_idx}')
    os.makedirs(save_dir)
    return save_dir 