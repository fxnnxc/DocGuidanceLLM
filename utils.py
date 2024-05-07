import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import torch
import os 
import json 
import imageio
from skimage import transform
from PIL import Image

sns.set_style("whitegrid")

def save_summary(lm, tokenizer, flags, step):
    save_dir = os.path.join(flags.save_dir, "summaries",)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 🖊️ save losses 
    fig, axes = plt.subplots(1,3, figsize=(9,3), dpi=100)
    fig.suptitle(f"step:{step}")
    axes[0].set_title("total loss")
    axes[1].set_title("lm loss")
    axes[0].plot(flags.training_info.losses, lw=2, c='blue')
    axes[1].plot(flags.training_info.losses_positive, lw=2, c='green')
    axes[2].plot(flags.training_info.losses_negative, lw=2, c='red')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'step{step}_losses.png'), bbox_inches='tight')
    
    # 🖊️ save document representations
    # fig, axes = plt.subplots(2,1, figsize=(10, 8), dpi=100)
    # fig.suptitle(f"step:{step}")
    # axes[0].set_title("Document Representations")
    # axes[1].set_title("Document Memory Selections")

    
    # doc_stacked = []
    # for k, param in lm.memory_module.keys.items():
    #     doc_stacked.append(param.data.clone().detach().cpu().numpy())
    # doc_stacked = np.stack(doc_stacked, axis=0)
    # sns.heatmap(doc_stacked, cmap='seismic', center=0,  linewidths=0.5, linecolor='black', ax=axes[0])

    
    # # 🖊️ save memory selections 
    # key_stacked = [] 
    # with torch.no_grad():
    #     for k, param in lm.memory_module.keys.items():
    #         key_rep = lm.memory_module.compute_key(param.data.clone().unsqueeze(0))
    #         key_stacked.append(key_rep.squeeze(0).detach().cpu().numpy())
    # key_stacked = np.stack(key_stacked, axis=0)
    # sns.heatmap(key_stacked, cmap='seismic', 
    #             center=0, vmax=1, vmin=-1, linewidths=0.5, linecolor='black', ax=axes[1])
    # axes[0].set_ylabel("Docs")
    # axes[1].set_ylabel("Docs")
    # axes[0].set_xlabel("Vector Dimension")
    # axes[1].set_xlabel("Vector Dimension")
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'step{step}_docreps.png'), bbox_inches='tight')
    # plt.close()
    
    # # 🖊️ save memory selections 
    fig, axes = plt.subplots(1,1, figsize=(10, 4), dpi=100)
    fig.suptitle(f"step:{step}")
    axes.set_title("Document Representations")
    
    # 🖊️ save memory selections 
    key_stacked = [] 
    with torch.no_grad():
        for k, param in lm.memory_module.keys.items():
            key_rep = lm.memory_module.compute_key(param.data.clone().unsqueeze(0))
            key_stacked.append(key_rep.squeeze(0).detach().cpu().numpy())
    key_stacked = np.stack(key_stacked, axis=0)
    sns.heatmap(key_stacked, cmap='seismic', 
                center=0, vmax=1, vmin=-1, linewidths=0.5, linecolor='black', ax=axes)
    axes.set_ylabel("Docs")
    axes.set_xlabel("Vector Dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'step{step}_docreps.png'), bbox_inches='tight')
    plt.close()

    # 🖊️ save logit lens 
    logits_stacked = []
    topk_stakced = {}
    K=20
    with torch.no_grad():
        for doc in range(key_stacked.shape[0]):
            # get memory 
            key = torch.tensor(key_stacked[doc]).float().unsqueeze(0)
            key = key.to(lm.memory_module.down_proj.weight.device)
            mem = lm.memory_module.down_proj(key)
            
            # logit lens 
            head_layer = None 
            if "pythia" in flags.lm_name:
                 head_layer = lm.lm.embed_out
            elif "llama2" in flags.lm_name :
                 head_layer = lm.lm.lm_head
            mem = mem.to(head_layer.weight.device)
            
            logits = head_layer(mem).squeeze(0)
            logits_stacked.append(logits.cpu().detach().numpy())
            topk = torch.torch.topk(logits, K, dim=0).indices.detach().cpu()
            topk_stakced[doc] = []
            for d in topk:
                topk_stakced[doc].append(tokenizer.decode([d]))
    logits_stacked = np.stack(logits_stacked, axis=0)
    # visualize logits
    fig, axes = plt.subplots(1,1,figsize=(12,3), dpi=100)
    fig.suptitle(f"step:{step}")
    sns.heatmap(logits_stacked, cmap='Blues',ax=axes)
    axes.set_title("logit lens")
    axes.set_ylabel("Docs")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'step{step}_logits.png'), bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(save_dir, f'step{step}_topk.json'), 'w') as outfile:
        for k, v in topk_stakced.items():
            json.dump({k:v}, outfile)
            outfile.write("\n")
    with open(os.path.join(flags.save_dir, 'topk.json'), 'w') as outfile:
        for k, v in topk_stakced.items():
            json.dump({k:v}, outfile)
            outfile.write("\n")    
        
    # make gifs at the base directory
    images_docrep = []
    images_losses = []
    images_logits = []
    summary_path = os.path.join(flags.save_dir, 'summaries')
    for filename in sorted(os.listdir(summary_path), key=lambda x:int(x.split("_")[0][4:])):
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
                Image.fromarray((transform.resize(imageio.imread(os.path.join(summary_path, filename)), (300, 1200,), mode='constant')*255).astype(np.uint8))
                )
    imageio.mimsave(os.path.join(flags.save_dir, 'docreps.gif'), images_docrep, fps=1)
    imageio.mimsave(os.path.join(flags.save_dir, 'logits.gif'),  images_logits, fps=1)
    imageio.mimsave(os.path.join(flags.save_dir, 'losses.gif'),  images_losses, fps=1)
    plt.close()
    


from omegaconf import OmegaConf 
import datetime 
import random 
import torch 
import numpy as np 
import os 

def prepare_script(args, file_name="run", update_and_make_save_dir=True):
    flags  = OmegaConf.create({})
    flags.done = False
    flags.datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")
    for k, v in vars(args).items():
        print(">>>", k, ":" , v)
        setattr(flags, k, v)

    # ---- Set seed ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    # ---- make dirs ----
    if update_and_make_save_dir:
        train_idx=0
        while os.path.exists(os.path.join(args.save_dir, f'{file_name}_{train_idx}')):
            train_idx+=1 
        flags.save_dir = os.path.join(args.save_dir, f'{file_name}_{train_idx}')
        os.makedirs(flags.save_dir)

        OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))
    return flags 