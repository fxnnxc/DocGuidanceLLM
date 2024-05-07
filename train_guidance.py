import os 
import torch 
import numpy as np 
from tqdm import tqdm 
from omegaconf import OmegaConf 
from torch.utils.data import DataLoader
from transformers import default_data_collator
from utils import prepare_script
from foundations.llama2 import (get_llama2,
                                                get_llama2_lm_info,
                                                get_llama2_block,
)
from foundations.pythia import (
    get_pythia,
    get_pythia_lm_info,
    get_pythia_block,
)                                                         
                                                        
from wikitext import get_wikitext, get_selected_wiki_page_data_dict
from utils import save_summary
# -----------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lm_name", )
parser.add_argument("--lm_size", )
parser.add_argument("--lm_cache_dir", )
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_accumulation_steps", type=int, default=1)
parser.add_argument("--num_warmup_steps", type=int,)
parser.add_argument("--grad_clip_coeff", type=float, default=10)
parser.add_argument("--lr_scheduler", type=str, default='cosine')

parser.add_argument("--data_cache_dir",)
parser.add_argument("--max_labels", type=int )
parser.add_argument("--max_length", type=int )
parser.add_argument("--segment_length", type=int )
parser.add_argument("--max_segements", type=int, default=30)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir")

# ---- hook module & custom 
from hook_lm import update_hook_memory_pareser
from document_memories import update_llm_guidance_parser
parser = update_hook_memory_pareser(parser)
parser = update_llm_guidance_parser(parser)

args = parser.parse_args()
flags = prepare_script(args)

# ----------------- 🥕 Get Data -------------------
wikipage_dataset = get_wikitext('wikitext-103-v1', flags.data_cache_dir)

# --> 🥕 filter if fever_wikipage is not in the wikipages 
full_pages_in_wikipages = wikipage_dataset['id']
indices = np.random.choice([i for i in range(len(full_pages_in_wikipages))], flags.max_labels, replace=False)
selected_pages = [full_pages_in_wikipages[i] for i in indices]

# --> 🥕 make wikipage text data and  filter out non-exisitng contexts 
wiki_data_dict = get_selected_wiki_page_data_dict(wikipage_dataset, selected_pages, 
                                                  flags.segment_length, flags.max_segements)
selected_pages = [s for s in selected_pages if len(wiki_data_dict[s]) >0 ]
wiki_data_dict = {k:v for k, v in wiki_data_dict.items() if k in selected_pages}
flags.data_info = {'selected_pages': selected_pages}

keys = [str(i) for i in range(len(selected_pages))]
key_names = selected_pages

# ----------------- 🥕 Get Foundataion Model -------------------
# ----------------- 🦕 Use Adapter Based Model -------------------
from hook_lm import HookMemoryAdaptedLLM
from document_memories import LinearDocumentKeyMemory, NonLinearDocumentKeyMemory
assert flags.hook_memory_layer is not None
assert flags.hook_memory_dim is not None
if "llama2" in flags.lm_name:
    lm, tokenizer = get_llama2(flags.lm_name, flags.lm_size, flags.lm_cache_dir, num_gpus=flags.num_gpus,)
    info = get_llama2_lm_info( lm)
    lm_block = get_llama2_block(lm, flags.hook_memory_layer)
elif 'pythia' in flags.lm_name:
    lm, tokenizer = get_pythia(flags.lm_name, flags.lm_size, flags.lm_cache_dir, num_gpus=flags.num_gpus,)
    info = get_pythia_lm_info( lm)
    lm_block = get_pythia_block(lm, flags.hook_memory_layer)
    
lm = HookMemoryAdaptedLLM(lm)
lm.set_hooked_module(lm_block)

model_dim=info['model_dim'] 
intermediate_size=flags.hook_memory_dim
lm_act=flags.lm_act # relu 
key_dim=flags.key_dim # 16 
key_activation=flags.key_activation

if flags.document_key_module_type == "linear":
    module = LinearDocumentKeyMemory(model_dim, intermediate_size, lm_act, keys, key_dim, key_activation)
elif flags.document_key_module_tyep == "nonlinear":
    module = NonLinearDocumentKeyMemory(model_dim, intermediate_size, lm_act, keys, key_dim, key_activation, 
                 flags.key_internal_dim, flags.key_internal_activation, flags.key_internal_layers)
else:
    raise ValueError("not implemented module: " + str(flags.document_key_module_type))
lm.set_memory_module(module)
        
# --> 🥕 Tokenize 
def process(samples):
    text  = samples['text']
    batch_inputs = tokenizer(text, max_length=flags.max_length, padding=True, truncation=False, return_tensors='pt') 
    return batch_inputs

tokenizer.padding_side = "left"
num_proc=1 
tokenized_wiki_dict = {}
for page, dataset in wiki_data_dict.items():
    dataset = dataset.map(  
            process,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset...",
            batch_size=flags.batch_size,
            remove_columns= ['text',]
    )
    tokenized_wiki_dict[str(key_names.index(page))] = dataset

dataloaders = {}
for page_index_str, dataset in tokenized_wiki_dict.items():
    dataloader = DataLoader(dataset, batch_size=flags.batch_size, 
                                        shuffle=False, 
                                        collate_fn=default_data_collator,
                                        pin_memory=True)
    dataloaders[page_index_str] = dataloader
    
    
flags.num_minibatch_in_epoch = sum([len(v) for v in dataloaders.values()])
flags.num_training_steps = flags.num_epochs * flags.num_minibatch_in_epoch
optimizer = torch.optim.AdamW(lm.memory_module.parameters(), lr=flags.lr,)

# --- 🥕 Training Procedure : datalaoders ----
key_names = list(dataloaders.keys())
flags.training_info ={'losses':[], 'losses_positive':[], 'losses_negative':[]}

lm.zero_grad()
with tqdm(range(flags.num_training_steps)) as pbar:
    pbar.set_description(flags.save_dir)
    for e in range(flags.num_epochs):
        lm.train()
        losses = []
        losses_positive = []
        losses_negative = []
        for key_index, name in enumerate(key_names):
            current_dataloader = dataloaders[str(key_index)]
            # Wikitext ----------------------------------------------
            for batch in current_dataloader:
                pbar.update(1)
                for k, v in batch.items():
                    batch[k] = v.to(flags.device).long()
                    
                # 🦕 gpt-like logic
                batch['labels'] = batch['input_ids']
                
                # --- 🥕 positive loss handling 
                lm.memory_module.set_key(str(key_index))
                outputs = lm.forward(input_ids=batch['input_ids'], 
                                        attention_mask=batch['attention_mask'], 
                                        labels=batch['labels'], )
                lm.memory_module.remove_key()
                loss_positive = outputs.loss
                
                # --- 🥕 negative loss handling 
                loss_positive = loss_positive / flags.num_accumulation_steps
                # loss with negative  key
                def get_custom_key(memory_module, negative_key_type, current_key):
                    if negative_key_type == "zero":
                        custom_key = memory_module.generate_random_key()
                        custom_key.fill_(0)
                    elif negative_key_type == "other":
                        candidates = list(set(key_names) - set([current_key])) 
                        other = np.random.choice(candidates)
                        custom_key = memory_module.get_key(other)
                    elif negative_key_type == "perturb":
                        current_key = memory_module.get_key(current_key) 
                        custom_key = current_key + torch.randn_like(current_key, device=current_key.device) * current_key.max() * flags.delta
                    elif negative_key_type == "random":
                        custom_key = memory_module.generate_random_key()
                    else:
                        raise ValueError("not implemented negative key type: " + str(negative_key_type))
                    return custom_key

                lm.memory_module.set_key(custom_key=get_custom_key(lm.memory_module, flags.negative_key_type, str(key_index)))
                outputs = lm.forward(input_ids=batch['input_ids'], 
                                        attention_mask=batch['attention_mask'], 
                                        labels=batch['labels'], )
                lm.memory_module.remove_key()

                loss_negative = outputs.loss 
                loss_negative = flags.guidance * torch.clamp_max(
                                                        torch.nn.MSELoss()(loss_negative, 
                                                                            torch.tensor(flags.target_negative_ce, 
                                                                                         device=loss_negative.device)), 
                                                        loss_positive)
                loss = loss_positive + loss_negative 
                
                loss.backward()
                losses.append(loss.item())
                losses_positive.append(loss_positive.item())
                losses_negative.append(loss_negative.item())
                if ((pbar.n + 1) % flags.num_accumulation_steps == 0): 
                    torch.nn.utils.clip_grad_norm_(lm.memory_module.parameters(), flags.grad_clip_coeff)
                    optimizer.step()
                    lm.zero_grad()
                    optimizer.zero_grad()
                    # lr_schedulr.step()
                    
                pbar.set_postfix({"loss":f"{loss.item():.4f}", 
                                  "P": f"{loss_positive.item():.4f}",
                                  "N": f"{loss_negative.item():.4f}",
                                  })
            
        # end of epoch
        flags.training_info['losses'].append(float(f"{np.nanmean(losses):.5f}"))
        flags.training_info['losses_positive'].append(float(f"{np.nanmean(losses_positive):.5f}"))
        flags.training_info['losses_negative'].append(float(f"{np.nanmean(losses_negative):.5f}"))
        OmegaConf.save(flags, os.path.join(flags.save_dir, 'config.yaml')) 
        if e % flags.save_freq == 0:
            save_summary(lm, tokenizer, flags, pbar.n)
            
# end training
path = os.path.join(flags.save_dir, "model.pt")
lm.save(path)
OmegaConf.save(flags, os.path.join(flags.save_dir, 'config.yaml')) 
print(f"[INFO] 🥕 model is saved at: {path}")