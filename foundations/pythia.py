
import os
import torch 
from transformers import GPTNeoXForCausalLM, AutoTokenizer
PYTHIA_MODEL_SIZES_LAYERS = {
    '70m':6, 
    '160m':12, 
    '410m':24, 
    '1b':16, 
    '1.4b':24, 
    '2.8b':32, 
    '6.9b':32, 
    '12b':36
}

def infer_pythia_device_map(model_size, split:int):
    model_size = PYTHIA_MODEL_SIZES_LAYERS[model_size]
    device_map = {'gpt_neox.embed_in':0, 
                  'gpt_neox.final_layer':split-1,
                  'gpt_neox.final_layer_norm':split-1,
                  'embed_out': split-1,
                  }
    gpu = 0
    per_block = model_size//split
    for i in range(1, model_size+1):
        device_map[f'gpt_neox.layers.{i-1}'] = gpu
        if i % per_block ==0 and  gpu < split-1:
            gpu += 1
    return device_map        

def get_pythia(lm_name, lm_size, lm_cache_dir, num_gpus:int, precision=None):
    if num_gpus == 0:
        my_device_map = "cpu"
    else:
        my_device_map = infer_pythia_device_map(lm_size, num_gpus)
        
    if precision=='int8':
        precision_args = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    elif precision=='half':
        precision_args = dict(torch_dtype=torch.float16)
    else:
        precision_args = {}
        
    if lm_size in PYTHIA_MODEL_SIZES_LAYERS.keys():
        model = GPTNeoXForCausalLM.from_pretrained(
            os.path.join('EleutherAI', f"pythia-{lm_size}-deduped"),
            revision='step143000',
            cache_dir=lm_cache_dir if lm_cache_dir != "none" else None,
            device_map = my_device_map,
            **precision_args,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join('EleutherAI', f"pythia-{lm_size}-deduped"),
            revision='step143000',
            cache_dir=lm_cache_dir if lm_cache_dir != "none" else None,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def get_pythia_lm_info(lm):
    info = {
        'model_dim':lm.config.hidden_size,
        'memory_size':lm.config.intermediate_size,
        "num_layers":lm.config.num_hidden_layers
    }
    return info 
     
def get_pythia_block(lm, layer_index):
    return lm.gpt_neox.layers[layer_index]           