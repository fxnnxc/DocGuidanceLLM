import os 
import torch 
from transformers import LlamaForCausalLM, LlamaTokenizer

LLAMA_MODEL_SIZES_LAYERS = {
    '7b':32, 
    '13b':40, 
}

def get_llama2(lm_name, lm_size, lm_cache_dir, num_gpus:int, precision=None):
    if num_gpus == 0:
        my_device_map='cpu'
    else:
        my_device_map = infer_llama_device_map(lm_size, num_gpus)
    
    if lm_name=="llama2":
        model_name = f"meta-llama/Llama-2-{lm_size}"
        link = "https://huggingface.co/collections/meta-llama/llama-2-family-661da1f90a9d678b6f55773b"
        raise ValueError("[Error] llama2 is not available in huggingface. See:" +link )
    elif lm_name=="llama2_hf":
        model_name = f"meta-llama/Llama-2-{lm_size}-hf"
    elif lm_name=="llama2_chat":
        model_name = f"meta-llama/Llama-2-{lm_size}-chat"
        link = "https://huggingface.co/collections/meta-llama/llama-2-family-661da1f90a9d678b6f55773b"
        raise ValueError("[Error] llama2_chat is not available in huggingface. See:" +link )
    elif lm_name=="llama2_chat_hf":
        model_name = f"meta-llama/Llama-2-{lm_size}-chat-hf"
    else:
        raise ValueError(f"not implemented llama2: {lm_name}")
    if precision=='int8':
        precision_args = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    elif precision=='half':
        precision_args = dict(torch_dtype=torch.float16)
    else:
        precision_args = {}
    print(f"I: loding llama2 from: {lm_cache_dir}")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        cache_dir=lm_cache_dir if lm_cache_dir != "none" else None,
        device_map=my_device_map,
        **precision_args,
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name, 
        cache_dir=lm_cache_dir if lm_cache_dir != "none" else None
    )
    tokenizer.sep_token_id =  tokenizer.eos_token_id 
    tokenizer.pad_token_id =  tokenizer.eos_token_id 
    return model, tokenizer

# --------------------------------------------------------------------

def infer_llama_device_map(model_size, split:int):
    model_size = LLAMA_MODEL_SIZES_LAYERS[model_size]
    device_map = {'model.embed_tokens':0, 
                  'model.norm':split-1,
                  'lm_head':split-1,
                  }
    gpu = 0
    per_block = model_size//split
    for i in range(1, model_size+1):
        device_map[f'model.layers.{i-1}'] = gpu
        if i % per_block ==0 and  gpu < split-1:
            gpu += 1
    return device_map   


def get_llama2_lm_info(lm):
    info = {
        'model_dim':lm.config.hidden_size,
        'memory_size':lm.config.intermediate_size,
        "num_layers":lm.config.num_hidden_layers
    }
    return info 
     
def get_llama2_block(lm, layer_index):
    return lm.model.layers[layer_index]           