


def update_llm_guidance_parser(parser):
    parser.add_argument("--guidance", type=float, help="guidance scale for the negative part")
    parser.add_argument("--negative_key_type", help='negative key type')
    parser.add_argument("--key_dim", help='conversion of string to key_dim vector', type=int)
    parser.add_argument("--key_activation", help='the final activation of the key generation')
    parser.add_argument("--target_negative_ce", default=2.5, help='forgetting for netative', type=float)
    
    parser.add_argument("--document_key_module_type", choices=['linear', 'nonlinear'], help="the choice of linear or non-linear key generation")
    parser.add_argument("--lm_act", help='internal activation of the document key memory')
    # for nonlinear generation
    parser.add_argument("--key_internal_dim", type=int)
    parser.add_argument("--key_internal_activation", type=str)
    parser.add_argument("--key_internal_layers", type=int)
    return parser

import torch 
import torch.nn as nn 
import torch.nn.functional as F  
from transformers.activations import ACT2FN

def layer_init(layer, std=2**(1/2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, 'bias') and layer.bias is not None :
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_mlp(n_entries, num_layers,  act, hidden_dim, num_outputs, **kwargs):
    if act == "relu":
        act = nn.ReLU
    elif act == "tanh":
        act = nn.Tanh
    elif act == "sigmoid":
        act = nn.Sigmoid
    elif act == 'gelu':
        act = nn.GELU        
    if num_layers ==1 :
        net = layer_init(nn.Linear(n_entries, num_outputs))
    else: 
        net = [layer_init(nn.Linear(n_entries, hidden_dim)), act()]
        for i in range(num_layers-1):
            net.append(layer_init(nn.Linear(hidden_dim, hidden_dim)))
            net.append(act())
        net.append(layer_init(nn.Linear(hidden_dim, num_outputs)))
        net = nn.Sequential(*net)

    return net

def make_activation(activation):
    if activation == "swiglu":
        return SwiGLU()
    else:
        return ACT2FN[activation] 


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x



class DocumentKeyMemoryModule(nn.Module):
    def __init__(self, key_names, key_dim):
        super().__init__()
        self.key_names = key_names
        self.current_key = None 
        key_weights = layer_init(nn.Linear(key_dim, len(self.key_names), bias=False)).weight
        self.keys = nn.ParameterDict({
            key: nn.Parameter(key_weights[k,:], requires_grad=False) for k, key in enumerate(self.key_names)
        })
    
    def set_key(self, key_index=None, custom_key=None):
        if custom_key is not None:
            assert custom_key.shape == next(self.keys.values()).shape
            self.current_key = custom_key
        else:
            assert key_index is not None
            self.current_key = self.keys[key_index].data
    
    def remove_key(self):
        self.current_key = None 

    def generate_random_key(self, std=1.0):
        key1 = next(self.keys.values())
        random_key = torch.randn_like(key1) * std
        return random_key
    
    def get_key(self, name):
        return torch.clone(self.keys[name].data)
        

class LinearDocumentKeyMemory(DocumentKeyMemoryModule):
    def __init__(self, model_dim, intermediate_size, lm_act, keys, key_dim, key_activation, ):
        super().__init__(keys, key_dim)
        self.act_fn = make_activation(lm_act)
        self.up_proj = layer_init(nn.Linear(model_dim, intermediate_size, bias=True))
        self.gate_proj = layer_init(nn.Linear(key_dim, intermediate_size if key_activation != "swiglu" else intermediate_size*2, bias=True))
        self.down_proj = layer_init(nn.Linear(intermediate_size, model_dim, bias=True))
        self.linear_for_lmm = layer_init(nn.Linear(model_dim, model_dim, bias=True))        
        self.key_activation = make_activation(key_activation)        
        self.ln = nn.LayerNorm(model_dim)
        
    def compute_key(self, current_key):
        key = self.key_activation(self.gate_proj(current_key))
        return key 
            
    def forward(self, x, **kwargs):
        hidden = x 
        x = self.act_fn(self.up_proj(x))
        if self.current_key is not None:
            x = x * self.compute_key(self.current_key)
        x = self.down_proj(x) 
        x = self.linear_for_lmm(x)
        x = self.act_fn(x)
        x = hidden + x 
        # x = self.ln(x)
        return x

class NonLinearDocumentKeyMemory(DocumentKeyMemoryModule):
    def __init__(self, model_dim, intermediate_size, lm_act, keys, key_dim, key_activation, 
                 key_internal_dim, key_internal_activation, key_internal_layers):
        super().__init__(keys, key_dim)
        
        self.act_fn = make_activation(lm_act)
        self.up_proj = layer_init(nn.Linear(model_dim, intermediate_size, bias=True))
        self.gate_proj = layer_init(nn.Linear(key_dim, intermediate_size if key_activation != "swiglu" else intermediate_size*2, bias=True))
        self.key_module = make_mlp(key_dim, key_internal_layers, key_internal_activation, key_internal_dim, 
                                   intermediate_size if key_activation != "swiglu" else intermediate_size*2)
        self.down_proj = layer_init(nn.Linear(intermediate_size, model_dim, bias=True))
        self.linear_for_lmm = layer_init(nn.Linear(model_dim, model_dim, bias=True))
        self.key_activation = make_activation(key_activation)
        self.current_key = None 
        key_weights = layer_init(nn.Linear(key_dim, len(keys), bias=False)).weight
        self.keys = nn.ParameterDict({
            key: nn.Parameter(key_weights[k,:], requires_grad=False) for k, key in enumerate(keys)
        })
        self.ln = nn.LayerNorm(model_dim)
    
    def compute_key(self, current_key):
        key = self.key_activation(self.key_module(current_key))
        return key         
    
    def forward(self, x, **kwargs):
        hidden = x
        x = self.act_fn(self.up_proj(x))
        if self.current_key is not None:
            x = x * self.compute_key(self.current_key)
        x = self.down_proj(x) 
        x = self.linear_for_lmm(x)
        x = self.act_fn(x)
        x = hidden + x 
        # x = self.ln(x)
        return x
    
