
def update_hook_memory_pareser(parser):
    parser.add_argument("--hook_memory_layer", type=int, default=None)
    parser.add_argument("--hook_memory_dim",   type=int, default=None)
    parser.add_argument("--memory_module",   type=str, default='base')
    return parser 

# ---------------------- 
# 🥕 Hook Memory 🥕

import torch 
class HookMemoryAdaptedLLM():
    def __init__(self, lm):
        self.lm = lm 
        self.hook = None 
        self.memory_module = None 
        self.hooked_module = None 
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, adapt_memory=True, **kwargs):
        if adapt_memory:
            self.register_hook()
            outputs = self.lm(**kwargs)
            self.remove_hook()
        else:
            outputs = self.lm(**kwargs)
        return outputs     
    
    # --- 🥕 Hook Related
    def register_hook(self):
        if self.hook is not None:
            print("nothing happned because hook already exists")
            return 
        def hook_fn(module, input, output):
            try:
                hidden = output[0]
                new_hidden = module.saved_memory_module(hidden)
                hidden = new_hidden # + hidden  # residual conntextion
                output = (hidden,)+ output[1:]
            except Exception as e:
                print(e)
            return output
        
        self.hook = self.hooked_module.register_forward_hook(hook_fn)
    
    def remove_hook(self):
        if self.hook is None:
            print("nothing happend because hook does not exist")
            return 
        self.hook.remove()
        self.hook = None
        
    # --- 🥕 Memory Module Handling
    def set_memory_module(self, memory_module):
        self.memory_module = memory_module 
        self.hooked_module.saved_memory_module = self.memory_module
        device = next(self.hooked_module.parameters()).device
        self.memory_module.to(device)
        
    def drop_memory_module(self):
        if not hasattr(self, "memory_module"):
            print("no memory module is dropped")
            return 
        del self.hooked_module.saved_memory_module
        del self.memory_module

    def set_hooked_module(self, module):
        self.hooked_module = module 
        
    def save(self, path, save_state_dict=False):
        torch.save(self.memory_module, path)
        
    def load(self, path, load_state_dict=False):
        memory_module = torch.load(path)
        self.set_memory_module(memory_module)
        
    def zero_grad(self):
        self.lm.zero_grad()
        self.memory_module.zero_grad()
    
    def train(self):
        self.lm.train()
        self.memory_module.train()
        
    def eval(self):
        self.lm.eval()
        self.memory_module.eval()
        
    def generate(self, *args, **kwargs):
        return self.lm.generate(*args, **kwargs)
        