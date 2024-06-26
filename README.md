# Memorizing Documents with Guidance in Large Language Models

* 📚 Official Repo for IJCAI 2024 - Memorizing Documents with Guidance in Large Language Models 
* 🖋️ Project Page [[github.io](https://fxnnxc.github.io/main_papers/2024_guidace_loss_for_documents/)]
* 📜 Paper [[pdf](https://arxiv.org/abs/2406.15996)]

<p align="center" >
<img src="/assets/document-wise-memory.jpg" width="100%">
</p> 

## Visualizing Document Selection 

Please see `utils.py` the implementation of memory selection. 

<img src="/assets/output_relu.gif" width="100%">
<img src="/assets/output_tanh.gif" width="100%">


## Run Experiment 


```bash
pip install -e .
```


```
# --- LLM Related ---
lm_name=pythia 
lm_size=1b  
num_gpus=1
lm_cache_dir=none  # see foundations/llama2.py

# --- Data Related ---
data_cache_dir=none  # see wikitext.py
max_labels=10        # number of labels 
segment_length=128   # number of words for segmentation of long document 
max_segements=10     # how many segments
max_length=256       # maximum number of tokens 

# --- Document Memory Related  ---  
key_dim=2           # dimension of random document representation
key_activation=tanh # inductive bias of document memory selection  
hook_memory_dim=32  # how many memories 
hook_memory_layer=15   # location of the memory  
lm_act=gelu            # activation fn of LLM 
negative_key_type=other         # negative key  (random, other, zero)
document_key_module_type=linear # linear (DocRep to memory selection)
target_negative_ce=4.5      # tau 
guidance=0.1                # alpha 

# --- Optimization Related ---
batch_size=16
lr=1e-3
num_accumulation_steps=1
grad_clip_coeff=5
num_epochs=500
save_freq=10

seed=1
save_dir=outputs/$lm_name/$lm_size
python train_guidance.py \
        --lm_name $lm_name \
        --lm_size $lm_size \
        --lm_cache_dir $lm_cache_dir \
        --num_gpus $num_gpus \
        --max_labels $max_labels \
        --segment_length $segment_length \
        --max_segements $max_segements \
        --max_length $max_length \
        --data_cache_dir $data_cache_dir \
        --lr $lr \
        --batch_size $batch_size \
        --num_epochs $num_epochs \
        --save_freq $save_freq \
        --num_accumulation_steps $num_accumulation_steps \
        --grad_clip_coeff $grad_clip_coeff \
        --seed $seed \
        --save_dir $save_dir \
        --hook_memory_dim $hook_memory_dim \
        --hook_memory_layer $hook_memory_layer \
        --lm_act $lm_act \
        --key_dim $key_dim \
        --key_activation $key_activation \
        --negative_key_type $negative_key_type \
        --document_key_module_type $document_key_module_type \
        --target_negative_ce $target_negative_ce \
        --guidance $guidance 
```


