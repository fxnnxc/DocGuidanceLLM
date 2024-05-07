# Training of Documentwise-Memories 
# Author : Bumjin Park
# -------------------------------------------
# 🍊 General Training of Document-Wise Memories 
# -------------------------------------------

# --- LLM Related ---
lm_name=pythia 
lm_size=1b  
num_gpus=1
lm_cache_dir=none

# --- Data Related ---
data_cache_dir=none
max_labels=10
max_segements=10
segment_length=128
max_length=256

# --- Document Memory Related  ---
key_dim=2
key_activation=tanh # tanh
hook_memory_dim=32
hook_memory_layer=15
lm_act=gelu
negative_key_type=other
document_key_module_type=linear # linear nonlinear
target_negative_ce=4.5
guidance=0.1

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