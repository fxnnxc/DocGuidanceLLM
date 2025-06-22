# 🧠 Document Memory with Guidance in Large Language Models

<div align="center">

[![IJCAI 2024](https://img.shields.io/badge/IJCAI-2024-blue?style=for-the-badge&logo=academia)](https://ijcai-24.org/)
[![Project](https://img.shields.io/badge/Project-Document%20Memory%20Guidance-darkblue?style=for-the-badge&logo=github)](https://fxnnxc.github.io/main_papers/2024_guidace_loss_for_documents/)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-darkgreen?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2406.15996)


**Memorizing Documents with Guidance in Large Language Models**

| [**🌐 Project Page**](https://bumjini.github.io/articles/2024_ijcai_document_memory//) | [**📄 Paper**](https://arxiv.org/abs/2406.15996) |

</div>

---

## 🎯 Abstract

This paper introduces a novel approach to enhance document memory in large language models (LLMs) through guided learning mechanisms. We propose a **document-wise memory selection** framework that enables models to selectively memorize and retrieve document-specific information using learnable document representations and guidance loss functions.

### 🔑 Key Contributions
- **Framework**: Novel document-wise memory selection mechanism
- **Guidance Loss**: Innovative guidance-based training approach for document memory
- **Visualization**: Comprehensive analysis of memory selection patterns
- **Evaluation**: Extensive experiments across multiple model architectures

---

## 🔍 TL;DR  
Large language models can be enhanced with **document-wise memory selection** using learnable document representations and **guidance loss functions** to improve document memorization and retrieval capabilities.

<p align="center" >
<img src="/assets/document-wise-memory.png" width="100%">
</p> 

---

## 📁 Repository Structure
```
DocGuidanceLLM/
├── foundations/                # Model foundation implementations
│   ├── llama2.py              # Llama2 model utilities
│   └── pythia.py              # Pythia model utilities
├── document_memories.py        # Document memory implementation
├── hook_lm.py                 # Language model hooking utilities
├── train_guidance.py          # Main training script
├── utils.py                   # Utility functions and memory selection
├── wikitext.py                # WikiText dataset processing
├── run.sh                     # Experiment runner script
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Supported Models
The following models are supported in `foundations/`:
- **Llama2**: Various sizes through `llama2.py`
- **Pythia**: Various sizes through `pythia.py`

### Running Experiments

#### 1. Training with Document Memory Guidance
```bash
# Run the main training experiment
bash run.sh

# Or run with custom parameters
python train_guidance.py \
    --lm_name pythia \
    --lm_size 1b \
    --num_gpus 1 \
    --max_labels 10 \
    --segment_length 128 \
    --max_segements 10 \
    --max_length 256 \
    --lr 1e-3 \
    --batch_size 16 \
    --num_epochs 500 \
    --hook_memory_dim 32 \
    --hook_memory_layer 15 \
    --key_dim 2 \
    --key_activation tanh \
    --guidance 0.1
```

#### 2. Experiment Configuration
Edit the parameters in `run.sh` to customize your experiments:

```bash
# --- LLM Related ---
lm_name=pythia 
lm_size=1b  
num_gpus=1

# --- Document Memory Related  ---  
key_dim=2           # dimension of random document representation
key_activation=tanh # inductive bias of document memory selection  
hook_memory_dim=32  # how many memories 
hook_memory_layer=15   # location of the memory  
guidance=0.1        # alpha (guidance parameter)
```

---

## 📊 Visualizing Document Selection 

Please see `utils.py` for the implementation of memory selection. 

**Visualization of ReLU Activation**
<img src="/assets/output_relu.gif" width="100%">

**Visualization of Tanh Activation**
<img src="/assets/output_tanh.gif" width="100%">

---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{park2024document,
  title={Memorizing Documents with Guidance in Large Language Models},
  author={Park, Bumjin and Choi, Jaesik},
  booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2024}
}
```

## 📋 Requirements

Key dependencies include:
- PyTorch
- Transformers

