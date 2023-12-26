# T5 MODEL EXPORTER

This repository contains scripts to export T5 model to torchscript (cache mode) or onnx (non cache mode).

# How to use?
## Step 1: Setup
```
!git clone https://github.com/Binh707/Export-T5-to-TorchScript-ONNX.git
```
```
!pip install -r requirements.txt
```

## Step 2: Run scripts
To export T5 to torchscript:
```
!python export_to_torchscript.py \
--checkpoint_path 'VietAI/vit5-base' \
--model_type 'T5' \
--encoder_num_blocks 12 \
--num_heads 12 \
--embed_size_per_head 64 \
--embedding_size 768 \
--device 'cpu' \
--output_path './t5.pt'
```


To export SeamlessM4T to torchscript:
```
!python export_to_torchscript.py \
--checkpoint_path 'facebook/hf-seamless-m4t-medium' \
--model_type 'SeamlessM4T' \
--encoder_num_blocks 12 \
--num_heads 16 \
--embed_size_per_head 64 \
--embedding_size 1024 \
--device 'cpu' \
--output_path './seamlessm4t.pt'
```
