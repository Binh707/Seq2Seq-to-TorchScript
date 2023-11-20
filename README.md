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
To export to torchscript:
```
!python export_torchscript.py --pretrain_path "VietAI/vit5-base" --prompt_length 256 --encoder_num_blocks 12
```

To export to onnx:
```
!python export_onnx.py --pretrain_path "VietAI/vit5-base" --prompt_length 256 --encoder_num_blocks 12
```
