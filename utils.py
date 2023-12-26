import torch
import numpy as np


def init_past_key_values(encoder_ids_len=32, encoder_num_blocks=12,
                         num_heads=12, embed_size_per_head=64, device='cpu'):
    batch_size = 1
    decoder_ids_len = 1
    ten1 = (torch.ones([batch_size, num_heads, decoder_ids_len, embed_size_per_head], dtype=torch.float) * 0.1).to(device)
    ten2 = (torch.ones([batch_size, num_heads, decoder_ids_len, embed_size_per_head], dtype=torch.float) * 0.1).to(device)
    ten3 = (torch.ones([batch_size, num_heads, encoder_ids_len, embed_size_per_head], dtype=torch.float) * 0.1).to(device)
    ten4 = (torch.ones([batch_size, num_heads, encoder_ids_len, embed_size_per_head], dtype=torch.float) * 0.1).to(device)

    per_layer = (ten1, ten2, ten3, ten4)
    past_key_values = tuple([per_layer for i in range(encoder_num_blocks)])
    return past_key_values
