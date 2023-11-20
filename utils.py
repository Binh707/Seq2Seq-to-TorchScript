import torch
import numpy as np

def inputs_to_test_pipeline(tokenizer = None, num_of_sample = 1, max_length = 256):
    text_inputs = ["tôi học toán lý hóa văn sử địa" for i in range(num_of_sample)]
    tokenized_batch = tokenizer(text_inputs, max_length = max_length, padding = 'max_length',
                                truncation = True, return_tensors = 'np')
    return tokenized_batch.input_ids, tokenized_batch.attention_mask

def inputs_to_test_accuracy(tokenizer = None, text_inputs = None, max_length = 256):
    tokenized_batch = tokenizer(text_inputs, max_length=max_length, padding='max_length',
                                truncation=True, return_tensors='np')
    return tokenized_batch.input_ids, tokenized_batch.attention_mask


def init_past_key_values(encoder_ids_len = 32, decoder_ids_len = 1, encoder_num_blocks = 12):
    ten1 = torch.ones([1, 12, decoder_ids_len, 64], dtype = torch.float) * 0.1
    ten2 = torch.ones([1, 12, decoder_ids_len, 64], dtype = torch.float) * 0.1
    ten3 = torch.ones([1, 12, encoder_ids_len, 64], dtype = torch.float) * 0.1
    ten4 = torch.ones([1, 12, encoder_ids_len, 64], dtype = torch.float) * 0.1

    per_layer = (ten1, ten2, ten3, ten4)
    past_key_values = tuple([per_layer for i in range(encoder_num_blocks)])
    return past_key_values