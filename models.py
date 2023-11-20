import torch
from utils import init_past_key_values
from typing import Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

class T5Encoder(torch.nn.Module):
    def __init__(self, pretrain_model = None):
        super(T5Encoder, self).__init__()
        self.encoder = pretrain_model.get_encoder()

    def forward(self,
                input_ids : torch.LongTensor,
                attention_mask: torch.LongTensor,
                ):

        output = self.encoder(input_ids = input_ids,
                              attention_mask = attention_mask,
                              return_dict = True)

        return output[0]





class T5Decoder(torch.nn.Module):
    def __init__(self, pretrain_model = None):
        super(T5Decoder, self).__init__()
        self.decoder = pretrain_model.get_decoder()
        self.lm_head = pretrain_model.lm_head

    def forward(self,
                input_ids : torch.LongTensor,
                attention_mask : torch.LongTensor,
                encoder_hidden_states : torch.FloatTensor,
                ):

        output = self.decoder(input_ids = input_ids,
                              attention_mask = attention_mask,
                              encoder_hidden_states = encoder_hidden_states,
                              use_cache = True,
                              return_dict = True)

        logits = self.lm_head(output[0])
        tuple_outputs = (logits, output.past_key_values)
        return tuple_outputs





class CacheT5Decoder(torch.nn.Module):
    def __init__(self, pretrain_model = None):
        super(CacheT5Decoder, self).__init__()
        self.decoder = pretrain_model.get_decoder()
        self.lm_head = pretrain_model.lm_head

    def forward(self,
                input_ids : torch.LongTensor,
                attention_mask : torch.LongTensor,
                past_key_values : Tuple[Tuple[torch.FloatTensor]],
                encoder_hidden_states : torch.FloatTensor,
                ):

        output = self.decoder(input_ids = input_ids,
                              attention_mask = attention_mask,
                              past_key_values = past_key_values,
                              encoder_hidden_states = encoder_hidden_states,
                              use_cache = True,
                              return_dict = True)

        logits = self.lm_head(output[0])
        tuple_outputs = (logits, output.past_key_values)
        return tuple_outputs





class T5Seq2Seq(torch.nn.Module):
    def __init__(self, pretrain_path = "VietAI/vit5-base", prompt_length = 256, enocder_num_block = 12):
        super(T5Seq2Seq, self).__init__()

        self.pad_token_id = 0
        self.eos_token_id = 1
        self.prompt_length = prompt_length
        pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_path)

        # init dumpy input for tracing
        dummy_en_ids = torch.ones([1, self.prompt_length], dtype = torch.long)
        dummy_de_ids = torch.ones([1, 1], dtype = torch.long)
        dummy_hidden_states = torch.ones([1, self.prompt_length, 768], dtype = torch.float) * 0.1
        dummy_past_key_values = init_past_key_values(encoder_ids_len = self.prompt_length,
                                                     decoder_ids_len = 1, encoder_num_blocks = enocder_num_block)

        # define encoder & decoder
        self.encoder = torch.jit.trace(T5Encoder(pretrain_model),
                                       (dummy_en_ids, dummy_en_ids))
        self.decoder = torch.jit.trace(T5Decoder(pretrain_model),
                                       (dummy_de_ids, dummy_de_ids, dummy_hidden_states))
        self.cache_decoder = torch.jit.trace(CacheT5Decoder(pretrain_model),
                                             (dummy_de_ids, dummy_de_ids, dummy_past_key_values, dummy_hidden_states))


    def forward(self, encoder_input_ids, encoder_attention_mask, max_length):
        batch_size = encoder_input_ids.shape[0]
        # compute encoder hidden state
        hidden_states = self.encoder(encoder_input_ids, encoder_attention_mask)

        # prepare input_ids & attention_mask for decoder's start step
        decoder_input_ids = torch.ones([batch_size, 1], dtype = torch.long) * pad_token_id
        decoder_attention_mask = torch.ones([batch_size, 1], dtype = torch.long)

        # generate first token (exclude start token ~ pad token)
        decoder_outputs = self.decoder(decoder_input_ids, decoder_attention_mask, hidden_states)
        next_token_logits = decoder_outputs[0][:, -1, :]
        past_key_value = decoder_outputs[1]

        next_tokens = torch.argmax(next_token_logits, dim = -1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim = -1)

        # looping to generate
        eos_token_id = [self.eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id, dtype = torch.long)
        pad_token_id = self.pad_token_id
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long)
        this_peer_finished = False

        while True:
            # get last ids for inputs
            de_cur_in_ids = decoder_input_ids[:, -1:]

            # foward current ids to decoder
            decoder_outputs = self.cache_decoder(de_cur_in_ids, decoder_attention_mask, past_key_value, hidden_states)

            # predict new tokens
            next_token_logits = decoder_outputs[0][:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim = -1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update past_key_value & decoder_input_ids
            past_key_value = decoder_outputs[1]
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim = -1)


            # check stopping criteria
            ## if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim = 0)
            )
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

            ## if batch of sentences exceeds a specified max_length
            if decoder_input_ids.shape[-1] >= int(max_length):
                this_peer_finished = True

            if this_peer_finished:
                break

        return decoder_input_ids

