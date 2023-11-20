import argparse
import torch
from models import T5Seq2Seq

def args_parser():
    parser = argparse.ArgumentParser(description="Training ...")
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--encoder_num_blocks", required=True, type=int)
    parser.add_argument("--prompt_length", required=True, type=int)
    parser.add_argument("--ouput_path", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    T5_model = T5Seq2Seq(pretrain_path = args.checkpoint_path,
                         prompt_length = args.prompt_length,
                         encoder_num_blocks = args.encoder_num_blocks)
    scripted_model = torch.jit.script(T5_model)
    scripted_model.save(args.output_path)