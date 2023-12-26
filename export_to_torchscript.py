import argparse
import torch
from T5_modules  import T5Seq2Seq

def args_parser():
    parser = argparse.ArgumentParser(description="Exporting ...")
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--model_type", required=True, default='T5', type=str)
    parser.add_argument("--encoder_num_blocks", required=True, type=int)
    parser.add_argument("--num_heads", required=True, type=int)
    parser.add_argument("--embed_size_per_head", required=True, type=int)
    parser.add_argument("--embedding_size", required=True, type=int)
    parser.add_argument("--device", required=True, default='cpu', type=str)
    parser.add_argument("--output_path", required=True, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    if args.model_type == 'T5':
        model = T5Seq2Seq(pretrain_path=args.checkpoint_path,
                          encoder_num_blocks=args.encoder_num_blocks,
                          num_heads=args.num_heads,
                          embed_size_per_head=args.embed_size_per_head,
                          embeddings_size=args.embedding_size,
                          device=args.device)
        scripted_model = torch.jit.script(model)
        scripted_model.save(args.output_path)
    else:
        print('Error: ' + args.model_type + ' not supported !')
