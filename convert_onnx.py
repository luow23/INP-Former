import torch
import torch.nn as nn
import os
from functools import partial
import argparse
from utils import setup_seed

import onnx
import onnxsim
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

def set_model(args):
    setup_seed(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
        [nn.Parameter(torch.randn(args.INP_num, embed_dim))
         for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                       target_layers=target_layers, remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                       fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    # model = model

    model.load_state_dict(torch.load(args.pretrained_model_path))
    model.eval()

    return model


def export_onnx(args, torch_model):
    _, extension = os.path.splitext(args.pretrained_model_path)
    export_onnx_path = args.pretrained_model_path.replace(extension, ".onnx")
    dummy_input = torch.randn(*args.input_size)

    torch.onnx.export(torch_model, dummy_input, export_onnx_path, export_params=True, opset_version=13,
                      do_constant_folding=True, input_names=['input'])

    simplified_model, _ = onnxsim.simplify(export_onnx_path)
    onnx.save(simplified_model, export_onnx_path)

    print(f"Model has been successfully converted to ONNX and saved at {export_onnx_path}")


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--encoder', type=str,
                        default='dinov2reg_vit_base_14')  # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=tuple, default=(1, 3, 392, 392))
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--pretrained_model_path', type=str, default="INP-Former/model.pth")

    args = parser.parse_args()

    torch_model = set_model(args)
    export_onnx(args, torch_model)
