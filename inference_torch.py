import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from time import time
from functools import partial
import argparse
import warnings

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import setup_seed, get_gaussian_kernel, cal_anomaly_maps

warnings.filterwarnings("ignore")

def set_model(encoder_name=None, torch_model_path=None, INP_num=6):
    setup_seed(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoder = vit_encoder.load(encoder_name)
    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(INP_num, embed_dim))
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
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model = model.to(device)

    model.load_state_dict(torch.load(torch_model_path))
    model.eval()

    return model

def pre_process(image_path, input_size):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    modified_image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    modified_image = modified_image.astype(np.float32) / 255.0
    modified_image = (modified_image - mean) / std
    modified_image = np.transpose(modified_image, (2, 0, 1))
    modified_image = np.expand_dims(modified_image, axis=0).astype(np.float32)
    
    return torch.as_tensor(modified_image, dtype=torch.float32)

def visualize(output_folder_path, image_path, anomaly_map_image):
    origin_image = cv2.imread(image_path)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    origin_height, origin_width = origin_image.shape[:2]
    
    heat_map = min_max_norm(anomaly_map_image)
    heat_map_resized = cv2.resize(heat_map, (origin_width, origin_height))
    heat_map_image = cvt2heatmap(heat_map_resized * 255)

    overlay = cv2.addWeighted(origin_image, 0.6, heat_map_image, 0.4, 0)

    overlay_save_path = os.path.join(output_folder_path, f"overlay_{os.path.basename(image_path)}")
    cv2.imwrite(overlay_save_path, overlay)

    heat_map_save_path = os.path.join(output_folder_path, f"heatmap_{os.path.basename(image_path)}")
    cv2.imwrite(heat_map_save_path, heat_map_image)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def cvt2heatmap(gray):
    heat_map = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heat_map

def main_process(image_folder_path, output_folder_path, torch_model_path, encoder_name, INP_num, input_size, max_ratio, visualize_output, device):
    
    os.makedirs(output_folder_path, exist_ok=True)
    
    all_files = os.listdir(image_folder_path)
    
    torch_model = set_model(encoder_name=encoder_name, torch_model_path=torch_model_path, INP_num=INP_num)

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    for idx, file in enumerate(all_files):
        start_time = time()
        image_path = os.path.join(image_folder_path, file)
        base_name = file.split(".")[0]
        input_image = pre_process(image_path, input_size).to(device)

        output = torch_model(input_image)
        en = output[0]
        de = output[1]
        
        anomaly_map, _ = cal_anomaly_maps(en, de, input_size)
        anomaly_map = F.interpolate(anomaly_map, size=256, mode='bilinear', align_corners=False)
        anomaly_map = gaussian_kernel(anomaly_map)
        anomaly_map_image = anomaly_map.squeeze().cpu().detach().numpy()

        if max_ratio == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
        else:
            anomaly_map = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
            sp_score = sp_score.mean(dim=1)

        if visualize_output:
            visualize(output_folder_path, image_path, anomaly_map_image)
        
        end_time = time()
        elapsed_time = (end_time - start_time) * 1000
        
        print(f"{idx:05d} | {elapsed_time} ms | Image: {base_name}, Anomaly Score: {sp_score.item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Torch Inference for Anomaly Detection')
    
    parser.add_argument('--image_folder_path', type=str, required=True, help='Path to input image folder')
    parser.add_argument('--output_folder_path', type=str, required=True, help='Path to save visualized outputs')
    parser.add_argument('--torch_model_path', type=str, required=True, help='Path to the PyTorch model file')
    parser.add_argument('--encoder_name', type=str, default='dinov2reg_vit_base_14', help='Encoder model name')
    parser.add_argument('--INP_num', type=int, default=6, help='Number of INP tokens')
    parser.add_argument('--input_size', type=int, default=392, help='Input size for model inference')
    parser.add_argument('--max_ratio', type=float, default=0.01, help='Max ratio used for score thresholding')
    parser.add_argument('--visualize_output', action='store_true', help='Flag to visualize the results')

    args = parser.parse_args()

    main_process(
        image_folder_path=args.image_folder_path,
        output_folder_path=args.output_folder_path,
        torch_model_path=args.torch_model_path,
        encoder_name=args.encoder_name,
        INP_num=args.INP_num,
        input_size=args.input_size,
        max_ratio=args.max_ratio,
        visualize_output=args.visualize_output
    )