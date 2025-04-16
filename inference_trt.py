import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from time import time
import argparse

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = np.arange(kernel_size)
    x_grid = np.repeat(x_coord, kernel_size).reshape(kernel_size, kernel_size)
    y_grid = x_grid.T
    xy_grid = np.stack([x_grid, y_grid], axis=-1).astype(np.float32)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * np.pi * variance)) * np.exp(-np.sum((xy_grid - mean) ** 2., axis=-1) / (2 * variance))

    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.reshape(kernel_size, kernel_size)

    return gaussian_kernel

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    x1_norm = np.linalg.norm(x1, axis=dim, keepdims=True).clip(min=eps)
    x2_norm = np.linalg.norm(x2, axis=dim, keepdims=True).clip(min=eps)

    dot_product = np.sum(x1 * x2, axis=dim, keepdims=True)
    similarity = dot_product / (x1_norm * x2_norm)
    similarity = (np.round(1 - similarity, decimals=4))

    return np.squeeze(similarity, axis=dim)


def resize_with_align_corners(image, out_size):
    in_height, in_width = image.shape[-2:]
    out_height, out_width = out_size

    x_indices = np.linspace(0, in_width - 1, out_width).astype(np.float32)
    y_indices = np.linspace(0, in_height - 1, out_height).astype(np.float32)
    map_x, map_y = np.meshgrid(x_indices, y_indices)

    resized_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return resized_image

def resize_without_align_corners(image, out_size):
    batch_size, channels, _, _ = image.shape
    out_height, out_width = out_size

    resized_images = np.zeros((batch_size, channels, out_height, out_width), dtype=image.dtype)

    for b in range(batch_size):
        for c in range(channels):
            resized_images[b, c] = cv2.resize(image[b, c], (out_width, out_height), interpolation=cv2.INTER_LINEAR)

    return resized_images

def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []

    for idx, i in enumerate(range(len(ft_list))):
        fs = fs_list[i]
        ft = ft_list[i]

        a_map = cosine_similarity(fs, ft)
        a_map = np.squeeze(a_map)
        a_map = resize_with_align_corners(a_map, out_size)
        a_map = np.expand_dims(a_map, axis=0)
        a_map = np.expand_dims(a_map, axis=0)
        a_map_list.append(a_map)

    anomaly_map = np.round(np.mean(np.concatenate(a_map_list, axis=1), axis=1, keepdims=True), decimals=4)

    return anomaly_map, a_map_list

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cudart.cudaStreamCreate()[1]

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(tensor_name)
            shape = self.engine.get_tensor_shape(tensor_name)
            
            host_mem = np.zeros(shape, dtype=trt.nptype(dtype))
            device_mem = cudart.cudaMalloc(host_mem.nbytes)[1]
            
            self.bindings.append(device_mem)
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
            else:
                self.outputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
        
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), 
                self.bindings[i]
            )

    def infer(self, input_data):
        if not isinstance(self.inputs[0]['host'], np.ndarray):
            raise TypeError("Host memory must be numpy.ndarray")
            
        np.copyto(self.inputs[0]['host'], input_data.reshape(self.inputs[0]['host'].shape))
        
        cudart.cudaMemcpyAsync(
            self.inputs[0]['device'],
            self.inputs[0]['host'].ctypes.data,
            self.inputs[0]['host'].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream
        )
        
        self.context.execute_async_v3(self.stream)
        
        outputs = {}
        for out in self.outputs:
            cudart.cudaMemcpyAsync(
                out['host'].ctypes.data,
                out['device'],
                out['host'].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream
            )
            outputs[out['name']] = out['host'].copy()
        
        cudart.cudaStreamSynchronize(self.stream)
        return outputs


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

    return modified_image

def visualize(output_folder_path, image_path, anomaly_map_image):
    origin_image = cv2.imread(image_path)
    # origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
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

def main_process(image_folder_path, output_folder_path, trt_engine_path, input_size=392, max_ratio=0.01, visualize_output=True):
    os.makedirs(output_folder_path, exist_ok=True)
    trt_model = TRTInference(trt_engine_path)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4)

    for idx, file in enumerate(os.listdir(image_folder_path)):
        start_time = time()
        image_path = os.path.join(image_folder_path, file)
        input_image = pre_process(image_path, input_size)
        
        outputs = trt_model.infer(input_image)

        en = [outputs['2527'], outputs['2529']]
        de = [outputs['2531'], outputs['2533']]
        
        anomaly_map, _ = cal_anomaly_maps(en, de, input_size)
        anomaly_map = resize_without_align_corners(anomaly_map, (256, 256))[0, 0, :, :]
        anomaly_map = cv2.filter2D(anomaly_map, -1, gaussian_kernel)

        if visualize_output:
            visualize(output_folder_path, image_path, anomaly_map)
            
        print(f"{idx:05d} | {(time()-start_time)*1000:.2f} ms | Image: {os.path.splitext(file)[0]}, Score: {np.max(anomaly_map):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder_path', type=str, required=True)
    parser.add_argument('--output_folder_path', type=str, required=True)
    parser.add_argument('--trt_model_path', type=str, required=True)
    parser.add_argument('--input_size', type=int, default=392)
    parser.add_argument('--max_ratio', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--visualize_output', action='store_true')
    args = parser.parse_args()

    main_process(
        args.image_folder_path,
        args.output_folder_path,
        args.trt_model_path,
        args.input_size,
        args.max_ratio,
        args.visualize_output
    )