import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import argparse

class TRTEngineConverter:
    def __init__(self):
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self._verify_environment()

    def _verify_environment(self):
        device = cuda.Device(0)
        cc = device.compute_capability()
        print(f"Detected GPU Compute Capability: {cc[0]}.{cc[1]}")

    def _create_config(self, builder):
        config = builder.create_builder_config()
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        return config

    def parse_and_build(self, onnx_path, output_path):
        with trt.Builder(self.logger) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, self.logger) as parser:

            # 1. Create config
            config = self._create_config(builder)

            # 2. Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    error_str = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
                    raise RuntimeError(f"Failed to parse ONNX model:\n{error_str}")

            # 3. Check input layer
            input_tensor = network.get_input(0)
            if input_tensor.name != 'input':
                raise ValueError(f"Input tensor name mismatch: {input_tensor.name}")

            # 4. Build engine
            engine = builder.build_serialized_network(network, config)
            if not engine:
                raise RuntimeError("Failed to build TensorRT engine.")

            # 5. Save engine
            with open(output_path, 'wb') as f:
                f.write(engine)
            print(f"Successfully converted and saved: {output_path}")

def main(onnx_model_path, trt_model_path):
    converter = TRTEngineConverter()
    converter.parse_and_build(onnx_model_path, trt_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export ONNX model to TensorRT engine')

    parser.add_argument('--onnx_model_path', type=str, default="/INP-Former/model.onnx")
    parser.add_argument('--trt_model_path', type=str, default="/INP-Former/model.trt")

    args = parser.parse_args()
    main(args.onnx_model_path, args.trt_model_path)