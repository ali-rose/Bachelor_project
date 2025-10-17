import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_trt_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def trt_dtype_to_np(dtype):
    """Convert TensorRT data type to numpy data type."""
    if dtype == trt.DataType.FLOAT:
        return np.float32
    if dtype == trt.DataType.HALF:
        return np.float16
    if dtype == trt.DataType.INT8:
        return np.int8
    if dtype == trt.DataType.INT32:
        return np.int32
    # Add more mappings if necessary
    return np.float32  # Default case

# Load the TensorRT engine
engine_path = '/root/wav2lip/Wav2Lip/wav2lip_dynaminc.engine'  # Adjust the path as necessary
engine = load_trt_engine(engine_path)

# Print engine binding information
print("Engine has {} bindings.".format(engine.num_bindings))
for binding_index in range(engine.num_bindings):
    binding_name = engine.get_binding_name(binding_index) if hasattr(engine, 'get_binding_name') else engine.get_tensor_name(binding_index)
    binding_shape = engine.get_binding_shape(binding_index) if hasattr(engine, 'get_binding_shape') else engine.get_tensor_shape(binding_index)
    binding_dtype = engine.get_binding_dtype(binding_index) if hasattr(engine, 'get_binding_dtype') else engine.get_tensor_dtype(binding_index)
    print("Binding {}: Name: {}, Shape: {}, Dtype: {}".format(binding_index, binding_name, binding_shape, binding_dtype))

    # Calculate and print the size of the binding
    np_dtype = trt_dtype_to_np(binding_dtype)
    size = trt.volume(binding_shape) * np.dtype(np_dtype).itemsize
    print("Binding {}: Size in bytes: {}".format(binding_index, size))
