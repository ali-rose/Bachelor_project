import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def print_engine_bindings(engine):
    for binding_index in range(engine.num_bindings):
        binding_name = engine.get_binding_name(binding_index)
        binding_dims = engine.get_binding_shape(binding_index)
        binding_dtype = engine.get_binding_dtype(binding_index)
        # 判断当前绑定是输入还是输出
        if engine.binding_is_input(binding_index):
            binding_type = "Input"
        else:
            binding_type = "Output"
        print(f"{binding_type} '{binding_name}': Shape {binding_dims}, Dtype {binding_dtype}")

engine_path = '/root/wav2lip/Wav2Lip/wav2lip_dynamic.engine'  # 替换为你的.engine文件路径
engine = load_engine(engine_path)
print_engine_bindings(engine)
