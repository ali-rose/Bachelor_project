import tensorrt as trt

def build_engine_dynamic_wav2lip(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 可根据需求调整

        # 对于 Wav2Lip 模型，你需要为 'mel' 和 'image' 输入设置动态形状
        profile = builder.create_optimization_profile()
        
        # 这里假设 'mel' 和 'image' 的动态形状范围，实际应用时需要根据模型需求调整
        # 注意：这里的维度需要与你转换的 ONNX 模型匹配
       # 为 'mel' 输入设置动态形状，这里假设批量大小可以变化，但其余维度固定
        profile.set_shape('mel', min=(1, 1, 80, 16), opt=(64, 1, 80, 16), max=(128, 1, 80, 16))
        # 为 'image' 输入设置动态形状，同样假设批量大小可以变化，但其余维度固定
        profile.set_shape('image', min=(1, 6, 96, 96), opt=(64, 6, 96, 96), max=(128, 6, 96, 96))
        
        
        config.add_optimization_profile(profile)
        
        engine = builder.build_engine(network, config)
        if engine:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print(f"TensorRT引擎构建成功并保存至：{engine_file_path}")
        else:
            print("构建 TensorRT 引擎失败。")
        return engine

# 调用函数构建引擎
build_engine_dynamic_wav2lip("wav2lip_dynamic.onnx", "wav2lip_dynamic128.engine")
