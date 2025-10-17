import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx
import logging
import psutil
from memory_profiler import profile
import onnxruntime as ort
from onnx.external_data_helper import convert_model_to_external_data

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    logger.info(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def optimize_onnx_model(onnx_file_path):
    logger.info(f"开始优化ONNX模型: {onnx_file_path}")
    model = onnx.load(onnx_file_path)
    
    # 将权重分离到外部文件
    external_data_file = onnx_file_path + "_weights.pb"
    convert_model_to_external_data(model, all_tensors_to_one_file=True, location=external_data_file)
    
    # 使用ONNX Runtime优化模型
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    optimized_model = ort.InferenceSession(onnx_file_path, sess_options).get_optimized_model()
    
    optimized_onnx_file = onnx_file_path.replace('.onnx', '_optimized.onnx')
    onnx.save_model(optimized_model, optimized_onnx_file)
    logger.info(f"优化后的ONNX模型保存至: {optimized_onnx_file}")
    
    return optimized_onnx_file

@profile
def build_engine(onnx_file_path, engine_file_path, max_batch_size=1, max_workspace_size=4 << 30, fp16_mode=True):
    logger.info(f"开始构建引擎: {onnx_file_path}")
    logger.info(f"ONNX model size: {os.path.getsize(onnx_file_path) / (1024*1024):.2f} MB")
    log_memory_usage()

    try:
        # 优化ONNX模型
        optimized_onnx_file = optimize_onnx_model(onnx_file_path)
        
        logger.info("创建 TensorRT logger")
        trt_logger = trt.Logger(trt.Logger.VERBOSE)

        logger.info("创建 TensorRT builder")
        builder = trt.Builder(trt_logger)

        logger.info("创建 TensorRT network")
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        logger.info("创建 TensorRT config")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        logger.info(f"Workspace size set to: {max_workspace_size}")

        if fp16_mode:
            logger.info("启用 FP16 模式")
            config.set_flag(trt.BuilderFlag.FP16)

        logger.info("创建 ONNX parser")
        parser = trt.OnnxParser(network, trt_logger)

        logger.info("开始加载 ONNX 模型")
        with open(optimized_onnx_file, 'rb') as model:
            model_content = model.read()
            logger.info(f"ONNX 模型内容大小: {len(model_content) / (1024*1024):.2f} MB")
            if not parser.parse(model_content):
                logger.error('解析 ONNX 文件失败')
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return None
        logger.info("ONNX 模型解析成功")
        log_memory_usage()

        logger.info("创建优化配置")
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input = network.get_input(i)
            shape = input.shape
            profile.set_shape(input.name, (1, *shape[1:]), (max_batch_size, *shape[1:]), (max_batch_size, *shape[1:]))
        config.add_optimization_profile(profile)
        logger.info("优化配置创建完成")

        logger.info("开始构建 TensorRT 引擎")
        engine = builder.build_serialized_network(network, config)
        logger.info("TensorRT 引擎构建完成")
        log_memory_usage()

        logger.info(f"保存 TensorRT 引擎到: {engine_file_path}")
        with open(engine_file_path, "wb") as f:
            f.write(engine)
        logger.info("TensorRT 引擎保存成功")

        return engine

    except Exception as e:
        logger.error(f"构建引擎过程中发生错误: {e}", exc_info=True)
        log_memory_usage()
        return None

def main():
    try:
        logger.info("开始转换编码器模型")
        encoder_onnx_path = "/root/autodl-tmp/onnx_model/encoder_model.onnx"
        encoder_engine_path = "/root/autodl-tmp/trt_model/encoder_model.trt"
        build_engine(encoder_onnx_path, encoder_engine_path)

        logger.info("开始转换解码器模型")
        decoder_onnx_path = "/root/autodl-tmp/onnx_model/decoder_model.onnx"
        decoder_engine_path = "/root/autodl-tmp/trt_model/decoder_model.trt"
        build_engine(decoder_onnx_path, decoder_engine_path)

        logger.info("TensorRT 模型转换完成")

    except Exception as e:
        logger.error(f"主程序发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()