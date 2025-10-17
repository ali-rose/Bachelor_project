import torch
from models import Wav2Lip

# 初始化模型和加载权重
model = Wav2Lip()
checkpoint = torch.load('/root/wav2lip/Wav2Lip/checkpoints/wav2lip_gan.pth', map_location='cuda:2')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 假设的输入
# 根据模型的输入维度进行调整
x_dummy = torch.randn(1, 6, 96, 96)  # 假设的面部序列输入
mel_dummy = torch.randn(1, 1, 80, 16)  # 假设的音频序列输入

# 导出ONNX模型
torch.onnx.export(
    model, 
    (mel_dummy, x_dummy), 
    "wav2lip_dynamic.onnx", 
    opset_version=12, 
    input_names=['mel', 'image'], 
    output_names=['output'], 
    dynamic_axes={
        'mel' : {0: 'batch_size'},  # 第0维（批量大小）是动态的
        'image' : {0: 'batch_size'},  # 第0维（批量大小）是动态的
        'output' : {0: 'batch_size'}  # 假设输出的批量大小也是动态的
    },
    do_constant_folding=True,  # 优化模型中的常量折叠
)



'''
for binding_index in range(engine.num_bindings):
            binding_shape = context.get_tensor_shape('mel')
            print("Binding shape for index", binding_index, "is", binding_shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding_index))
            
            # 根据形状和数据类型分配内存
            host_mem = np.empty(binding_shape, dtype=dtype)
            
            if engine.binding_is_input(binding_index):
                if binding_index == mel_binding_index:
                    print("123", host_mem)
                    host_mem[:] = mel_input  # 假设 mel_input 已准备
                elif binding_index == image_binding_index:
                    print("456", host_mem)
                    print("456", image_binding_index)
                    host_mem[:] = image_input  # 假设 image_input 已准备
                inputs.append(host_mem)
            else:
                outputs.append(host_mem)
            
            bindings[binding_index] = host_mem
            '''