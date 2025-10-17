import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# 设置目录
data_dir = "/root/autodl-tmp/nllb_processing"
model_dir = "/root/autodl-tmp/nllb-200-3.3B"
onnx_output_dir = "/root/autodl-tmp/onnx_model"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(onnx_output_dir, exist_ok=True)

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=data_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, cache_dir=data_dir)

# 获取模型配置
config = AutoConfig.from_pretrained(model_dir)

# 导出为ONNX
main_export(
    model_name_or_path=model_dir,
    output=onnx_output_dir,
    task="translation",
    do_validation=True,
    no_post_process=True,
    opset=14,
    use_past=True,  # 添加这个参数以生成包含past key values的模型
)

# 使用ORTModelForSeq2SeqLM加载并保存ONNX模型
ort_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_output_dir, use_cache=False)

print(f"ONNX模型已导出并保存到 {onnx_output_dir}")

# 如果之后需要使用ONNX模型，可以这样加载
# ort_model = ORTModelForSeq2SeqLM.from_pretrained(output_dir)

# # 准备一个更长、更复杂的样例输入
# src_text = "This is a longer and more complex sample input. It contains multiple sentences and should better represent the kind of text the model might encounter in real-world scenarios. By using a more substantial example, we aim to capture more of the model's behavior during the ONNX export process."
# src_lang = "eng_Latn"
# tgt_lang = "deu_Latn"

# # 编码输入
# inputs = tokenizer(f"{src_lang}: {src_text}", return_tensors="pt", padding="max_length", max_length=128)
# decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

# # 设置模型为评估模式
# model.eval()

# # 指定输出路径
# onnx_output_path = "/root/autodl-tmp/onnx_model/nllb_model_complex.onnx"

# # 使用 torch.onnx.export 导出模型
# torch.onnx.export(
#     model,
#     (inputs.input_ids, inputs.attention_mask, decoder_input_ids),
#     onnx_output_path,
#     input_names=['input_ids', 'attention_mask', 'decoder_input_ids'],
#     output_names=['logits'],
#     dynamic_axes={
#         'input_ids': {0: 'batch_size', 1: 'sequence'},
#         'attention_mask': {0: 'batch_size', 1: 'sequence'},
#         'decoder_input_ids': {0: 'batch_size', 1: 'sequence'},
#         'logits': {0: 'batch_size', 1: 'sequence', 2: 'vocab_size'}
#     },
#     opset_version=13
# )

# print(f"ONNX model exported successfully to {onnx_output_path}.")