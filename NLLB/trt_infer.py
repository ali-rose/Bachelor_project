import torch
from transformers import AutoTokenizer
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

model_dir = "/root/autodl-tmp/nllb-200-3.3B"
trt_model_dir = "/root/autodl-tmp/trt_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir)

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

trt_logger = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(trt_logger)

encoder_engine = load_engine(trt_runtime, f"{trt_model_dir}/encoder_model.trt")
decoder_engine = load_engine(trt_runtime, f"{trt_model_dir}/decoder_model.trt")

encoder_context = encoder_engine.create_execution_context()
decoder_context = decoder_engine.create_execution_context()

MAX_LENGTH = 256
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

language_map = {
    "英语": "eng_Latn",
    "中文": "zho_Hans",
    "法语": "fra_Latn",
    "德语": "deu_Latn",
    "日语": "jpn_Jpan",
    "韩语": "kor_Hang",
}

print("Engine type:", type(encoder_engine))
print("Engine attributes:", dir(encoder_engine))

def print_binding_info(engine):
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        print(f"Binding {i}:")
        print(f"  Name: {name}")
        print(f"  Shape: {engine.get_tensor_shape(name)}")
        print(f"  Dtype: {engine.get_tensor_dtype(name)}")
        print(f"  Is input: {engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT}")
        print()

print("Encoder Engine Bindings:")
print_binding_info(encoder_engine)
print("\nDecoder Engine Bindings:")
print_binding_info(decoder_engine)

def translate_with_detailed_timing(texts, src_lang, tgt_lang):
    src_code = language_map.get(src_lang, src_lang)
    tgt_code = language_map.get(tgt_lang, tgt_lang)
    
    start_time = time.time()
    
    encode_start_time = time.time()
    inputs = tokenizer([f"{src_code}: {text}" for text in texts], return_tensors="pt", padding=True)
    input_ids = inputs['input_ids'].numpy()
    attention_mask = inputs['attention_mask'].numpy()
    encode_end_time = time.time()
    
    tgt_lang_token = tokenizer.convert_tokens_to_ids(tgt_code)
    
    generate_start_time = time.time()
    
    # 编码器推理
    batch_size, seq_length = input_ids.shape
    encoder_output = np.empty((batch_size, seq_length, 2048), dtype=np.float32)

    d_input_ids = cuda.mem_alloc(input_ids.nbytes)
    d_attention_mask = cuda.mem_alloc(attention_mask.nbytes)
    d_encoder_output = cuda.mem_alloc(encoder_output.nbytes)

    cuda.memcpy_htod(d_input_ids, input_ids)
    cuda.memcpy_htod(d_attention_mask, attention_mask)

    bindings = [int(d_input_ids), int(d_attention_mask), int(d_encoder_output)]

    encoder_context.execute_v2(bindings=bindings)
    cuda.memcpy_dtoh(encoder_output, d_encoder_output)
    
   # 解码器推理
    decoder_input_ids = np.full((batch_size, 1), BOS_TOKEN_ID, dtype=np.int32)
    decoder_attention_mask = np.ones((batch_size, 1), dtype=np.int32)
    
    generated_sequences = []
    for _ in range(MAX_LENGTH):

        
        decoder_context.set_input_shape("encoder_attention_mask", attention_mask.shape)
        decoder_context.set_input_shape("input_ids", decoder_input_ids.shape)
        decoder_context.set_input_shape("encoder_hidden_states", encoder_output.shape)
        
        logits_shape = (batch_size, decoder_input_ids.shape[1], 256206)
        logits = np.empty(logits_shape, dtype=np.float32)

        d_decoder_input_ids = cuda.mem_alloc(decoder_input_ids.nbytes)
        d_decoder_attention_mask = cuda.mem_alloc(decoder_attention_mask.nbytes)
        d_encoder_output = cuda.mem_alloc(encoder_output.nbytes)
        d_logits = cuda.mem_alloc(logits.nbytes)

        cuda.memcpy_htod(d_decoder_input_ids, decoder_input_ids)
        cuda.memcpy_htod(d_decoder_attention_mask, decoder_attention_mask)
        cuda.memcpy_htod(d_encoder_output, encoder_output)

        bindings = [int(d_decoder_attention_mask), int(d_decoder_input_ids), int(d_encoder_output), int(d_logits)]

        decoder_context.execute_v2(bindings=bindings)
        cuda.memcpy_dtoh(logits, d_logits)
        
        next_token_logits = logits[:, -1, :]
        next_tokens = np.argmax(next_token_logits, axis=-1)
        
        decoder_input_ids = np.concatenate([decoder_input_ids, next_tokens[:, np.newaxis]], axis=-1)
        decoder_attention_mask = np.concatenate([decoder_attention_mask, np.ones((batch_size, 1), dtype=np.int32)], axis=-1)
        
        generated_sequences.append(next_tokens)
        
        if np.all(next_tokens == EOS_TOKEN_ID):
            break
    
    generated_sequences = np.array(generated_sequences).T
    
    generate_end_time = time.time()
    
    decode_start_time = time.time()
    outputs = [tokenizer.decode(seq[seq != PAD_TOKEN_ID], skip_special_tokens=True) for seq in generated_sequences]
    decode_end_time = time.time()
    
    encode_time = encode_end_time - encode_start_time
    generate_time = generate_end_time - generate_start_time
    decode_time = decode_end_time - decode_start_time
    total_time = time.time() - start_time
    
    input_tokens = np.sum(attention_mask)
    output_tokens = sum(len(tokenizer.encode(output)) for output in outputs)
    
    time_per_output_token = generate_time / output_tokens if output_tokens > 0 else 0
    
    return outputs, {
        "total_time": total_time,
        "encode_time": encode_time,
        "generate_time": generate_time,
        "decode_time": decode_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "time_per_input_token": encode_time / input_tokens if input_tokens > 0 else 0,
        "time_per_output_token": time_per_output_token,
        "time_per_token_overall": generate_time / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0
    }

def run_detailed_translation_test(texts, src_lang="英语", tgt_lang="中文", num_runs=5):
    total_stats = {
        "total_time": 0,
        "encode_time": 0,
        "generate_time": 0,
        "decode_time": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "time_per_input_token": 0,
        "time_per_output_token": 0,
        "time_per_token_overall": 0
    }
    
    for _ in range(num_runs):
        translations, stats = translate_with_detailed_timing(texts, src_lang, tgt_lang)
        for key in total_stats:
            total_stats[key] += stats[key]
    
    avg_stats = {key: value / num_runs for key, value in total_stats.items()}
    
    print(f"\n源文本 ({src_lang}):")
    for text in texts:
        print(text)
    print(f"\n翻译 ({tgt_lang}):")
    for translation in translations:
        print(translation)
    print(f"\n平均统计 ({num_runs} 次运行):")
    print(f"总时间: {avg_stats['total_time']:.4f} 秒")
    print(f"编码时间: {avg_stats['encode_time']:.4f} 秒")
    print(f"生成时间: {avg_stats['generate_time']:.4f} 秒")
    print(f"解码时间: {avg_stats['decode_time']:.4f} 秒")
    print(f"输入 tokens: {avg_stats['input_tokens']:.0f}")
    print(f"输出 tokens: {avg_stats['output_tokens']:.0f}")
    print(f"每个输入 token 的处理时间: {avg_stats['time_per_input_token']*1000:.4f} 毫秒")
    print(f"每个输出 token 的生成时间: {avg_stats['time_per_output_token']*1000:.4f} 毫秒")
    print(f"整体每个 token 的时间: {avg_stats['time_per_token_overall']*1000:.4f} 毫秒")

test_texts = [
    "Gwen Tennyson is Ben Tennyson's cousin and Max Tennyson's granddaughter. In the Original Series, Gwen had green eyes and short red hair held by a blue hairclip and wore sapphire earrings, an elbow-length blue raglan shirt with a cat logo, white capri pants, and white sneakers with dark blue stripes without socks. She's spent her summer vacation on a road trip with Ben and Max going on adventures and fighting bad guys, be they alien or human.",
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! How are you today?"
]

run_detailed_translation_test(test_texts)