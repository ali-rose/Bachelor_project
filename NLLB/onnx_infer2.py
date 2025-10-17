import os
import time
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

# 设置模型和tokenizer路径
model_dir = "/root/autodl-tmp/onnx_model"
tokenizer_dir = "/root/autodl-tmp/nllb-200-3.3B"

# 加载tokenizer和ONNX模型
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = ORTModelForSeq2SeqLM.from_pretrained(model_dir, use_cache=False)  # 设置use_cache=False

# 定义语言代码
ENGLISH = "eng_Latn"
CHINESE = "zho_Hans"
GERMAN = "deu_Latn"

def translate_with_detailed_timing(text, source_lang, target_lang):
    # 开始计时
    start_time = time.time()
    
    # 记录编码开始时间
    encode_start_time = time.time()
    inputs = tokenizer(f"{source_lang}: {text}", return_tensors="pt")
    encode_end_time = time.time()
    
    # 获取目标语言的token ID
    tgt_lang_token = tokenizer.convert_tokens_to_ids(target_lang)
    
    # 记录生成开始时间
    generate_start_time = time.time()
    
    # 生成翻译
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tgt_lang_token,
        max_length=128,
        num_beams=5,
        num_return_sequences=1
    )
    
    # 记录生成结束时间
    generate_end_time = time.time()
    
    # 记录解码开始时间
    decode_start_time = time.time()
    output = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    decode_end_time = time.time()
    
    # 计算各阶段时间
    encode_time = encode_end_time - encode_start_time
    generate_time = generate_end_time - generate_start_time
    decode_time = decode_end_time - decode_start_time
    total_time = time.time() - start_time
    
    # 计算token数量
    input_tokens = len(inputs['input_ids'][0])
    output_tokens = len(tokenizer.encode(output))
    
    # 计算每个输出token的生成时间
    time_per_output_token = generate_time / len(translated[0])
    
    return output, {
        "total_time": total_time,
        "encode_time": encode_time,
        "generate_time": generate_time,
        "decode_time": decode_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "time_per_input_token": encode_time / input_tokens,
        "time_per_output_token": time_per_output_token,
        "time_per_token_overall": generate_time / (input_tokens + output_tokens)
    }

def run_detailed_translation_test(text, src_lang, tgt_lang, num_runs=1):
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
        translation, stats = translate_with_detailed_timing(text, src_lang, tgt_lang)
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # 计算平均值
    avg_stats = {key: value / num_runs for key, value in total_stats.items()}
    
    print(f"\n源文本 ({src_lang}): {text}")
    print(f"翻译 ({tgt_lang}): {translation}")
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


# 测试中文到英文的翻译
chinese_text = "今天天气真好。"
run_detailed_translation_test(chinese_text, CHINESE, ENGLISH)

# 测试英文到中文的翻译
english_text = "Gwen Tennyson is Ben Tennyson's cousin and Max Tennyson's granddaughter. In the Original Series, Gwen had green eyes and short red hair held by a blue hairclip and wore sapphire earrings, an elbow-length blue raglan shirt with a cat logo, white capri pants, and white sneakers with dark blue stripes without socks. She's spent her summer vacation on a road trip with Ben and Max going on adventures and fighting bad guys, be they alien or human."
run_detailed_translation_test(english_text, ENGLISH, GERMAN)