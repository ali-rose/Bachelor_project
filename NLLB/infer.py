import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from optimum.onnxruntime import ORTModelForSeq2SeqLM


model_dir = "/root/autodl-tmp/nllb-200-3.3B"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

language_map = {
    "英语": "eng_Latn",
    "中文": "zho_Hans",
    "法语": "fra_Latn",
    "德语": "deu_Latn",
    "日语": "jpn_Jpan",
    "韩语": "kor_Hang",
    # 可以根据需要添加更多语言
}

def translate_with_detailed_timing(text, src_lang, tgt_lang):
    src_code = language_map.get(src_lang, src_lang)
    tgt_code = language_map.get(tgt_lang, tgt_lang)
    
    # 开始计时
    start_time = time.time()
    
    # 记录编码开始时间
    encode_start_time = time.time()
    inputs = tokenizer(f"{src_code}: {text}", return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    encode_end_time = time.time()
    
    tgt_lang_token = tokenizer.convert_tokens_to_ids(tgt_code)
    
    # 记录生成开始时间
    generate_start_time = time.time()
    
    with torch.no_grad():
        translated = model.generate(
            **inputs, 
            forced_bos_token_id=tgt_lang_token,
            max_length=128,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # 记录生成结束时间
    generate_end_time = time.time()
    
    # 记录解码开始时间
    decode_start_time = time.time()
    output = tokenizer.batch_decode(translated.sequences, skip_special_tokens=True)[0]
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
    time_per_output_token = generate_time / len(translated.sequences[0])
    
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

def run_detailed_translation_test(text, src_lang="英语", tgt_lang="中文", num_runs=5):
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

# 运行测试
test_text = "Gwen Tennyson is Ben Tennyson's cousin and Max Tennyson's granddaughter. In the Original Series, Gwen had green eyes and short red hair held by a blue hairclip and wore sapphire earrings, an elbow-length blue raglan shirt with a cat logo, white capri pants, and white sneakers with dark blue stripes without socks. She's spent her summer vacation on a road trip with Ben and Max going on adventures and fighting bad guys, be they alien or human."
run_detailed_translation_test(test_text)