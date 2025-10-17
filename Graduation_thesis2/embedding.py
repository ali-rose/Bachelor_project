from transformers import AutoTokenizer, AutoModel
import torch
import pdfplumber

# 本地模型和tokenizer的路径
embedding_model_path = '/root/AiLi/ChatPDF-main/checkpoints'
embedding_tokenizer_path = '/root/AiLi/ChatPDF-main/checkpoints'

# 检查路径是否正确，并加载模型和tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(embedding_tokenizer_path)
    model = AutoModel.from_pretrained(embedding_model_path)
    model.eval()
except Exception as e:
    print(f"Error loading model or tokenizer from local path: {e}")
    raise

# PDF文件路径和窗口参数
pdf_path = "1.pdf"
window_size = 512
step_size = 256

sentence_embeddings_list = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            # 对整个文本分段
            tokenized_text = tokenizer.tokenize(text)
            print("tokenized_text---------------------:", len(tokenized_text))
            for i in range(0, len(tokenized_text), step_size):
                # 保证不超过最大长度
                print("tokenized_text2222222---------------------:", len(tokenized_text[i:i + window_size]))
                segment = tokenizer.convert_tokens_to_string(tokenized_text[i:i + window_size])
                encoded_input = tokenizer.encode_plus(segment, padding='max_length', truncation=True, max_length=window_size, return_tensors='pt')
                with torch.no_grad():
                    model_output = model(**encoded_input)
                    sentence_embedding = model_output.last_hidden_state[:, 0, :]  # 使用CLS token的输出作为句子嵌入
                    # print("Sentence embedding------------------------:", sentence_embedding)
                    sentence_embeddings_list.append(sentence_embedding)

# 合并并标准化所有页面的嵌入向量
if sentence_embeddings_list:
    sentence_embeddings = torch.cat(sentence_embeddings_list, dim=0)
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:", sentence_embeddings)
