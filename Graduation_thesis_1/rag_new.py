import faiss
import numpy as np
import pandas as pd
import pickle
import os
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# 下载punkt句子分割模型
# nltk.download('punkt')


# ---------------------------- 配置部分 ----------------------------

# 路径配置
MODEL_PATH = "/root/autodl-tmp/glm4-9B/glm3-6b"  # 生成模型路径
EMBEDDING_MODEL_NAME = "/root/autodl-tmp/glm4-9B/embeddings"  # 本地嵌入模型名称或路径
DATA_PATH = '/root/autodl-tmp/glm4-9B/government_documents_2024-12-18_13-29-26_adjust.xlsx'      # 替换为你的Excel文件路径
EMBEDDING_PATH = '/root/autodl-tmp/glm4-9B/embedding_cashe/embeddings_law.npy'    # 嵌入向量保存路径
METADATA_PATH = '/root/autodl-tmp/glm4-9B/embedding_cashe/metadata_law.pkl'       # 元数据保存路径
INDEX_PATH = '/root/autodl-tmp/glm4-9B/embedding_cashe/policy_faiss_law.index'    # FAISS索引保存路径
TOP_K = 5                             # 检索相似文档块的数量
CHUNK_SIZE = 8192                   # 每个文本块的字符数，可根据需要调整

# ----------------------- 模型加载部分 ----------------------------

# 加载生成模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 或 float16
    trust_remote_code=True,
    device_map="auto"
).eval()

# 定义停止生成的条件
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        if isinstance(stop_ids, int):
            stop_ids = [stop_ids]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stop = StopOnTokens()

# 加载嵌入模型
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ----------------------- 文本分块函数 ----------------------------

def split_text_into_chunks(text, chunk_size):
    """
    将长文本按句子分割，并将句子组合成指定字符数的块。

    参数:
        text (str): 要分割的文本。
        chunk_size (int): 每个块的最大字符数。

    返回:
        list: 文本块列表。
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 如果加入当前句子后超过chunk_size，先保存当前块
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # 如果单个句子就超过chunk_size，强制分割
                chunks.append(sentence[:chunk_size])
                current_chunk = sentence[chunk_size:]
        else:
            # 加入当前句子
            current_chunk += " " + sentence if current_chunk else sentence

    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# ----------------------- 嵌入生成函数 ----------------------------

def get_embeddings_local(text):
    """
    使用本地嵌入模型生成嵌入向量。

    参数:
        text (str): 要生成嵌入的文本。

    返回:
        np.ndarray: 嵌入向量。
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()

        # 文本分块
        chunks = split_text_into_chunks(text, CHUNK_SIZE)

        # 生成所有块的嵌入
        if len(chunks) > 0:
            embeddings = embedding_model.encode(chunks)
            # 取所有块嵌入的平均值作为整体文本的嵌入
            embedding = np.mean(embeddings, axis=0)
        else:
            embedding = np.zeros(embedding_model.get_sentence_embedding_dimension(), dtype='float32')

        return embedding
    except Exception as e:
        print(f"生成嵌入时出错: {e}")
        return np.zeros(embedding_model.get_sentence_embedding_dimension(), dtype='float32')

# ----------------------- 生成回答函数 ----------------------------

def generate_response_local(prompt):
    """
    使用本地GLM-4-9B模型生成回答。

    参数:
        prompt (str): 提示文本。

    返回:
        str: 生成的回答。
    """
    try:
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generate_kwargs = {
            "input_ids": tokenizer.encode(prompt, return_tensors="pt").to(model.device),
            "attention_mask": torch.ones((1, tokenizer.encode(prompt, return_tensors="pt").shape[1]), dtype=torch.long).to(model.device),
            "streamer": streamer,
            "max_new_tokens": 1024,
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.6,
            "stopping_criteria": StoppingCriteriaList([stop]),
            "repetition_penalty": 1.2,
            "eos_token_id": model.config.eos_token_id,
        }
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        response = ""
        print("GLM-4:", end="", flush=True)
        for new_token in streamer:
            if new_token:
                print(new_token, end="", flush=True)
                response += new_token
        print()

        torch.cuda.empty_cache()
        return response.strip()
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return "抱歉，生成回答时出现了错误。"

# ----------------------- 数据准备函数 ----------------------------

def prepare_data(data_path, embedding_path, metadata_path, index_path):
    """
    准备数据集和FAISS索引。

    参数:
        data_path (str): Excel文件路径。
        embedding_path (str): 嵌入向量文件路径。
        metadata_path (str): 元数据文件路径。
        index_path (str): FAISS索引文件路径。

    返回:
        faiss.Index: FAISS索引对象。
        list: 元数据列表。
    """
    try:
        # 加载数据集
        print("加载数据集...")
        df = pd.read_excel(data_path)

        # 选择需要的列
        df = df[['标题', '网址', '发布时间', '正文']]

        # 检查是否已经存在嵌入和索引
        if os.path.exists(embedding_path) and os.path.exists(index_path) and os.path.exists(metadata_path):
            # 加载预生成的嵌入和索引
            print("加载预生成的嵌入和索引...")
            embeddings = np.load(embedding_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            index = faiss.read_index(index_path)
        else:
            print("生成嵌入和构建FAISS索引...")
            # 生成嵌入
            embeddings = []
            metadata = []
            for idx, row in enumerate(df.itertuples(index=False)):
                title = row.标题
                link = row.网址
                publish_date = row.发布时间
                text = row.正文 if pd.notna(row.正文) else ""

                # 生成整个正文的嵌入
                embedding = get_embeddings_local(text)
                embeddings.append(embedding)
                metadata.append({
                    '标题': title,
                    '网址': link,
                    '发布时间': publish_date,
                    '正文': text
                })

                if (idx + 1) % 50 == 0:
                    print(f"已处理 {idx + 1} 条数据")

            embeddings = np.array(embeddings).astype('float32')

            # 保存嵌入
            np.save(embedding_path, embeddings)

            # 保存元数据
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            # 构建FAISS索引
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # 保存索引
            faiss.write_index(index, index_path)

        print("数据集和向量索引准备完毕。")
        return index, metadata
    except Exception as e:
        print(f"准备数据时出错: {e}")
        return None, None

# ----------------------- 生成回答函数 ----------------------------

def generate_answer(question, index, metadata, top_k=TOP_K):
    """
    根据用户问题生成回答。

    参数:
        question (str): 用户问题。
        index (faiss.Index): FAISS索引对象。
        metadata (list): 元数据列表。
        top_k (int): 检索相似文档块的数量。

    返回:
        str: 生成的回答。
    """
    try:
        # 生成问题的嵌入
        question_embedding = get_embeddings_local(question)
        question_vector = np.array([question_embedding]).astype('float32')

        # 在FAISS中检索最相似的文档块
        distances, indices = index.search(question_vector, top_k)

        # 获取相关文档块
        retrieved_docs = [metadata[idx] for idx in indices[0]]

        # 构建上下文
        context = "\n\n".join(
            [f"标题: {doc['标题']}\n内容: {doc['正文'][:1000]}\n网址: {doc['网址']}" for doc in retrieved_docs]
        )

        # 构建提示（Prompt）
        prompt = f"""
你是一个专业的政策咨询助手。根据以下提供的政策信息，回答用户的问题。

用户问题: {question}

相关政策信息:
{context}

请综合以上信息，提供详细、准确的回答，并引用相关政策的标题和链接。
"""

        # 使用本地模型生成回答
        answer = generate_response_local(prompt)

        return answer
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return "抱歉，生成回答时出现了错误。"

# ------------------------- 主函数 ----------------------------

def main():
    print("\n政策咨询助手已启动。请输入你的问题（输入 'exit' 退出）：")

    # 准备数据和索引
    index, metadata = prepare_data(DATA_PATH, EMBEDDING_PATH, METADATA_PATH, INDEX_PATH)

    if index is None or metadata is None:
        print("数据准备失败，程序退出。")
        return

    # 交互循环
    print("\n政策咨询助手已启动。请输入你的问题（输入 'exit' 退出）：")
    while True:
        question = input("你的问题：").strip()
        if question.lower() in ['exit', 'quit']:
            print("退出政策咨询助手。")
            break
        if not question:
            print("请输入一个有效的问题。")
            continue
        answer = generate_answer(question, index, metadata)
        print("\n回答：")
        print(answer)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()