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
from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from rouge_chinese import Rouge
import jieba
import re

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')

# ---------------------------- 配置部分 ----------------------------
MODEL_PATH = "/root/autodl-tmp/glm4-9B/glm3-6b"  # 生成模型路径
EMBEDDING_MODEL_NAME = "/root/autodl-tmp/glm4-9B/embeddings"  # 嵌入模型路径
DATA_PATH = '/root/autodl-tmp/glm4-9B/merged_deduplicated.xlsx'  # 数据集路径
EMBEDDING_PATH = '/root/autodl-tmp/glm4-9B/embedding_cashe/embeddings_merged.npy'  # 嵌入向量缓存路径
METADATA_PATH = '/root/autodl-tmp/glm4-9B/embedding_cashe/metadata_merged.pkl'  # 元数据缓存路径
INDEX_PATH = '/root/autodl-tmp/glm4-9B/embedding_cashe/policy_faiss_merged.index'  # FAISS索引缓存路径
CATEGORY_EMBEDDING_PATH = '/root/autodl-tmp/glm4-9B/embedding_cashe/category_embeddings_merged.npy'  # 类别嵌入向量缓存路径
TOP_K = 5  # 检索的文档数量
CHUNK_SIZE = 8192  # 文本块大小
CATEGORY_MATCH_THRESHOLD = 0.2  # 类别匹配阈值（20%）

# ---------------------------- 模型加载部分 ----------------------------
# 加载生成模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
).eval()

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
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---------------------------- 评估指标类 ----------------------------
class ResponseEvaluator:
    def __init__(self):
        self.rouge = Rouge()
        
    def calculate_relevance_score(self, question, answer, context):
        question_embedding = embedding_model.encode([question])[0]
        answer_embedding = embedding_model.encode([answer])[0]
        context_embedding = embedding_model.encode([context])[0]
        
        qa_similarity = cosine_similarity([question_embedding], [answer_embedding])[0][0]
        ac_similarity = cosine_similarity([answer_embedding], [context_embedding])[0][0]
        
        relevance_score = (qa_similarity + ac_similarity) / 2
        return min(relevance_score * 10, 10)
    
    def calculate_coherence_score(self, answer):
        sentences = sent_tokenize(answer)
        if len(sentences) <= 1:
            return 10.0
            
        coherence_scores = []
        for i in range(len(sentences)-1):
            sent1_embedding = embedding_model.encode([sentences[i]])[0]
            sent2_embedding = embedding_model.encode([sentences[i+1]])[0]
            similarity = cosine_similarity([sent1_embedding], [sent2_embedding])[0][0]
            coherence_scores.append(similarity)
            
        return min(np.mean(coherence_scores) * 10, 10)
    
    def calculate_completeness_score(self, answer, context):
        try:
            # print("原始Answer:", answer)
            # print("原始Context:", context)

        # 使用结巴分词
            def tokenize(text):
                return list(jieba.cut(text))

            answer_tokens = tokenize(answer)
            context_tokens = tokenize(context)

            # print("Answer分词:", answer_tokens)
            # print("Context分词:", context_tokens)

        # 计算重叠词数
            overlap_tokens = set(answer_tokens) & set(context_tokens)
        
        # 手动计算相似度
            overlap_ratio = len(overlap_tokens) / len(set(context_tokens))
        
            print("重叠词数:", len(overlap_tokens))
            print("重叠比例:", overlap_ratio)

        # 使用重叠比例计算分数
            score = min(overlap_ratio * 10, 10)
        
            print("计算得分:", score)
        
            return score

        except Exception as e:
            print(f"分数计算错误: {e}")
            return 5.0
    
    def calculate_source_citation_score(self, answer):
        has_title = bool(re.search(r'《.*?》|".*?"', answer))
        has_link = bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', answer))
        
        if has_title and has_link:
            return 10.0
        elif has_title or has_link:
            return 7.0
        return 4.0
    
    def evaluate_response(self, question, answer, context):
        relevance_score = self.calculate_relevance_score(question, answer, context)
        coherence_score = self.calculate_coherence_score(answer)
        completeness_score = self.calculate_completeness_score(answer, context)
        citation_score = self.calculate_source_citation_score(answer)
        
        weights = {
            'relevance': 0.35,
            'coherence': 0.25,
            'completeness': 0.25,
            'citation': 0.15
        }
        
        overall_score = (
            relevance_score * weights['relevance'] +
            coherence_score * weights['coherence'] +
            completeness_score * weights['completeness'] +
            citation_score * weights['citation']
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'relevance_score': round(relevance_score, 2),
            'coherence_score': round(coherence_score, 2),
            'completeness_score': round(completeness_score, 2),
            'citation_score': round(citation_score, 2)
        }

# ---------------------------- 文本处理函数 ----------------------------
def split_text_into_chunks(text, chunk_size):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                chunks.append(sentence[:chunk_size])
                current_chunk = sentence[chunk_size:]
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# ---------------------------- 嵌入生成函数 ----------------------------
def get_embeddings_local(text):
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        
        chunks = split_text_into_chunks(text, CHUNK_SIZE)
        
        if len(chunks) > 0:
            embeddings = embedding_model.encode(chunks)
            embedding = np.mean(embeddings, axis=0)
        else:
            embedding = np.zeros(embedding_model.get_sentence_embedding_dimension(), dtype='float32')
        
        return embedding
    except Exception as e:
        print(f"生成嵌入向量时出错: {e}")
        return np.zeros(embedding_model.get_sentence_embedding_dimension(), dtype='float32')

# ---------------------------- 类别匹配函数 ----------------------------
def match_category(question, categories):
    question_words = set(jieba.lcut(question.lower()))
    
    category_scores = {}
    for category in categories:
        category_words = set(jieba.lcut(category.lower()))
        matching_words = question_words.intersection(category_words)
        score = len(matching_words) / len(category_words)
        if score >= CATEGORY_MATCH_THRESHOLD:
            category_scores[category] = score
    
    matched_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    return [cat for cat, score in matched_categories]

# ---------------------------- 回答生成函数 ----------------------------
def generate_response_local(prompt):
    try:
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generate_kwargs = {
            "input_ids": tokenizer.encode(prompt, return_tensors="pt").to(model.device),
            "attention_mask": torch.ones((1, tokenizer.encode(prompt, return_tensors="pt").shape[1])).to(model.device),
            "streamer": streamer,
            "max_new_tokens": 8192,
            "do_sample": True,
            "top_p": 0.7,
            "temperature": 0.8,
            "stopping_criteria": StoppingCriteriaList([stop]),
            "repetition_penalty": 1.2,
            "eos_token_id": model.config.eos_token_id,
        }
        
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        response = ""
        print("生成回答中:", end="", flush=True)
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

# ---------------------------- 数据准备函数 ----------------------------
def prepare_data(data_path, embedding_path, metadata_path, index_path, category_embedding_path):
    try:
        print("加载数据集...")
        df = pd.read_excel(data_path)
        df = df[['标题', '网址', '发布时间', '主题类别', '正文']]
        
        if (os.path.exists(embedding_path) and 
            os.path.exists(metadata_path) and 
            os.path.exists(index_path) and 
            os.path.exists(category_embedding_path)):
            
            print("加载缓存的嵌入向量和索引...")
            embeddings = np.load(embedding_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            index = faiss.read_index(index_path)
            category_embeddings = np.load(category_embedding_path)
            
        else:
            print("生成新的嵌入向量和索引...")
            embeddings = []
            metadata = []
            for idx, row in enumerate(df.itertuples(index=False)):
                text = row.正文 if pd.notna(row.正文) else ""
                embedding = get_embeddings_local(text)
                embeddings.append(embedding)
                metadata.append({
                    '标题': row.标题,
                    '网址': row.网址,
                    '发布时间': row.发布时间,
                    '主题类别': row.主题类别,
                    '正文': text
                })
                
                if (idx + 1) % 50 == 0:
                    print(f"已处理 {idx + 1} 条文档")
            
            embeddings = np.array(embeddings).astype('float32')
            categories = df['主题类别'].unique()
            category_embeddings = np.array([
                get_embeddings_local(category) for category in categories
            ]).astype('float32')
            
            np.save(embedding_path, embeddings)
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            np.save(category_embedding_path, category_embeddings)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            faiss.write_index(index, index_path)
            
        print("数据准备完成")
        return index, metadata, category_embeddings
    except Exception as e:
        print(f"准备数据时出错: {e}")
        return None, None, None

# ---------------------------- 回答生成主函数 ----------------------------
def generate_answer(question, index, metadata, category_embeddings, evaluator, top_k=TOP_K):
    try:
        categories = list(set(doc['主题类别'] for doc in metadata))
        matched_categories = match_category(question, categories)
        
        question_embedding = get_embeddings_local(question)
        question_vector = np.array([question_embedding]).astype('float32')

        # Create a mapping of document indices to preserve relationship with original embeddings
        if matched_categories:
            # Get indices of documents that match the categories
            filtered_indices = [
                idx for idx, doc in enumerate(metadata)
                if doc['主题类别'] in matched_categories
            ]
            
            if len(filtered_indices) < top_k:
                # If we don't have enough matches, use all documents
                filtered_indices = list(range(len(metadata)))
        else:
            # If no categories matched, use all documents
            filtered_indices = list(range(len(metadata)))

        # Get the embeddings for the filtered documents using their original indices
        filtered_embeddings = np.array([
            np.load(EMBEDDING_PATH)[idx] for idx in filtered_indices
        ]).astype('float32')
        
        # Create temporary index for search
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)
        
        # Perform search
        distances, temp_indices = temp_index.search(question_vector, min(top_k, len(filtered_indices)))
        
        # Map back to original metadata using filtered_indices
        retrieved_docs = [metadata[filtered_indices[idx]] for idx in temp_indices[0]]

        
        # Rest of the function remains the same...
        context_parts = []
        total_length = 0
        max_content_length = 600  # 每个文档内容的最大长度
        max_total_length = 3000   # 总context的最大长度

        for doc in retrieved_docs:
            # 格式化每个文档的内容
            content = doc['正文'][:max_content_length] if doc['正文'] else ""
            doc_context = f"文档：《{doc['标题']}》\n要点：{content}\n"
            
            # 检查总长度
            if total_length + len(doc_context) > max_total_length:
                break
            
            context_parts.append(doc_context)
            total_length += len(doc_context)

        context = "\n".join(context_parts)

        # 优化prompt结构和长度
#         prompt = f"""请根据以下政策文件回答用户问题。

# 问题：{question}

# 参考资料：
# {context}

# 请提供简明的回答，重点说明：
# 1. 相关政策要点
# 2. 具体措施和规定
# 3. 政策依据来源"""
        prompt = f"""
你是一个专业的政策咨询助手。我会给你一些政策文件作为参考信息，请你基于这些信息回答用户的问题。

用户问题: {question}

参考政策信息:
{context}

请根据以上参考信息，提供详细、准确的回答。要求：
1. 回答要有理有据，直接引用相关政策内容
2. 需要明确指出信息来源（包括政策标题和链接）
3. 如果问题涉及多个方面，要分点回答
4. 语言要清晰、专业，避免过于口语化
"""

        print("\n处理的文档数量:", len(context_parts))
        print("Context总长度:", len(context))
        print("Prompt总长度:", len(prompt))

        answer = generate_response_local(prompt)
        
        if not answer or answer.isspace():
            # 如果没有得到有效回答，尝试使用更简化的prompt
            simplified_prompt = f"""简要回答以下问题：{question}

参考要点：
{context[:1000]}"""
            answer = generate_response_local(simplified_prompt)

        # ... (评估部分代码保持不变) ...
        
        # print("\n生成的回答:", answer)  # 添加调试输出
        
        evaluation_results = evaluator.evaluate_response(question, answer, context)
        return answer, evaluation_results, matched_categories
        
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return "抱歉，生成回答时出现了错误。", None, None

# ---------------------------- 主函数 ----------------------------
def main():
    print("\n增强版政策咨询助手已启动...")
    print("正在初始化系统组件...")
    
    # 初始化组件
    index, metadata, category_embeddings = prepare_data(
        DATA_PATH, EMBEDDING_PATH, METADATA_PATH, 
        INDEX_PATH, CATEGORY_EMBEDDING_PATH
    )
    evaluator = ResponseEvaluator()
    
    if index is None or metadata is None or category_embeddings is None:
        print("系统初始化失败，程序退出。")
        return
    
    print("\n系统初始化完成！")
    print("支持的功能：")
    print("1. 智能分类匹配，提高搜索精确度")
    print("2. 多维度评估回答质量")
    print("3. 本地缓存向量，提升响应速度")
    print("\n请输入您的问题（输入 'exit' 退出）：")
    
    while True:
        question = input("\n您的问题：").strip()
        if question.lower() in ['exit', 'quit']:
            print("感谢使用！再见！")
            break
        if not question:
            print("请输入有效的问题。")
            continue
            
        print("\n处理中...")
        
        # 获取回答和评估结果
        answer, evaluation_results, matched_categories = generate_answer(
            question, index, metadata, category_embeddings, evaluator
        )
        
        # 评估结果显示在回答之后
        if evaluation_results:
            print("\n回答质量评估：")
            print(f"总体评分: {evaluation_results['overall_score']}/10")
            print(f"相关性评分: {evaluation_results['relevance_score']}/10")
            print(f"连贯性评分: {evaluation_results['coherence_score']}/10")
            print(f"完整性评分: {evaluation_results['completeness_score']}/10")
            print(f"引用规范评分: {evaluation_results['citation_score']}/10")
        
        if matched_categories:
            print("\n匹配到的文档类别：")
            print(", ".join(matched_categories))
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()