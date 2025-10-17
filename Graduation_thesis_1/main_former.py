import pandas as pd
import numpy as np
import json
import re
import os
import torch
import faiss
from threading import Thread
import signal
import sys
import atexit
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from rouge_chinese import Rouge
import jieba
from datetime import datetime

# 下载必要的NLTK数据
nltk.download('punkt', quiet=True)

# -------------------------------- 配置 --------------------------------
MODEL_PATH = "/root/autodl-tmp/glm4-9B/qwen2.5-7B"  # 生成模型路径
EMBEDDING_MODEL_NAME = "/root/autodl-tmp/glm4-9B/embeddings"  # 嵌入模型路径

# 数据路径
STRUCTURED_DATA_PATH = 'structured_data.xlsx'
SEMI_STRUCTURED_DATA_PATH = 'semi_structured_data.json'
UNSTRUCTURED_DATA_PATH = 'unstructured_data.txt'

# 每种数据格式的缓存路径
CACHE_DIR = 'embedding_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# 特定数据格式的缓存路径
FORMAT_PATHS = {
    'structured': {
        'embedding': os.path.join(CACHE_DIR, 'structured_embeddings.npy'),
        'metadata': os.path.join(CACHE_DIR, 'structured_metadata.pkl'),
        'index': os.path.join(CACHE_DIR, 'structured_faiss.index')
    },
    'semi_structured': {
        'embedding': os.path.join(CACHE_DIR, 'semi_structured_embeddings.npy'),
        'metadata': os.path.join(CACHE_DIR, 'semi_structured_metadata.pkl'),
        'index': os.path.join(CACHE_DIR, 'semi_structured_faiss.index')
    },
    'unstructured': {
        'embedding': os.path.join(CACHE_DIR, 'unstructured_embeddings.npy'),
        'metadata': os.path.join(CACHE_DIR, 'unstructured_metadata.pkl'),
        'index': os.path.join(CACHE_DIR, 'unstructured_faiss.index')
    }
}

TOP_K = 5  # 检索文档数量
CHUNK_SIZE = 8192  # 文本块大小

# 添加结果收集变量
RESULTS_EXCEL_PATH = 'rag_evaluation_results.xlsx'
evaluation_results = []  # 存储所有测试结果

# -------------------------------- 模型加载 --------------------------------

def configure_model_generation(model_name):
    """根据模型名称配置适当的生成参数"""
    if "qwen2.5-7B" in model_name:
        return {
            "do_sample": True,
            "top_p": 0.7,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
        }
    elif "DeepSeek-r1" in model_name:
        return {
            "do_sample": True,
            "top_p": 0.7,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
        }
    else:
        # 默认配置
        return {
            "do_sample": True,
            "top_p": 0.7,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
        }

def format_prompt_for_model(prompt, model_name, tokenizer):
    """根据模型类型格式化提示"""
    if any(name in model_name.lower() for name in ["qwen", "deepseek", "llama", "mistral"]):
        # 使用聊天模板
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # 如果apply_chat_template失败，回退到基本格式
            return prompt
    else:
        # 默认情况下直接返回原始提示
        return prompt
# 加载生成模型
print("正在加载语言模型...")
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
            # 如果是整数，直接检查
            return input_ids[0][-1] == stop_ids
        elif isinstance(stop_ids, list):
            # 如果是列表，遍历检查
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
        return False

stop = StopOnTokens()
print("正在加载嵌入模型...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# -------------------------------- 评估指标 --------------------------------
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
            # 使用jieba分词
            def tokenize(text):
                return list(jieba.cut(text))

            answer_tokens = tokenize(answer)
            context_tokens = tokenize(context)

            # 计算词汇重叠
            overlap_tokens = set(answer_tokens) & set(context_tokens)
            overlap_ratio = len(overlap_tokens) / len(set(context_tokens))
            
            return min(overlap_ratio * 10, 10)
        except Exception as e:
            print(f"计算完整性分数时出错: {e}")
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

# -------------------------------- 文本处理函数 --------------------------------
def split_text_into_chunks(text, chunk_size):
    if not isinstance(text, str):
        text = str(text)
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

# -------------------------------- 嵌入生成 --------------------------------
def get_embeddings(text):
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
        print(f"生成嵌入时出错: {e}")
        return np.zeros(embedding_model.get_sentence_embedding_dimension(), dtype='float32')

# -------------------------------- 答案生成 --------------------------------
def generate_response(prompt):
    try:
        # 获取模型名称（从MODEL_PATH中提取）
        model_name = os.path.basename(MODEL_PATH)
        
        # 格式化提示以适应不同模型
        formatted_prompt = format_prompt_for_model(prompt, model_name, tokenizer)
        
        # 编码输入
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # 获取针对当前模型的生成参数
        gen_params = configure_model_generation(model_name)
        
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones((1, input_ids.shape[1])).to(model.device),
            "streamer": streamer,
            "max_new_tokens": 8192,
            "stopping_criteria": StoppingCriteriaList([stop]),
            **gen_params  # 添加模型特定的参数
        }
        
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        response = ""
        for new_token in streamer:
            if new_token:
                response += new_token
        
        torch.cuda.empty_cache()
        return response.strip()
    except Exception as e:
        print(f"生成回复时出错: {e}")
        print(f"错误类型: {type(e)}")
        print(f"错误详情: {str(e)}")
        import traceback
        traceback.print_exc()
        return "抱歉，生成回复时出现错误。"

# -------------------------------- 数据加载器 --------------------------------
def load_structured_data():
    try:
        print("正在加载结构化数据...")
        df = pd.read_excel(STRUCTURED_DATA_PATH)
        metadata = []
        
        for idx, row in df.iterrows():
            metadata.append({
                'id': row.get('政策ID', f'S{idx}'),
                'title': row.get('政策标题', ''),
                'date': row.get('发布日期', ''),
                'authority': row.get('发文机关', ''),
                'category': row.get('政策领域', ''),
                'url': row.get('政策链接', ''),
                'status': row.get('生效状态', ''),
                'content': row.get('主要措施', '')
            })
        
        return metadata
    except Exception as e:
        print(f"加载结构化数据时出错: {e}")
        return []

def load_semi_structured_data():
    try:
        print("正在加载半结构化数据...")
        with open(SEMI_STRUCTURED_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = []
        for doc in data:
            # 根据JSON结构提取内容
            content = ""
            if 'content' in doc:
                if 'key_points' in doc['content'] and doc['content']['key_points']:
                    content += "\n".join(doc['content']['key_points'])
                
                if 'raw_text' in doc['content'] and doc['content']['raw_text']:
                    content += "\n" + doc['content']['raw_text']
            
            metadata.append({
                'id': doc.get('document_id', ''),
                'title': doc.get('metadata', {}).get('title', ''),
                'date': doc.get('metadata', {}).get('publish_date', ''),
                'authority': doc.get('metadata', {}).get('issuing_authority', ''),
                'category': doc.get('metadata', {}).get('policy_category', ''),
                'url': doc.get('metadata', {}).get('url', ''),
                'status': doc.get('content', {}).get('status', ''),
                'content': content
            })
        
        return metadata
    except Exception as e:
        print(f"加载半结构化数据时出错: {e}")
        return []

def load_unstructured_data():
    try:
        print("正在加载非结构化数据...")
        with open(UNSTRUCTURED_DATA_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 按文档分隔符分割
        docs = re.split(r'-{80,}', text)
        metadata = []
        
        for doc in docs:
            if not doc.strip():
                continue
                
            # 使用正则表达式提取信息
            doc_id_match = re.search(r'\[文档ID: (\d+)\]', doc)
            title_match = re.search(r'标题: (.*)', doc)
            date_match = re.search(r'发布时间: (.*)', doc)
            authority_match = re.search(r'发文机关: (.*)', doc)
            category_match = re.search(r'主题类别: (.*)', doc)
            url_match = re.search(r'网址链接: (.*)', doc)
            
            # 提取正文内容 - "正文内容:"之后的所有内容
            content_match = re.search(r'正文内容:([\s\S]*)', doc)
            
            metadata.append({
                'id': doc_id_match.group(1) if doc_id_match else '',
                'title': title_match.group(1) if title_match else '',
                'date': date_match.group(1) if date_match else '',
                'authority': authority_match.group(1) if authority_match else '',
                'category': category_match.group(1) if category_match else '',
                'url': url_match.group(1) if url_match else '',
                'status': '',  # 非结构化格式中不可用
                'content': content_match.group(1).strip() if content_match else ''
            })
        
        return metadata
    except Exception as e:
        print(f"加载非结构化数据时出错: {e}")
        return []

# -------------------------------- 索引准备 --------------------------------
def prepare_data_and_index(data_format):
    try:
        paths = FORMAT_PATHS[data_format]
        
        # 检查缓存的嵌入和索引是否都存在
        if (os.path.exists(paths['embedding']) and os.path.exists(paths['metadata']) and os.path.exists(paths['index'])):
            print(f"正在加载{data_format}数据的缓存嵌入和索引...")
            embeddings = np.load(paths['embedding'])
            with open(paths['metadata'], 'rb') as f:
                import pickle
                metadata = pickle.load(f)
            index = faiss.read_index(paths['index'])
            return index, metadata
        
        # 如果缓存不存在，基于格式加载数据
        if data_format == 'structured':
            metadata = load_structured_data()
        elif data_format == 'semi_structured':
            metadata = load_semi_structured_data()
        else:  # unstructured
            metadata = load_unstructured_data()
        
        if not metadata:
            print(f"没有找到{data_format}数据。")
            return None, None
        
        print(f"为{data_format}数据生成新的嵌入和索引...")
        embeddings = []
        
        for idx, doc in enumerate(metadata):
            # 使用content字段作为嵌入源
            content = doc['content']
            embedding = get_embeddings(content)
            embeddings.append(embedding)
            
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1} 个文档")
        
        embeddings = np.array(embeddings).astype('float32')
        
        # 保存嵌入和元数据
        np.save(paths['embedding'], embeddings)
        with open(paths['metadata'], 'wb') as f:
            import pickle
            pickle.dump(metadata, f)
        
        # 创建并保存FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, paths['index'])
        
        return index, metadata
    except Exception as e:
        print(f"准备{data_format}数据时出错: {e}")
        return None, None

# -------------------------------- RAG流程 --------------------------------
def retrieve_relevant_docs(question, index, metadata, top_k=TOP_K):
    try:
        # 生成问题嵌入
        question_embedding = get_embeddings(question)
        question_vector = np.array([question_embedding]).astype('float32')
        
        # 搜索相似文档
        distances, indices = index.search(question_vector, min(top_k, index.ntotal))
        
        # 获取检索到的文档
        retrieved_docs = [metadata[idx] for idx in indices[0]]
        
        return retrieved_docs
    except Exception as e:
        print(f"检索文档时出错: {e}")
        return []

def format_context(retrieved_docs, data_format):
    try:
        context_parts = []
        total_length = 0
        max_content_length = 600  # 每个文档的最大长度
        max_total_length = 3000   # 最大总上下文长度
        
        for doc in retrieved_docs:
            # 根据数据格式进行格式化
            if data_format == 'structured':
                doc_context = f"文档：《{doc['title']}》\n发布日期：{doc['date']}\n发文机关：{doc['authority']}\n政策领域：{doc['category']}\n要点：{doc['content'][:max_content_length]}\n链接：{doc['url']}\n"
            elif data_format == 'semi_structured':
                doc_context = f"文档：《{doc['title']}》\n发布日期：{doc['date']}\n发文机关：{doc['authority']}\n政策领域：{doc['category']}\n状态：{doc['status']}\n要点：{doc['content'][:max_content_length]}\n链接：{doc['url']}\n"
            else:  # unstructured
                doc_context = f"文档：《{doc['title']}》\n发布日期：{doc['date']}\n发文机关：{doc['authority']}\n要点：{doc['content'][:max_content_length]}\n链接：{doc['url']}\n"
            
            # 检查总长度
            if total_length + len(doc_context) > max_total_length:
                break
            
            context_parts.append(doc_context)
            total_length += len(doc_context)
        
        return "\n".join(context_parts)
    except Exception as e:
        print(f"格式化上下文时出错: {e}")
        return ""

def generate_answer_with_format(question, data_format):
    try:
        # 准备数据并检索文档
        index, metadata = prepare_data_and_index(data_format)
        if not index or not metadata:
            return f"准备{data_format}数据索引失败。", None
        
        # 获取相关文档并格式化上下文
        retrieved_docs = retrieve_relevant_docs(question, index, metadata)
        context = format_context(retrieved_docs, data_format)
        
        # 使用通用模板
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
        
        # 生成回答
        answer = generate_response(prompt)
        
        # 评估回复
        evaluator = ResponseEvaluator()
        evaluation_results = evaluator.evaluate_response(question, answer, context)
        
        return answer, evaluation_results
    except Exception as e:
        print(f"使用{data_format}格式生成答案时出错: {e}")
        return f"使用{data_format}格式生成答案时出错: {e}", None

# -------------------------------- 测试和比较函数 --------------------------------
def test_data_formats(question, formats=['structured', 'semi_structured', 'unstructured']):
    global evaluation_results
    results = {}
    
    print(f"\n测试问题: {question}")
    print("-" * 50)
    
    # 创建当前测试的结果记录
    test_record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question
    }
    
    for data_format in formats:
        print(f"\n使用{data_format}数据格式处理...")
        answer, evaluation = generate_answer_with_format(question, data_format)
        
        results[data_format] = {
            'answer': answer,
            'evaluation': evaluation
        }
        
        print(f"\n{data_format.upper()}回答:")
        print("-" * 30)
        print(answer)
        
        if evaluation:
            print("\n评估:")
            print(f"总体得分: {evaluation['overall_score']}/10")
            print(f"相关性: {evaluation['relevance_score']}/10")
            print(f"连贯性: {evaluation['coherence_score']}/10")
            print(f"完整性: {evaluation['completeness_score']}/10")
            print(f"引用: {evaluation['citation_score']}/10")
            
            # 将评估结果添加到测试记录
            test_record[f'{data_format}_overall'] = evaluation['overall_score']
            test_record[f'{data_format}_relevance'] = evaluation['relevance_score']
            test_record[f'{data_format}_coherence'] = evaluation['coherence_score']
            test_record[f'{data_format}_completeness'] = evaluation['completeness_score']
            test_record[f'{data_format}_citation'] = evaluation['citation_score']
        
        print("-" * 50)
    
    # 比较结果
    if all(results[fmt]['evaluation'] for fmt in formats):
        print("\n分数比较:")
        print("-" * 30)
        print(f"{'格式':<15} {'总体':<10} {'相关性':<10} {'连贯性':<10} {'完整性':<10} {'引用':<10}")
        print("-" * 70)
        
        for fmt in formats:
            eval_data = results[fmt]['evaluation']
            print(f"{fmt:<15} {eval_data['overall_score']:<10} {eval_data['relevance_score']:<10} "
                  f"{eval_data['coherence_score']:<10} {eval_data['completeness_score']:<10} {eval_data['citation_score']:<10}")
    
    # 添加记录到结果列表
    evaluation_results.append(test_record)
    
    # 每次测试完成后都保存一次结果，确保不会丢失数据
    save_results_to_excel()
    
    return results

# -------------------------------- 结果保存函数 --------------------------------
def save_results_to_excel():
    """将评估结果保存到Excel文件"""
    global evaluation_results
    
    try:
        if evaluation_results:
            df = pd.DataFrame(evaluation_results)
            df.to_excel(RESULTS_EXCEL_PATH, index=False)
            print(f"结果已保存到 {RESULTS_EXCEL_PATH}")
    except Exception as e:
        print(f"保存结果到Excel时出错: {e}")

# 注册退出处理函数
def exit_handler():
    print("\n程序退出，正在保存最终结果...")
    save_results_to_excel()

atexit.register(exit_handler)

# 注册信号处理器（用于捕获CTRL+C等终止信号）
def signal_handler(sig, frame):
    print("\n接收到终止信号，正在保存结果并退出...")
    save_results_to_excel()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# -------------------------------- 主函数 --------------------------------
def main():
    global evaluation_results
    
    print("\nRAG数据格式比较系统已启动...")
    
    # 尝试加载现有的评估结果
    try:
        if os.path.exists(RESULTS_EXCEL_PATH):
            existing_df = pd.read_excel(RESULTS_EXCEL_PATH)
            evaluation_results = existing_df.to_dict('records')
            print(f"已加载 {len(evaluation_results)} 条现有评估记录")
    except Exception as e:
        print(f"加载现有评估结果时出错: {e}")
        evaluation_results = []
    
    # 检查数据文件是否存在
    missing_files = []
    if not os.path.exists(STRUCTURED_DATA_PATH):
        missing_files.append(STRUCTURED_DATA_PATH)
    if not os.path.exists(SEMI_STRUCTURED_DATA_PATH):
        missing_files.append(SEMI_STRUCTURED_DATA_PATH)
    if not os.path.exists(UNSTRUCTURED_DATA_PATH):
        missing_files.append(UNSTRUCTURED_DATA_PATH)
    
    if missing_files:
        print(f"警告: 以下数据文件缺失: {', '.join(missing_files)}")
        print("只会测试可用的数据格式。")
    
    # 确定可用的格式
    available_formats = []
    if os.path.exists(STRUCTURED_DATA_PATH):
        available_formats.append('structured')
    if os.path.exists(SEMI_STRUCTURED_DATA_PATH):
        available_formats.append('semi_structured')
    if os.path.exists(UNSTRUCTURED_DATA_PATH):
        available_formats.append('unstructured')
    
    if not available_formats:
        print("未找到数据文件。请确保数据文件存在。")
        return
    
    print(f"可用的数据格式: {', '.join(available_formats)}")
    
    try:
        while True:
            print("\n选项:")
            print("1. 使用自定义问题测试")
            print("2. 运行标准测试问题")
            print("3. 退出")
            
            choice = input("选择一个选项 (1-3): ").strip()
            
            if choice == '1':
                question = input("\n输入您的问题: ").strip()
                if question:
                    test_data_formats(question, available_formats)
                else:
                    print("问题不能为空。")
            
            elif choice == '2':
                # 用于比较格式的标准测试问题
                test_questions = [                   
                    "教育政策如何平衡素质教育与应试教育，以适应新时代人才培养需求？",
                    "政府对农村教育信息化建设的政策投入与资源分配机制有哪些？",
                    "教育政策在促进教育公平方面，针对弱势群体的特殊扶持政策有哪些？",
                    "职业教育政策在推动企业深度参与人才培养过程中的激励机制有哪些？",
                    "高校 “强基计划” 相关政策的实施效果评估与改进方向有哪些？",
                    "教育政策在鼓励教师创新教学方法与教学模式方面的支持措施有哪些？",
                    "教育政策在促进家校共育方面的引导与规范措施有哪些？",
                    "教育政策在应对在线教育快速发展带来的挑战与机遇方面有哪些举措？",
                    "教育政策在推动教育国际化过程中，如何保障国家教育主权与文化安全？",
                    "教育政策在促进学生身心健康发展方面的具体措施与监督机制有哪些？",
                    "知识产权质押融资风险补偿政策的运行机制与效果评估标准有哪些？",
                    "政府对知识产权密集型产业的政策扶持与培育措施有哪些？",
                    "知识产权政策在促进高校、科研机构与企业知识产权协同创新方面有哪些举措？",
                    "知识产权维权援助政策的服务范围与工作流程有哪些？",
                    "知识产权运营服务体系建设的政策支持与资金投入机制有哪些？",
                    "知识产权政策在应对知识产权海外纠纷方面的援助与指导措施有哪些？",
                    "政府对知识产权服务业发展的政策引导与规范管理有哪些？",
                    "知识产权政策在促进地理标志产品保护与发展方面的措施有哪些？",
                    "知识产权政策在激励企业加大研发投入与知识产权创造方面的作用机制有哪些？",
                    "知识产权政策在推动知识产权与金融深度融合方面的创新模式有哪些？",
                    "科技政策在引导企业加大基础研究投入方面有哪些激励措施？",
                    "政府对新型研发机构建设与发展的政策支持与管理机制有哪些？",
                    "科技政策在促进科技成果转化为现实生产力方面的关键环节政策有哪些？",
                    "科技人才评价机制改革政策的主要内容与实施效果评估有哪些？",
                    "科技政策在推动区域科技协同创新方面的政策协同与资源共享机制有哪些？",
                    "科技政策在应对科技伦理问题方面的监管措施与引导机制有哪些？",
                    "政府对科技金融产品创新的政策支持与风险防控措施有哪些？",
                    "科技政策在促进科技资源开放共享方面的政策要求与监督机制有哪些？",
                    "科技政策在鼓励企业开展国际科技合作与竞争方面的政策支持有哪些？",
                    "科技政策在推动科技服务乡村振兴战略方面的具体举措与实施效果评估有哪些？",
                    "信息产业政策在推动 5G 技术广泛应用和产业融合发展方面，有哪些扶持措施？",
                    "电信普遍服务政策在扩大农村和偏远地区网络覆盖方面，有哪些新的推进计划？",
                    "信息产业创新政策在鼓励芯片研发和关键核心技术突破方面，有哪些资金支持和人才培养机制？",
                    "网络安全政策在保障信息产业安全和用户数据安全方面，有哪些监管措施和技术手段？",
                    "信息产业发展政策在促进软件和信息技术服务业发展方面，有哪些税收优惠和产业引导？",
                    "电信市场监管政策在规范市场竞争秩序和保障消费者权益方面，有哪些新规定？",
                    "信息产业政策在推动数字经济与实体经济深度融合方面，有哪些示范项目和实施路径？",
                    "物联网产业发展政策在标准制定和应用场景拓展方面，有哪些政策支持和创新举措？",
                    "信息产业政策在应对国际信息技术竞争和技术封锁方面，有哪些应对策略和自主创新计划？",
                    "电信基础设施建设政策在优化网络布局和提高网络性能方面，有哪些新的规划和投资重点？",
                    "公路建设政策在促进农村公路高质量发展和助力乡村振兴方面，有哪些新的资金投入和建设标准？",
                    "公路养护管理政策在提高公路使用寿命和服务质量方面，有哪些技术创新和管理模式？",
                    "高速公路收费政策在优化收费标准和收费期限方面，有哪些调整思路和改革方向？",
                    "公路运输政策在促进货运物流降本增效和提升运输效率方面，有哪些政策措施？",
                    "公路交通安全政策在加强事故预防和应急救援方面，有哪些新的技术应用和管理机制？",
                    "智慧公路建设政策在推进公路信息化和智能化发展方面，有哪些关键技术和示范项目？",
                    "公路建设市场监管政策在规范招投标行为和保障工程质量方面，有哪些强化措施？",
                    "公路产业政策在带动相关产业发展和促进区域经济增长方面，有哪些协同发展策略？",
                    "公路政策在应对新能源汽车普及对公路基础设施需求变化方面，有哪些适应性调整？",
                    "公路建设与生态环境保护协调政策在减少公路建设对生态环境影响方面，有哪些具体措施？",
                    "综合交通运输政策在加强不同运输方式衔接和多式联运发展方面，有哪些协调机制和政策支持？",
                    "交通枢纽建设政策在提升枢纽功能和服务水平方面，有哪些规划和建设重点？",
                    "交通运输新业态发展政策在规范网约车、共享单车等行业方面，有哪些监管措施？",
                    "工业产业结构调整政策在培育新兴工业产业集群和推动传统产业转型升级方面，有哪些举措？",
                    "交通节能减排政策在推广新能源交通工具和提高交通能源利用效率方面，有哪些目标和行动？",
                    "工业绿色发展政策在加强工业污染防治和资源循环利用方面，有哪些技术创新和政策引导？",
                    "交通运输安全监管政策在强化安全风险防控和事故责任追究方面，有哪些新的要求？",
                    "工业技术创新政策在促进工业企业自主创新和提高核心竞争力方面，有哪些资金扶持和平台建设？",
                    "交通基础设施建设政策在应对城市交通拥堵和提高交通承载能力方面，有哪些新思路？",
                    "工业和交通领域政策在促进区域协调发展和缩小地区差距方面，如何协同推进？",
                    "我国针对机械制造与重工业的产业扶持政策有哪些？",
                    "机械制造与重工业领域的环保政策是如何具体实施的？",
                    "政府对机械制造与重工业的科技创新政策支持有哪些？",
                    "机械制造与重工业在税收政策上有哪些优惠？",
                    "促进机械制造与重工业智能化发展的政策有哪些？",
                    "目前机械制造与重工业的人才培养政策是怎样的？",
                    "机械制造与重工业的进出口政策有哪些调整？",
                    "针对机械制造与重工业的节能减排政策有哪些具体要求？",
                    "地方政府对机械制造与重工业的招商引资政策有哪些？",
                    "机械制造与重工业在产业结构调整方面的政策导向是什么？",
                    "民航业的航线审批政策是怎样的？",
                    "政府对民航业的补贴政策有哪些？",
                    "民航业在安全监管方面的政策有哪些？",
                    "促进民航业绿色发展的政策措施有哪些？",
                    "民航业的机场建设政策有哪些要点？",
                    "针对民航业的人才吸引政策有哪些？",
                    "民航业的国际合作政策有哪些规定？",
                    "民航业在应对突发事件时的政策保障有哪些？",
                    "政府对民航业新技术应用的政策支持有哪些？",
                    "民航业在票务价格方面的政策调控是怎样的？",
                    "邮政业的普遍服务政策有哪些具体内容？",
                    "邮政业在快递业务监管方面的政策有哪些？",
                    "政府对邮政业的基础设施建设政策有哪些？",
                    "邮政业的绿色包装政策有哪些要求？",
                    "促进邮政业与电商协同发展的政策有哪些？",
                    "邮政业的税收优惠政策有哪些？",
                    "邮政业在跨境业务方面的政策有哪些？",
                    "邮政业的人才培养政策是怎样的？",
                    "政府对邮政业科技创新的政策支持有哪些？",
                    "邮政业在乡村服务方面的政策举措有哪些？",
                    "铁路建设的投融资政策有哪些？",
                    "铁路运输的价格调控政策是怎样的？",
                    "铁路行业的安全管理政策有哪些？",
                    "促进铁路智能化发展的政策有哪些？",
                    "铁路在环保方面的政策措施有哪些？",
                    "政府对铁路科技创新的政策支持有哪些？",
                    "铁路行业的人才发展政策是怎样的？",
                    "铁路在国际合作方面的政策有哪些？",
                    "铁路建设的土地政策有哪些要点？",
                    "铁路运输在保障民生方面的政策举措有哪些？"
                ]
                
                for question in test_questions:
                    test_data_formats(question, available_formats)
            
            elif choice == '3':
                print("退出系统。谢谢！")
                break
            
            else:
                print("无效选择。请选择有效的选项。")
    finally:
        # 确保退出前保存结果
        save_results_to_excel()

if __name__ == "__main__":
    main()