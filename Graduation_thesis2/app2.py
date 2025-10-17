from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import pdfplumber
import faiss
from typing import List
import numpy as np
import spacy

app = FastAPI()

nlp = spacy.load("zh_core_web_sm")

# 模型和tokenizer的初始化
embedding_model_path = '/root/AiLi/ChatPDF-main/checkpoints'
embedding_tokenizer_path = '/root/AiLi/ChatPDF-main/checkpoints'
tokenizer = AutoTokenizer.from_pretrained(embedding_tokenizer_path)
embedding_model = AutoModelForCausalLM.from_pretrained(embedding_model_path).half().cuda(0)
embedding_model.eval()

llm_model_path = "/root/chatglm3-6b/chatglm3-6b"
llm_tokenizer_path = '/root/chatglm3-6b/chatglm3-6b'
llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path, trust_remote_code=True)
llm_model = AutoModel.from_pretrained(llm_model_path, trust_remote_code=True).half().cuda(0)
llm_model.eval()

VECTOR_SEARCH_TOP_K = 6

# 用于存储上传的PDF文档的文本和它们的索引
documents = []
index = None
window_size = 512
step_size = 256

sentence_embeddings_list = []

# 路由来处理文件上传和文本提取
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        return {"message": "File type not supported."}

    # 重置全局变量
    global documents, sentence_embeddings_list, index
    documents = []
    sentence_embeddings_list = []
    index = None

    try:
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    tokenized_text = tokenizer.tokenize(text)
                    for i in range(0, len(tokenized_text), step_size):
                        segment = tokenizer.convert_tokens_to_string(tokenized_text[i:i + window_size])
                        documents.append(text)
                        encoded_input = tokenizer.encode_plus(segment, padding='max_length', truncation=True, max_length=window_size, return_tensors='pt').to('cuda:0')
                        with torch.no_grad():
                            model_output = embedding_model(**encoded_input)
                            sentence_embedding = model_output[0][:, 0]  # 使用CLS token的输出作为句子嵌入
                            sentence_embeddings_list.append(sentence_embedding)
    
        if sentence_embeddings_list:
            sentence_embeddings = torch.cat(sentence_embeddings_list, dim=0)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            # 使用FAISS创建向量索引
            index = faiss.IndexFlatL2(sentence_embeddings.size(1))
            index.add(sentence_embeddings.cpu().numpy().astype('float32'))

        return {"message": "文档读取成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 路由来处理查询
@app.post("/query/")
async def query_pdf(query: str = Form(...)):
    if index is None or not documents:
        raise HTTPException(status_code=400, detail="No indexed documents available.")
    
    query_encoded = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda:0')
    with torch.no_grad():
        query_embedding = embedding_model(**query_encoded)[0][:, 0]
    distances, indices = index.search(query_embedding.cpu().numpy().astype('float32'), VECTOR_SEARCH_TOP_K)
    related_docs = [documents[i] for i in indices[0]]
    context = ' '.join(related_docs)  # 假设我们直接将相关文档作为context

    # 如果提取的内容过长，截断以适应模型的输入限制
    if len(context) > 8192:
        context = context[:8192]
    # 构建prompt，确保格式和变量正确替换
    prompt_template = f"""
    <指令>仅根据提供的已知信息回答问题。如果提供的信息与问题无关，或信息不足以形成答案，请明确回答“根据已知信息无法回答该问题”。不允许使用模型自身的知识库或任何外部信息。所有回答必须完全基于已提供的文本内容。答案请使用中文。</指令>
    <已知信息>{context}</已知信息>
    <问题>{query}</问题>
    """
    
    response, _ = llm_model.chat(llm_tokenizer, prompt_template, history=[])

    #判别器
    
    doc = nlp(response)
    keywords = set()
    for token in doc:
        # 检查中文词性标签，这里假设`NOUN`、`PROPN`、`VERB`适用于中文
        if token.pos_ in ['NOUN', 'PROPN', 'VERB']:
            keywords.add(token.text)
            
    context_doc = nlp(context)
    context_words = {token.text for token in context_doc}
    
    # 从关键词中筛选出名词和专有名词
    noun_keywords = {kw for kw in keywords if nlp(kw)[0].pos_ in ['NOUN', 'PROPN']}
    
    # 计算匹配的名词和专有名词的数量
    matched_keywords = {kw for kw in noun_keywords if kw in context_words}
    
    # 检查匹配的关键词是否至少是名词关键词的50%
    
    matcher=len(matched_keywords) >= 0.3 * len(noun_keywords)
    
    if matcher:
        return {"response": response}  # 如果所有关键词都在上下文中，返回原始答案
    else:
        return {"response": "根据文章信息无法回答该问题"}  # 如果任何关键词不在上下文中，修改答案

# 主页
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index3.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5082)
