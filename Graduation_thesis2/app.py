from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import pdfplumber
import faiss
from typing import List
import numpy as np

app = FastAPI()

# 模型和tokenizer的初始化
embedding_model_path = '/root/AiLi/ChatPDF-main/checkpoints'
embedding_tokenizer_path = '/root/AiLi/ChatPDF-main/checkpoints'
tokenizer = AutoTokenizer.from_pretrained(embedding_tokenizer_path)
embedding_model = AutoModelForCausalLM.from_pretrained(embedding_model_path).half().cuda()
embedding_model.eval()

llm_model_path = "/root/chatglm3-6b/chatglm3-6b"
llm_tokenizer_path = '/root/chatglm3-6b/chatglm3-6b'
llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path, trust_remote_code=True)
llm_model = AutoModel.from_pretrained(llm_model_path, trust_remote_code=True).half().cuda()
llm_model.eval()

VECTOR_SEARCH_TOP_K = 6

# 用于存储上传的PDF文档的文本和它们的索引
documents = []
index = None

# 路由来处理文件上传和文本提取
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        return {"message": "File type not supported."}

    try:
        with pdfplumber.open(file.file) as pdf:
            texts = [page.extract_text() for page in pdf.pages if page.extract_text()]
        global documents
        documents = texts

        # 向量化文档
        encoded_input = tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
            embeddings = model_output[0][:, 0]

        # 使用FAISS创建向量索引
        global index
        index = faiss.IndexFlatL2(embeddings.size(1))
        index.add(embeddings.cpu().numpy().astype('float32'))

        return {"message": "File uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 路由来处理查询
@app.post("/query/")
async def query_pdf(query: str = Form(...)):
    if index is None or not documents:
        raise HTTPException(status_code=400, detail="No indexed documents available.")
    
    query_encoded = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
    with torch.no_grad():
        query_embedding = embedding_model(**query_encoded)[0][:, 0]
    distances, indices = index.search(query_embedding.cpu().numpy().astype('float32'), VECTOR_SEARCH_TOP_K)
    related_docs = [documents[i] for i in indices[0]]
    prompt = ' '.join(related_docs)
    response, _ = llm_model.chat(llm_tokenizer, prompt, history=[])
    return {"response": response}

# 主页
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5082)
