# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("/root/autodl-tmp/glm4-9B/embeddings")

# sentences = [
#     "That is a happy person",
#     "That is a happy dog",
#     "That is a very happy person",
#     "Today is a sunny day",
#     "你好吗",
#     "我爱你"
# ]
# embeddings = model.encode(sentences)

# similarities = model.similarity(embeddings, embeddings)
# print(similarities.shape)
# # [4, 4]

import nltk
from nltk.tokenize import sent_tokenize


try:
    nltk.data.find('tokenizers/punkt')
    print("punkt 模型已安装。")
    nltk.data.find('tokenizers/punkt_tab')
    print("punkt_tab 模型已安装。")
except LookupError:
    print("punkt 模型未找到，需要下载。")


