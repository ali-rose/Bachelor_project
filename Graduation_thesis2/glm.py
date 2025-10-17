from transformers import AutoTokenizer, AutoModel

model_path = "/root/chatglm3-6b/chatglm3-6b"
tokenizer_path = '/root/chatglm3-6b/chatglm3-6b'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)