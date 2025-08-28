from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
Token = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

print(Token)
print(model)

text = "你好世界"

token_slow_list = Token(text)
print(token_slow_list)

token_slow = {
    "input_ids": torch.tensor([token_slow_list.input_ids]),
    "attention_mask" : torch.tensor([token_slow_list.attention_mask])
}

print("token_slow:", token_slow)
token_fast = Token(text , return_tensors = "pt")
# return_tensors="pt" 是告訴 tokenizer：把輸出資料直接轉成 PyTorch tensor 格式
# 其他格式"tf"：TensorFlow tensor ,"np"：NumPy array
print("token_fast:", token_fast)