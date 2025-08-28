from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
Token = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


# ------------------------- 模型檢查 -------------------------

#打印模型與Token格式
print(Token)
print(model)

#參數量11億，符合1.1B
model.num_parameters()

#模型設定
print(model.config)


# ------------------------- 文字 > tokenizer -------------------------

enc = Token("Hello World!" , return_tensor = "pt")


