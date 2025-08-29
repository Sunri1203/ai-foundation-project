from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "Greeting!"

enc = tok(prompt, return_tensors="pt").to(model.device)
#print(type(enc))
#print(enc.keys())
# 固定 seed，確保重現
set_seed(41)

out = model.generate(
    **enc,
    max_new_tokens=50,
    do_sample=True,     # 啟用抽樣模式
    temperature=0.7,    # 控制隨機性
    top_p=0.9           # nucleus sampling
)

print(tok.decode(out[0], skip_special_tokens=True))
