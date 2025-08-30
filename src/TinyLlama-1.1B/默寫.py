from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
Token = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def render_chat(messages):
    # 自定義、簡單、穩定的模板（示例）
    # <|system|> … <|user|> … <|assistant|>
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        parts.append(f"<|{role}|>\n{content}\n")
    parts.append("<|assistant|>\n")  # 告訴模型從這裡開始續寫
    return "".join(parts)


message_contain = [
    {
        "role": "system" , "content" : "You are a inteligate assistant!"
    },
    {
        "role" : "user" , "content" : "Greeting, my friend!"
    }
]
#print(message_contain)

message_form = render_chat(message_contain)
#print(message_form)

token_in = Token(message_form, return_tensors = "pt").to(model.device)
print(token_in)
#print(type(token_in))
#print(token_in.input_ids.shape)

with torch.no_grad():
    outputs = model.generate(**token_in, max_new_tokens = 80)
print(outputs)
print(outputs.shape)

print(Token.decode(outputs[0] , skip_special_tokens = True))

#今天的心得
#核心突破點：注意到 generate() 需要的是 input_ids 和 attention_mask → 才能正確推理
#with torch.no_grad()，省顯存、避免梯度追蹤