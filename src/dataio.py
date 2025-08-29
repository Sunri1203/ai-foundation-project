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

#tokenizer 回傳的是 mapping（dict/BatchEncoding）
enc = Token("Greetings!" , return_tensors = "pt")

#BatchEncoding是Hugging Face transformers套件的特殊字典容器，後續可以進行.to("cuda")的操作
#keyView是所有Key的名稱，會打印出所有欄位
print(type(enc), enc.keys())

#input_ids': tensor([[    1, 15043,  2787, 29991]]) 對應token在詞彙表的位置，文字 → 字典編號
#跟向量空間不同: input_ids = 「字典編號」 embedding 向量 = 「字典頁面上的內容」
print(enc.keys())

#格式大小
print(enc["input_ids"].shape)
print(enc["attention_mask"].shape)

# ------------------------- dict → 丟進模型（特徵 or 生成）-------------------------

#只需要推理，不用訓練更新，所以這段表示停止梯度計算 > 停用梯度追蹤，省記憶體、加速
with torch.no_grad():
    #generate(**enc)需要dict，勿把tensor格是放入會報錯(因為不是mapping)
    outputs_no_grad = model.generate(**enc, max_new_tokens = 30)
print("outputs_no_grad:", Token.decode(outputs_no_grad[0], skip_special_tokens = True))

#no_grad()只關「要不要記錄梯度」
outputs_with_grad = model.generate(**enc, max_new_tokens = 30)
print("outputs_with_grad:", Token.decode(outputs_with_grad[0], skip_special_tokens = True))

#改變輸出需要使用到抽樣sampling
outputs_sampling = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.9, max_new_tokens=30)
print("outputs_sampling:",Token.decode(outputs_sampling[0], skip_special_tokens = True))


# ------------------------- 看清楚「Tensor」長什麼樣 -------------------------

input_ids = enc["input_ids"].to(model.device)
attention_mask = enc["attention_mask"].to(model.device)

with torch.no_grad():
    outputs = model.generate(input_ids = input_ids, attention_mask = attention_mask , max_new_tokens = 30)

print(Token.decode(outputs[0], skip_speical_tokens = False))

# ------------------------- 多輪對話 -------------------------

#定義Prompt 與 question
message = [
    {
        "role": "system" , "content" : "You are a inteligent AI assistant."
    },
    {
        "role" : "user", "content" : "Greeting how are you?" 
    }
]

print(message)

#apply_chat_temple會把多輪對話轉成模型可懂的字串
prompt_text = Token.apply_chat_template(
    message, tokenize = False, add_generation_prompt = True
)

print(prompt_text)

enc_chat = Token(prompt_text, return_tensors = "pt").to(model.device)

enc_chat["input_ids"].shape
enc_chat["input_ids"].shape[1]


with torch.no_grad():
    outputs_chat = model.generate(**enc_chat, max_new_tokens = 80)

print(outputs_chat)
print(outputs_chat.shape)
print(Token.decode(outputs_chat[0], skip_special_token = True))

#54整體 - 42輸入問題input_ids = 模型回答
#原本資料 [ [prompt_tokens | answer_tokens] ]
#取 [0]:       [prompt_tokens | answer_tokens]
#取 [len(prompt):]:                [answer_tokens]
#.shape[x] 第幾欄位的長度 Ex : tensor[1 , 4] : shape[0] > 1 ,shape[1] > 4
take_outputs = outputs_chat[0, enc_chat["input_ids"].shape[1]:]
print(take_outputs.shape)
print(Token.decode(take_outputs , skip_special_tokens= True))