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
    outputs = model.generate(**enc, max_new_tokens = 30)
print(Token.decode(outputs[0], skip_special_tokens = True))

outputs_with_grad = model.generate(**enc, max_new_tokens = 30)
print(Token.decode(outputs_with_grad[0], skip_special_tokens = True))

#no_grad()只關「要不要記錄梯度」

它不會改變前向的數值或生成策略，只是省顯存、加速。

所以在推理時，輸出本來就應該一樣；不同的是資源占用。

你現在的生成是確定性的

generate() 的預設是 greedy decoding（不抽樣）：每步都選機率最大的 token。

確定性的演算法，給相同輸入、相同模型狀態 → 輸出必定相同。

想看到「會變」：改用抽樣