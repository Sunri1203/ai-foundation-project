from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Step 1: 原始文字
text = "Hello World!"
print("原始文字:", text)

# Step 2: 編碼 → 得到 BatchEncoding (dict-like)
enc = tok(text, return_tensors="pt")
print("input_ids:", enc["input_ids"])

# Step 3: ID 轉 token
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
print("對應 tokens:", tokens)

# Step 4: 解碼還原
decoded = tok.decode(enc["input_ids"][0])
print("還原文字:", decoded)