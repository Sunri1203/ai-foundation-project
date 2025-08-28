# ai-foundation-project
Using TinyLlama test data struction and performation.

# AI Foundation Project – LLM & ComfyUI Starter

本專案同時提供 **Jupyter 筆記（教學/展示）** 與 **模組化 Python 程式（重用/專案化）**：
- `notebooks/`：從 tokenizer → dict/tensor → generate → decode 的逐步筆記，含 GPU/CPU 基準測試與 ComfyUI workflow 截圖。
- `src/`：封裝載入模型、對話生成、可視化等常用函式，方便在 notebook 與腳本間重用。

## ✨ Features
- TinyLlama / Qwen / Mistral 的最小可行推理流程
- Chat template（HF `apply_chat_template`）正確用法
- GPU vs CPU 矩陣乘法/推理小型 benchmark
- 可重現的實驗結構（configs / logs 可選）

## 🚀 Quickstart
```bash
# 1) 建議使用 conda 環境
conda create -n ai-core python=3.11 -y
conda activate ai-core

# 2) 安裝套件
pip install -r requirements.txt

# 3) 跑最小測試（TinyLlama 生成）
python - << 'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
mid = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, device_map="auto")
messages = [
    {"role":"system","content":"You are an intelligent AI assistant."},
    {"role":"user","content":"Say hi in one short sentence."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=40)
print(tok.decode(out[0], skip_special_tokens=True))
PY


## 📦 專案結構
<details>
<summary>展開結構樹</summary>
ai-foundation-project/
├─ notebooks/
│ ├─ 01_tokenizer_intro.ipynb
│ ├─ 02_tinyllama_chat_template.ipynb
│ └─ 03_bench_gpu_vs_cpu.ipynb
├─ src/
│ ├─ init.py
│ ├─ dataio.py
│ ├─ modeling.py
│ ├─ chat_utils.py
│ └─ viz.py
├─ experiments/
│ ├─ configs/
│ └─ logs/
├─ tests/
│ └─ test_tokenize.py
├─ assets/
│ ├─ comfy_workflow.png
│ ├─ demo_generation.png
│ └─ cover.png
├─ .gitignore
├─ .gitattributes
├─ requirements.txt
├─ README.md
└─ LICENSE
</details>

## 🚀 快速開始
```bash
# 1) 建議使用 conda
conda create -n ai-core python=3.11 -y && conda activate ai-core

# 2) 安裝套件
pip install -r requirements.txt