#如何開始記錄VRAM
#測試場景定義:
#推理(inference) / 訓練(training/fine-tune)
#這會影響VRAM占用

import torch

print(torch.cuda.get_device_name(0))
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f}MB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f}MB")
print(f"Max Reserved: {torch.cuda.max_memory_allocated(0) / 1024 ** 2:.2f}MB")

#Allocated: 實際被Tensor使用的記憶體
#Reserved: Pytorch已經向CUDA要求的記憶體(會比Allocated多)

import argparse, time, os, psutil

PROC = psutil.Process(os.getpid())

def ram_snapshot(tag="pre"):
    vm = psutil.virtual_memory()
    rss_mb = PROC.memory_info().rss / 1024**2
    return {
        f"{tag}_sys_used_MB": round((vm.total - vm.available) / 1024**2, 2),
        f"{tag}_sys_percent": vm.percent,
        f"{tag}_proc_rss_MB": round(rss_mb, 2),
    }

def ram_peak_sampler(duration_s=0, interval_s=0.2):
    """在任務期間輪詢取樣，回傳過程中 process RSS 的峰值（MB）"""
    peak = PROC.memory_info().rss
    t0 = time.time()
    while time.time() - t0 < duration_s:
        rss = PROC.memory_info().rss
        if rss > peak:
            peak = rss
        time.sleep(interval_s)
    return round(peak / 1024**2, 2)

# optional: 用 NVML 取得 GPU 總量與目前使用
def get_nvml_mem():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        total_mb = info.total / 1024**2
        used_mb  = info.used  / 1024**2
        pynvml.nvmlShutdown()
        return round(total_mb,2), round(used_mb,2)
    except Exception:
        return None, None  # 沒裝也沒關係

def snapshot(tag="pre"):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    reserv = torch.cuda.memory_reserved(0) / 1024**2
    return {
        f"{tag}_allocated_MB": round(alloc, 2),
        f"{tag}_reserved_MB": round(reserv, 2),
    }

def reset_peaks():
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def peak_snapshot():
    torch.cuda.synchronize()
    max_alloc = torch.cuda.max_memory_allocated(0) / 1024**2
    max_reser = torch.cuda.max_memory_reserved(0) / 1024**2
    return {
        "peak_allocated_MB": round(max_alloc, 2),
        "peak_reserved_MB": round(max_reser, 2),
    }

def append_to_excel(xlsx_path, row: dict):
    import pandas as pd
    from pathlib import Path
    Path(os.path.dirname(xlsx_path)).mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_excel(xlsx_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([row])
    df.to_excel(xlsx_path, index=False)

def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen-7B", help="模型名稱")
    parser.add_argument("--task", default="inference", choices=["inference","train"])
    parser.add_argument("--precision", default="fp16", choices=["fp32","fp16","int8","4bit"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--note", default="")
    parser.add_argument("--xlsx", default=r"..\..\assets\Excel\VRAM_compare.xlsx")  # 以 VRAM_compare.py 為相對路徑
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name() if device=="cuda" else "cpu"
    total_mb, used_mb = get_nvml_mem()

    # 量測前快照
    pre = snapshot("pre")
    reset_peaks()

    # ====== 把你的實驗過程放到這裡 ======
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch, threading

    model_id = r"C:\Users\User\Hugging face\hf-models\qwen7b"
    dtype = torch.float16 if args.precision == "fp16" and torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre = {**pre, **ram_snapshot("pre")} #合併RAM指標

    #載入，這步會吃RAM/VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype = dtype,
        low_cpu_mem_usage = True,
        device_map = "auto" if device.type == "cuda" else None,
    )

    #WARMUP (不紀錄)
    _ = model.generate(
        **tokenizer("hello", return_tensors = "pt").to(device),
        max_new_tokens = 8 , do_sample=False, use_cache = True
    )

    # 正式量測
    reset_peaks()
    torch.cuda.synchronize()

    prompt = "You are a helpful assistant. Briefly say hi."
    inputs = tokenizer([prompt]*args.batch_size, return_tensors = "pt", padding = True, max_length=args.seq_len).to(device)

    # 在背景輪詢 RAM 峰值（持續到推理結束）
    peak_ram = {"v": None} 
    def _sample_ram():
        peak_ram["v"] = ram_peak_sampler(duration_s=10, interval_s=0.1)  # 估一個足夠覆蓋生成時間的秒數
        th = threading.Thread(target=_sample_ram); th.start()
    
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True
    )
    torch.cuda.synchronize()
    elapsed = round(time.time()-t0, 3)
    
    th = threading.Thread(target=_sample_ram); th.start()


    th.join()

    # 清理
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    del out, inputs
    torch.cuda.empty_cache()

    
    # 峰值/後快照
    peak = {**peak_snapshot()}
    post = {**snapshot("post"), **ram_snapshot("post")}
    if peak_ram["v"] is not None:
        post["peak_proc_rss_MB"] = peak_ram["v"]
    else:
        post["peak_proc_rss_MB"] = ram_snapshot("post")["post_proc_rss_MB"]
    # ====== TinyLlama inference experiment END ======


    # ====== 實驗過程結束 ======

    # 峰值 / 後快照
    peak = peak_snapshot()
    post = snapshot("post")

     # 彙整紀錄
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": gpu_name,
        "model": args.model,
        "task": args.task,
        "precision": args.precision,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "elapsed_s": elapsed,
        "nvml_total_MB": total_mb,
        "nvml_used_MB_pre": used_mb,
        **pre, **peak, **post,
        "note": args.note,
    }

    append_to_excel(args.xlsx, row)
    print("✅ logged:", row)

if __name__ == "__main__":
    main()