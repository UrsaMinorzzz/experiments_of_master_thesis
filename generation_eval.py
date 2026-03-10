import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

RETRIEVE_OUTPUT_PATH = "eval_contexts_n50.jsonl"

def exact_match_score(prediction: str, ground_truth: str) -> int:
    def normalize(text):
        text = re.sub(r'#.*', '', text) 
        return "".join(text.split())
    return 1 if normalize(prediction) == normalize(ground_truth) else 0

def load_data(file_path=RETRIEVE_OUTPUT_PATH):
    data =[]
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def run_evaluation():
    # 1. 独占显卡加载大模型
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, # 4060 支持 bf16，防止溢出
        device_map="cuda"
    )
    
    # 2. 读取第一阶段固化下来的 Context
    samples = load_data(RETRIEVE_OUTPUT_PATH)
    
    results = {"topk": [], "static_mmr": [], "ours":[]}
    
    def generate_answer(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def build_prompt(context, query):
        return f"<|im_start|>system\nYou are an expert coder. Complete the code based ONLY on the provided context.<|im_end|>\n<|im_start|>user\nContext:\n{context}\n\nCode to complete:\n{query}<|im_end|>\n<|im_start|>assistant\n"

    print("\n--- Starting Generation ---")
    for sample in tqdm(samples):
        query = sample['query']
        gt = sample['ground_truth']
        
        # 测试三种不同的 Context 对同一个模型的影响
        for condition in["topk", "static_mmr", "ours"]:
            context_text = sample[f"context_{condition}"]
            prompt = build_prompt(context_text, query)
            ans = generate_answer(prompt)
            score = exact_match_score(ans, gt)
            results[condition].append(score)

    # 3. 统计并汇报 EM
    print("\n" + "="*40)
    print(" END-TO-END GENERATION RESULTS (Exact Match)")
    print("="*40)
    for condition in["topk", "static_mmr", "ours"]:
        acc = (sum(results[condition]) / len(results[condition])) * 100
        print(f"Context [{condition}]: {acc:.2f}%")

if __name__ == "__main__":
    run_evaluation()