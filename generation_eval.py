import os
import json
import re
import time
from datetime import datetime
import logging
from tqdm import tqdm

# 移除 requests，引入 mlx
try:
    from mlx_lm import load, generate
except ImportError:
    raise ImportError("🚨 请先安装依赖: pip install mlx mlx-lm")

# ==========================================
# 1. Configuration
# ==========================================
# 使用 4bit 量化版本最大化 M4 的内存带宽利用率和推理速度
MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

GENERATION_OUTPUT_DIR = "generation_results"
os.makedirs(GENERATION_OUTPUT_DIR, exist_ok=True)

TEST_SAMPLES_NUM = 10
TOKEN_BUDGET = 2048
POLLUTION_EXPERIMENT = True

RETRIEVE_OUTPUT_PATH = f"data/eval_contexts_n{TEST_SAMPLES_NUM}_{TOKEN_BUDGET}tokens_{'P' if POLLUTION_EXPERIMENT else 'NP'}.jsonl" 

# ==========================================
# 0. Logging Setup
# ==========================================
os.makedirs("logging", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logging/gen_eval_n{TEST_SAMPLES_NUM}_{TOKEN_BUDGET}t_{'P' if POLLUTION_EXPERIMENT else 'NP'}_{timestamp}.log"

logger = logging.getLogger("GenEval")
logger.setLevel(logging.INFO)

fh = logging.FileHandler(log_filename, encoding='utf-8')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

# ==========================================
# 2. Local Model Initialization
# ==========================================
logger.info(f"Loading {MODEL_ID} into Apple Unified Memory...")
start_load = time.time()
# MLX Load 默认会将权重直接映射到统一内存
model, tokenizer = load(MODEL_ID)
logger.info(f"Model loaded in {time.time() - start_load:.2f} seconds.")

# ==========================================
# 3. Helper Functions
# ==========================================
def exact_match_score(prediction: str, ground_truth: str) -> int:
    """放宽的 EM 评估：去除一切空白符和多余的 markdown 标记"""
    def normalize(text):
        if not text: return ""
        text = text.replace("```python", "").replace("```", "")
        text = re.sub(r'#.*', '', text) 
        return re.sub(r'\s+', '', text)
    
    return 1 if normalize(prediction) == normalize(ground_truth) else 0

def load_data(file_path=RETRIEVE_OUTPUT_PATH):
    data =[]
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def call_local_mlx(prompt: str) -> str:
    """使用本地 MLX 进行推理，替代原来的网络 API 请求"""
    messages =[
        {
            "role": "system", 
            "content": (
                "You are a strict code completion API. "
                "You MUST respond with a valid JSON object containing exactly ONE key: 'completion'. "
                "The value must be the EXACT string of code that continues the user's incomplete code. "
                "Do not output markdown, explanations, or any other text outside the JSON object."
            )
        },
        {"role": "user", "content": prompt}
    ]
    
    # 使用 Qwen 官方的 Chat Template，避免特殊 token 错乱
    text_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 物理机上直接抛错（如果有的话），不需要 Try-Catch 掩盖 Bug
    raw_content = generate(
        model,
        tokenizer,
        prompt=text_prompt,
        max_tokens=128,
        # temp=0.0,  # 0.0 即 Greedy Decoding，符合论文复现要求
        verbose=False     # 关闭终端逐字打印以保持进度条整洁
    )
    
    # 保留原有的 JSON 抢救解析逻辑
    try:
        clean_json = raw_content.replace("```json", "").replace("```", "").strip()
        result_dict = json.loads(clean_json)
        return result_dict.get("completion", "")
    except json.JSONDecodeError:
        match = re.search(r'"completion"\s*:\s*"(.*)', raw_content, re.IGNORECASE | re.DOTALL)
        if match:
            salvaged = match.group(1)
            salvaged = re.sub(r'"?\s*\}?\s*$', '', salvaged)
            return salvaged
        else:
            return raw_content 

# ==========================================
# 4. Main Evaluation Pipeline
# ==========================================
def run_evaluation():
    try:
        samples = load_data(RETRIEVE_OUTPUT_PATH)
    except FileNotFoundError:
        logger.error(f"Retrieval data file not found: {RETRIEVE_OUTPUT_PATH}. Please run batch_metrics.py first.")
        return

    test_limit = min(TEST_SAMPLES_NUM, len(samples))
    samples = samples[:test_limit]
    
    results = {"topk":[], "static_mmr": [], "ours": []}
    logged_generations =[] 
    
    logger.info(f"Starting Generation Eval via Local MLX ({MODEL_ID})")
    logger.info(f"Configuration: Samples={test_limit}, Budget={TOKEN_BUDGET}, Pollution={POLLUTION_EXPERIMENT}")
    
    for sample in tqdm(samples, desc="Inference Progress"):
        query = sample['query']
        gt = sample['ground_truth']
        
        sample_log = {
            "sample_id": sample.get("sample_id", "unknown"),
            "query": query,
            "ground_truth": gt,
            "generations": {},
            "scores": {}
        }
        
        for condition in ["topk", "static_mmr", "ours"]:
            context_text = sample[f"context_{condition}"]
            
            prompt = (
                f"### Task\nRead the Context and the Incomplete Code. Output a JSON object with the key 'completion'.\n\n"
                f"### Context\n{context_text}\n\n"
                f"### Incomplete Code\n{query}\n\n"
                f"### JSON Output:\n"
            )
            
            ans = call_local_mlx(prompt)
            score = exact_match_score(ans, gt)
            
            results[condition].append(score)
            sample_log["generations"][condition] = ans
            sample_log["scores"][condition] = score
            
        logged_generations.append(sample_log)

    out_filename = os.path.join(GENERATION_OUTPUT_DIR, f"generation_n{test_limit}_{TOKEN_BUDGET}tokens_{'P' if POLLUTION_EXPERIMENT else 'NP'}_{timestamp}.jsonl")
    with open(out_filename, "w", encoding="utf-8") as f:
        for item in logged_generations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"Raw generation log exported to: {out_filename}")

    logger.info("=" * 50)
    logger.info(" END-TO-END GENERATION RESULTS (Exact Match)")
    logger.info("=" * 50)
    for condition in ["topk", "static_mmr", "ours"]:
        acc = (sum(results[condition]) / len(results[condition])) * 100
        correct_count = sum(results[condition])
        total_count = len(results[condition])
        logger.info(f"  Context[{condition.upper()}]: {acc:.2f}% ({correct_count}/{total_count})")
    logger.info("=" * 50)

if __name__ == "__main__":
    run_evaluation()