import os
import json
import requests
from tqdm import tqdm
import re
import time
from datetime import datetime
import logging
# ==========================================
# 1. Configuration
# ==========================================
# 安全读取 API KEY
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("🚨 Environment variable OPENROUTER_API_KEY is not set. Please export it before running.")

MODEL_ID = "qwen/qwen2.5-coder-7b-instruct"

GENERATION_OUTPUT_DIR = "generation_results"
os.makedirs(GENERATION_OUTPUT_DIR, exist_ok=True) # 确保输出目录存在

MAX_TEST_SAMPLES = 400
MAX_RETRIES = 3
TOKEN_BUDGET = 2048
POLLUTION_EXPERIMENT = True

RETRIEVE_OUTPUT_PATH = f"data/eval_contexts_n{MAX_TEST_SAMPLES}_{TOKEN_BUDGET}tokens_{'P' if POLLUTION_EXPERIMENT else 'NP'}.jsonl" 

# ==========================================
# 0. Logging Setup
# ==========================================
os.makedirs("logging", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logging/gen_eval_n{MAX_TEST_SAMPLES}_{TOKEN_BUDGET}t_{'P' if POLLUTION_EXPERIMENT else 'NP'}_{timestamp}.log"

# Configure logger
logger = logging.getLogger("GenEval")
logger.setLevel(logging.INFO)

# File Handler
fh = logging.FileHandler(log_filename, encoding='utf-8')
fh.setLevel(logging.INFO)

# Console Handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# ==========================================
# 2. Helper Functions
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

def call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_ID,
        "messages":[
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
        ],
        "temperature": 0.0, 
        "response_format": {"type": "json_object"},
        "max_tokens": 128
    }
    
    for _ in range(MAX_RETRIES):
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            raw_content = response.json()['choices'][0]['message']['content']
            
            try:
                clean_json = raw_content.replace("```json", "").replace("```", "").strip()
                result_dict = json.loads(clean_json)
                return result_dict.get("completion", "")
            except json.JSONDecodeError:
                # 暴力兜底法
                match = re.search(r'"completion"\s*:\s*"(.*)', raw_content, re.IGNORECASE | re.DOTALL)
                if match:
                    salvaged = match.group(1)
                    salvaged = re.sub(r'"?\s*\}?\s*$', '', salvaged)
                    return salvaged
                else:
                    return raw_content 
                
        except Exception as e:
            time.sleep(2)
            continue
    return ""

# ==========================================
# 3. Main Evaluation Pipeline
# ==========================================
def run_evaluation():
    try:
        samples = load_data(RETRIEVE_OUTPUT_PATH)
    except FileNotFoundError:
        logger.error(f"Retrieval data file not found: {RETRIEVE_OUTPUT_PATH}. Please run batch_metrics.py first.")
        return

    TEST_LIMIT = min(MAX_TEST_SAMPLES, len(samples))
    samples = samples[:TEST_LIMIT]
    
    results = {"topk": [], "static_mmr": [], "ours":[]}
    logged_generations =[] 
    
    logger.info(f"Starting Generation Eval via OpenRouter ({MODEL_ID})")
    logger.info(f"Configuration: Samples={TEST_LIMIT}, Budget={TOKEN_BUDGET}, Pollution={POLLUTION_EXPERIMENT}")
    
    for sample in tqdm(samples, desc="Querying LLM API"):
        query = sample['query']
        gt = sample['ground_truth']
        
        sample_log = {
            "sample_id": sample.get("sample_id", "unknown"),
            "query": query,
            "ground_truth": gt,
            "generations": {},
            "scores": {}
        }
        
        for condition in["topk", "static_mmr", "ours"]:
            context_text = sample[f"context_{condition}"]
            
            prompt = (
                f"### Task\nRead the Context and the Incomplete Code. Output a JSON object with the key 'completion'.\n\n"
                f"### Context\n{context_text}\n\n"
                f"### Incomplete Code\n{query}\n\n"
                f"### JSON Output:\n"
            )
            
            ans = call_openrouter(prompt)
            score = exact_match_score(ans, gt)
            
            results[condition].append(score)
            sample_log["generations"][condition] = ans
            sample_log["scores"][condition] = score
            
            time.sleep(0.5) 
                
        logged_generations.append(sample_log)

    # 导出到带时间戳的 JSONL
    out_filename = os.path.join(GENERATION_OUTPUT_DIR, f"generation_n{TEST_LIMIT}_{TOKEN_BUDGET}tokens_{'P' if POLLUTION_EXPERIMENT else 'NP'}_{timestamp}.jsonl")
    with open(out_filename, "w", encoding="utf-8") as f:
        for item in logged_generations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"Raw generation log exported to: {out_filename}")

    # 打印最终 EM 结果
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