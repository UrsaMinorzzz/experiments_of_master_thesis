import os
import json
import random
import logging
from datetime import datetime
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

# [Phase B] Imports
from selector import BudgetAwareSelector
from retriever import DenseRetriever

# ==========================================
# 1. Experiment Setup
# ==========================================
MAX_SAMPLES_NUMBER = 2665
SAMPLE_SIZE = MAX_SAMPLES_NUMBER
BUDGET = 2048
RETRIEVER_MODEL_NAME = "codesage/codesage-small-v2"
ALPHA = 1.0
BETA = 0.0
GAMMA_STATIC = 0.5
IF_STATIC_MMR = False
POLLUTION_EXPERIMENT = False
DATASET = "ZHENGRAN/cross_code_eval_python"

# ==========================================
# 0. Logging Setup
# ==========================================
os.makedirs("logging", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logging/retrieve_n{SAMPLE_SIZE}_{BUDGET}t_P_{timestamp}.log" if POLLUTION_EXPERIMENT else f"logging/log_n{SAMPLE_SIZE}_{BUDGET}t_NP_{timestamp}.log"

# Configure logger
logger = logging.getLogger("BatchMetrics")
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

class LenTokenizer:
    """Simple tokenizer for budget estimation"""
    def encode(self, text): return text.split()

def calculate_redundancy(selector, selected_list):
    if len(selected_list) < 2: return 0.0
    total_j = 0
    count = 0
    limit = min(len(selected_list), 10)
    for x in range(limit):
        for y in range(x+1, limit):
            s1 = selector._tokenize_to_set(selected_list[x]['code'])
            s2 = selector._tokenize_to_set(selected_list[y]['code'])
            total_j += selector._jaccard_similarity(s1, s2)
            count += 1
    return total_j / count if count else 0.0

def inject_clones(original_chunks, num_clones=5):
    """
    模拟企业级代码库中的高度冗余现象（Boilerplate/Copy-Paste）
    """
    if not original_chunks: return[]
    
    # 假设第一个 chunk 是最具相关性的（GT），我们对它进行克隆
    base_chunk = original_chunks[0]
    polluted_pool = list(original_chunks)
    
    for i in range(num_clones):
        # 制造极其相似，但 Token 层面略有差异的克隆体
        clone = base_chunk + f"\n# Clone variation {i}: " + " ".join(["logging.debug('check');"] * random.randint(1, 3))
        polluted_pool.append(clone)
        
    # 再加一些随机的长噪声进去，确保池子足够大，能撑爆 2048 的 Budget
    for i in range(10):
        polluted_pool.append(f"def dummy_function_{i}():\n    pass\n" * 20)
        
    random.shuffle(polluted_pool)
    return polluted_pool

def run_batch_test():
    # Instantiate Dense Retriever
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = DenseRetriever(model_name=RETRIEVER_MODEL_NAME, device=device)
    
    # 同时实例化两个 Selector：一个静态，一个动态
    selector_static = BudgetAwareSelector(alpha=ALPHA, beta=BETA, token_budget=BUDGET, gamma_static=GAMMA_STATIC, if_static_mmr=True)
    selector_ours = BudgetAwareSelector(alpha=ALPHA, beta=BETA, token_budget=BUDGET, if_static_mmr=False)
    
    tokenizer = LenTokenizer()
    
    ds = load_dataset(DATASET, split="train", streaming=True)
    
    stats = {
        "topk_red": [], "static_red": [], "ours_red":[],
        "topk_count": [], "static_count": [], "ours_count":[],
        "topk_avg_score": [], "static_avg_score": [], "ours_avg_score":[]
    }
    
    logger.info(f"Running Batch Experiment (Dense Retrieval) on {SAMPLE_SIZE} samples...")
    logger.info(f"Configurations: Budget={BUDGET}, Pollution={POLLUTION_EXPERIMENT}, Device={device}")
    
    saved_contexts =[]
    count = 0

    for sample in tqdm(ds, total=SAMPLE_SIZE, desc="Processing Samples"):
        if count >= SAMPLE_SIZE: break
        
        query = sample['prompt']
        retrieved_data = sample.get('crossfile_context_retrieval', {})
        if not retrieved_data or 'list' not in retrieved_data: continue
        
        raw_chunks = retrieved_data['list']
        candidate_texts = [item['retrieved_chunk'] for item in raw_chunks]

        if POLLUTION_EXPERIMENT:
            candidate_texts = inject_clones(candidate_texts)
        
        if not candidate_texts: continue
        
        # 1次检索
        retriever.index_candidates(candidate_texts)
        ranked_results = retriever.search(query, top_k=len(candidate_texts))
        
        candidates =[]
        for i, res in enumerate(ranked_results):
            candidates.append({
                "id": f"c_{i}",
                "code": res['content'],
                "score": res['score'] 
            })
            
        # --- 策略 1: Naive Dense Top-K ---
        baseline_topk_sel =[]
        query_len = len(tokenizer.encode(query))
        curr_tok = query_len
        for c in candidates:
            c_len = len(tokenizer.encode(c['code']))
            if curr_tok + c_len <= BUDGET:
                baseline_topk_sel.append(c)
                curr_tok += c_len
                
        # --- 策略 2: Static MMR (Baseline 2) ---
        baseline_static_sel = selector_static.select(query, candidates, tokenizer)

        # --- 策略 3: Budget-Aware Adaptive MMR (Ours) ---
        ours_sel = selector_ours.select(query, candidates, tokenizer)
        
        # --- Metrics ---
        stats["topk_red"].append(calculate_redundancy(selector_ours, baseline_topk_sel))
        stats["static_red"].append(calculate_redundancy(selector_ours, baseline_static_sel))
        stats["ours_red"].append(calculate_redundancy(selector_ours, ours_sel))
        
        stats["topk_count"].append(len(baseline_topk_sel))
        stats["static_count"].append(len(baseline_static_sel))
        stats["ours_count"].append(len(ours_sel))
        
        if baseline_topk_sel:
            stats["topk_avg_score"].append(np.mean([c['score'] for c in baseline_topk_sel]))
        if baseline_static_sel:
            stats["static_avg_score"].append(np.mean([c['score'] for c in baseline_static_sel]))
        if ours_sel:
            stats["ours_avg_score"].append(np.mean([c['score'] for c in ours_sel]))
        
        gt_text = sample.get('groundtruth')  
        right_context = sample.get('right_context')

        saved_contexts.append({
            "sample_id": count,
            "query": query,
            "ground_truth": gt_text, 
            "right_context": right_context,
            "context_topk": "\n".join([c['code'] for c in baseline_topk_sel]),
            "context_static_mmr": "\n".join([c['code'] for c in baseline_static_sel]),
            "context_ours": "\n".join([c['code'] for c in ours_sel])
        })

        count += 1
        retriever.clear_index()

    # 最后写入 JSONL 文件
    os.makedirs("data", exist_ok=True) # 确保 data 文件夹存在
    out_file = f"data/eval_contexts_n{SAMPLE_SIZE}_{BUDGET}tokens_{'P' if POLLUTION_EXPERIMENT else 'NP'}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for item in saved_contexts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"Saved {len(saved_contexts)} multi-context samples to {out_file}")

    # ==========================================
    # 3. Report
    # ==========================================
    logger.info("="*50)
    logger.info(f" EXPERIMENT RESULTS on n={SAMPLE_SIZE} with {BUDGET} tokens {'(Pollution)' if POLLUTION_EXPERIMENT else '(No Pollution)'}")
    logger.info("="*50)
    
    # 计算 Redundancy
    topk_mean_red = np.mean(stats['topk_red'])
    static_mean_red = np.mean(stats['static_red'])
    ours_mean_red = np.mean(stats['ours_red'])
    
    impr_static = (topk_mean_red - static_mean_red) / topk_mean_red * 100 if topk_mean_red > 0 else 0
    impr_ours = (topk_mean_red - ours_mean_red) / topk_mean_red * 100 if topk_mean_red > 0 else 0
    
    logger.info("Redundancy (Lower is better):")
    logger.info(f"  [1] Naive Top-K:    {topk_mean_red:.4f}")
    logger.info(f"  [2] Static MMR:     {static_mean_red:.4f} (Impr vs Top-K: {impr_static:.2f}%)")
    logger.info(f"  [3] Ours (Dynamic): {ours_mean_red:.4f} (Impr vs Top-K: {impr_ours:.2f}%)")
    
    logger.info("-" * 30)
    logger.info("Avg Item Count (Information Density):")
    logger.info(f"  [1] Naive Top-K:    {np.mean(stats['topk_count']):.2f} chunks")
    logger.info(f"  [2] Static MMR:     {np.mean(stats['static_count']):.2f} chunks")
    logger.info(f"  [3] Ours (Dynamic): {np.mean(stats['ours_count']):.2f} chunks")
    
    logger.info("-" * 30)
    logger.info("Avg Raw Cosine Score (Quality / Purity):")
    logger.info(f"  [1] Naive Top-K:    {np.mean(stats['topk_avg_score']):.4f}")
    logger.info(f"  [2] Static MMR:     {np.mean(stats['static_avg_score']):.4f}")
    logger.info(f"  [3] Ours (Dynamic): {np.mean(stats['ours_avg_score']):.4f}")
    
    logger.info("="*50)

if __name__ == "__main__":
    run_batch_test()