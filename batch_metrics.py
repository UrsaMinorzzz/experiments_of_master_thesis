import numpy as np
from datasets import load_dataset
from selector import BudgetAwareSelector
from rank_bm25 import BM25Okapi
import re
from tqdm import tqdm # 进度条

# 复用之前的辅助类
class Scorer:
    def tokenize(self, text):
        return [t for t in re.findall(r'\w+|[^\w\s]', text) if t.strip()]
    def score_candidates(self, query, candidates_text):
        if not candidates_text: return []
        tokenized_corpus = [self.tokenize(doc) for doc in candidates_text]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = self.tokenize(query)
        return bm25.get_scores(query_tokens)

class LenTokenizer:
    def encode(self, text): return text.split()

def calculate_redundancy(selector, selected_list):
    if len(selected_list) < 2: return 0.0
    total_j = 0
    count = 0
    # 为了速度，只采样前 5 个进行两两比较，或者全比较
    limit = min(len(selected_list), 10)
    for x in range(limit):
        for y in range(x+1, limit):
            s1 = set(selector._tokenize_to_set(selected_list[x]['code']))
            s2 = set(selector._tokenize_to_set(selected_list[y]['code']))
            total_j += selector._jaccard_similarity(s1, s2)
            count += 1
    return total_j / count if count else 0.0

def run_batch_test():
    # 1. 设置
    SAMPLE_SIZE = 100 # 跑前100个样本
    BUDGET = 2048
    
    ds = load_dataset("ZHENGRAN/cross_code_eval_python", split="train", streaming=True)
    selector = BudgetAwareSelector(alpha=1.0, beta=1.0, token_budget=BUDGET)
    scorer = Scorer()
    tokenizer = LenTokenizer()
    
    # 2. 统计器
    stats = {
        "baseline_red": [], "ours_red": [],
        "baseline_count": [], "ours_count": [],
        "baseline_avg_score": [], "ours_avg_score": []
    }
    
    print(f">>> Running Batch Experiment on {SAMPLE_SIZE} samples...")
    
    count = 0
    for sample in tqdm(ds, total=SAMPLE_SIZE):
        if count >= SAMPLE_SIZE: break
        
        # 数据清洗与准备 (同上一个脚本)
        query = sample['prompt']
        retrieved_data = sample.get('crossfile_context_retrieval', {})
        if not retrieved_data or 'list' not in retrieved_data: continue
        
        raw_chunks = retrieved_data['list']
        candidate_texts = [item['retrieved_chunk'] for item in raw_chunks]
        if not candidate_texts: continue
        
        # 算分
        scores = scorer.score_candidates(query, candidate_texts)
        candidates = []
        for idx, (chunk_obj, score) in enumerate(zip(raw_chunks, scores)):
            candidates.append({
                "id": f"c_{idx}", 
                "code": chunk_obj['retrieved_chunk'], 
                "score": float(score)
            })
            
        # --- Baseline (Top-K) ---
        candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)
        baseline_sel = []
        query_len = len(tokenizer.encode(query))
        curr_tok = query_len 
        for c in candidates_sorted:
            c_len = len(tokenizer.encode(c['code']))
            if curr_tok + c_len <= BUDGET:
                baseline_sel.append(c)
                curr_tok += c_len
                
        # --- Ours ---
        ours_sel = selector.select(query, candidates, tokenizer)
        
        # --- 记录指标 ---
        # 1. Redundancy
        stats["baseline_red"].append(calculate_redundancy(selector, baseline_sel))
        stats["ours_red"].append(calculate_redundancy(selector, ours_sel))
        
        # 2. Count (Retrieval Density)
        stats["baseline_count"].append(len(baseline_sel))
        stats["ours_count"].append(len(ours_sel))
        
        # 3. Relevance Maintenance (检查是否选了烂苹果)
        # 计算选中项的平均原始得分
        if baseline_sel:
            stats["baseline_avg_score"].append(np.mean([c['score'] for c in baseline_sel]))
        if ours_sel:
            stats["ours_avg_score"].append(np.mean([c['score'] for c in ours_sel]))
        
        # 3. Max Score (Check if we kept the best item)
        if baseline_sel:
            stats.setdefault("baseline_max_score", []).append(max([c['score'] for c in baseline_sel]))
        if ours_sel:
            stats.setdefault("ours_max_score", []).append(max([c['score'] for c in ours_sel]))

        count += 1

    # 3. 最终报告
    print("\n" + "="*40)
    print(" EXPERIMENT RESULTS (N=100)")
    print("="*40)
    
    b_mean_red = np.mean(stats['baseline_red'])
    o_mean_red = np.mean(stats['ours_red'])
    red_reduction = (b_mean_red - o_mean_red) / b_mean_red * 100 if b_mean_red > 0 else 0
    
    print(f"Redundancy (Lower is better):")
    print(f"  Baseline: {b_mean_red:.4f}")
    print(f"  Ours:     {o_mean_red:.4f}")
    print(f"  Improvement: -{red_reduction:.2f}%")
    
    print("-" * 20)
    print(f"Avg Item Count (Higher ~ possibly better density):")
    print(f"  Baseline: {np.mean(stats['baseline_count']):.2f}")
    print(f"  Ours:     {np.mean(stats['ours_count']):.2f}")
    
    print("-" * 20)
    print(f"Avg Relevance Score (Should be close):")
    print(f"  Baseline: {np.mean(stats['baseline_avg_score']):.2f}")
    print(f"  Ours:     {np.mean(stats['ours_avg_score']):.2f}")
    
    print("-" * 20)
    print(f"Max Relevance Score (Must be EQUAL):")
    print(f"  Baseline: {np.mean(stats['baseline_max_score']):.2f}")
    print(f"  Ours:     {np.mean(stats['ours_max_score']):.2f}")

    print("="*40)

if __name__ == "__main__":
    run_batch_test()