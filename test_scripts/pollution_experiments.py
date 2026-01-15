import numpy as np
from datasets import load_dataset
from selector import BudgetAwareSelector
from rank_bm25 import BM25Okapi
import re
from tqdm import tqdm
import copy

# 复用工具
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
    limit = min(len(selected_list), 10)
    for x in range(limit):
        for y in range(x+1, limit):
            s1 = set(selector._tokenize_to_set(selected_list[x]['code']))
            s2 = set(selector._tokenize_to_set(selected_list[y]['code']))
            total_j += selector._jaccard_similarity(s1, s2)
            count += 1
    return total_j / count if count else 0.0

def run_pollution_test():
    SAMPLE_SIZE = 50  # 跑50个样本就够看清趋势了
    BUDGET = 512      # 保持紧缺
    
    print(f">>> [Experiment] Synthetic Pollution Robustness Test")
    print(f">>> Scenario: Injecting 1 'Evil Clone' for every candidate to simulate repo duplication.")
    
    ds = load_dataset("ZHENGRAN/cross_code_eval_python", split="train", streaming=True)
    selector = BudgetAwareSelector(alpha=1.0, beta=1.0, token_budget=BUDGET)
    scorer = Scorer()
    tokenizer = LenTokenizer()
    
    stats = {"baseline_red": [], "ours_red": [], "baseline_count": [], "ours_count": []}
    
    count = 0
    for sample in tqdm(ds, total=SAMPLE_SIZE):
        if count >= SAMPLE_SIZE: break
        
        query = sample['prompt']
        retrieved_data = sample.get('crossfile_context_retrieval', {})
        if not retrieved_data or 'list' not in retrieved_data: continue
        
        raw_chunks = retrieved_data['list']
        if not raw_chunks: continue
        
        # 1. 初始打分
        original_texts = [item['retrieved_chunk'] for item in raw_chunks]
        scores = scorer.score_candidates(query, original_texts)
        
        candidates = []
        # 2. 【关键步骤】注入污染
        for idx, (chunk_obj, score) in enumerate(zip(raw_chunks, scores)):
            # 添加原始项
            orig_code = chunk_obj['retrieved_chunk']
            candidates.append({
                "id": f"c_{idx}_orig", 
                "code": orig_code, 
                "score": float(score)
            })
            
            # 添加 Evil Clone (只有空格差异，或者极小差异)
            # 模拟：项目中存在 copy-paste 的代码
            # 我们给它稍微低一点的分数 (98%)，这样 Baseline 依然会很想选它
            clone_code = orig_code.replace(" ", "  ") # 增加空格，逻辑不变
            candidates.append({
                "id": f"c_{idx}_CLONE",
                "code": clone_code,
                "score": float(score) * 0.98 # 高分冗余！
            })
            
        # --- Baseline (Fair Version) ---
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
        
        # --- Metrics ---
        stats["baseline_red"].append(calculate_redundancy(selector, baseline_sel))
        stats["ours_red"].append(calculate_redundancy(selector, ours_sel))
        stats["baseline_count"].append(len(baseline_sel))
        stats["ours_count"].append(len(ours_sel))
            
        count += 1

    # 结果报告
    print("\n" + "="*40)
    print(" POLLUTION TEST RESULTS")
    print("="*40)
    
    b_red = np.mean(stats['baseline_red'])
    o_red = np.mean(stats['ours_red'])
    
    print(f"Avg Redundancy (With Noise Injection):")
    print(f"  Baseline: {b_red:.4f} (Should be high, ~0.5)")
    print(f"  Ours:     {o_red:.4f} (Should be low)")
    
    reduction = (b_red - o_red) / b_red * 100 if b_red > 0 else 0
    print(f"  >>> Improvement: -{reduction:.2f}%")
    
    print("-" * 20)
    print(f"Avg Item Count:")
    print(f"  Baseline: {np.mean(stats['baseline_count']):.2f} (Stuffed with clones)")
    print(f"  Ours:     {np.mean(stats['ours_count']):.2f} (Unique items only)")

if __name__ == "__main__":
    run_pollution_test()