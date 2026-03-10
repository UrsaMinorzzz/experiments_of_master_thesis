import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
import random
import re

# [Phase B] Imports
from selector import BudgetAwareSelector
from retriever import DenseRetriever # Your new class

# 1. Setup
SAMPLE_SIZE = 100 
BUDGET = 2048
MODEL_NAME = "codesage/codesage-small-v2"
ALPHA = 1.0
BETA = 0.0
GAMMA_STATIC = 0.5
IF_STATIC_MMR = False

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
            # Using private method from selector for consistency, strictly we should expose it
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
        # 也可以替换换行、空格等
        polluted_pool.append(clone)
        
    # 再加一些随机的长噪声进去，确保池子足够大，能撑爆 2048 的 Budget
    for i in range(10):
        polluted_pool.append(f"def dummy_function_{i}():\n    pass\n" * 20)
        
    random.shuffle(polluted_pool)
    return polluted_pool

def run_batch_test():
    # [Change 1] Instantiate Dense Retriever
    # Ensure you have a GPU visible or it will fall back to CPU (slow)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = DenseRetriever(model_name=MODEL_NAME, device=device)
    
    selector = BudgetAwareSelector(alpha=ALPHA, beta=BETA, token_budget=BUDGET, gamma_static=GAMMA_STATIC, if_static_mmr=IF_STATIC_MMR)
    tokenizer = LenTokenizer()
    
    ds = load_dataset("ZHENGRAN/cross_code_eval_python", split="train", streaming=True)
    
    stats = {
        "baseline_red": [], "ours_red": [],
        "baseline_count": [], "ours_count": [],
        "baseline_avg_score": [], "ours_avg_score": []
    }
    
    print(f">>> Running Batch Experiment (Dense Retrieval) on {SAMPLE_SIZE} samples...")
    print(f"    Device: {device}")
    
    count = 0
    for sample in tqdm(ds, total=SAMPLE_SIZE):
        if count >= SAMPLE_SIZE: break
        
        query = sample['prompt']
        # CrossCodeEval data format
        retrieved_data = sample.get('crossfile_context_retrieval', {})
        if not retrieved_data or 'list' not in retrieved_data: continue
        
        raw_chunks = retrieved_data['list']
        # Extract plain text list for embedding
        candidate_texts = [item['retrieved_chunk'] for item in raw_chunks]

        # Simulate a polluted candidate pool with many near-duplicates (Boilerplate/Copy-Paste)
        candidate_texts = inject_clones(candidate_texts)
        
        if not candidate_texts: continue
        
        # [Change 2] Dense Indexing & Search (Ad-hoc)
        # We index the pool of candidates specifically for this query
        retriever.index_candidates(candidate_texts)
        
        # Retrieve ALL candidates sorted by Cosine Similarity
        # This assigns the 'score' field (Raw Cosine)
        ranked_results = retriever.search(query, top_k=len(candidate_texts))
        
        # Convert to format expected by Selector
        # retriever.search returns: {'content', 'score', ...}
        # We map it to: {'code', 'score', 'id'}
        candidates = []
        for i, res in enumerate(ranked_results):
            candidates.append({
                "id": f"c_{i}",
                "code": res['content'],
                "score": res['score'] # Raw Cosine [-1, 1]
            })
        
        # # Debug: Print candidate pool size
        # print(f"Pool size: {len(candidates)}")
            
        # --- Baseline: Naive Dense Top-K ---
        # Candidates are already sorted by score from retriever.search
        baseline_sel = []
        query_len = len(tokenizer.encode(query))
        curr_tok = query_len
        
        for c in candidates: # Already sorted desc
            c_len = len(tokenizer.encode(c['code']))
            if curr_tok + c_len <= BUDGET:
                baseline_sel.append(c)
                curr_tok += c_len
                
        # --- Ours: Budget-Aware MMR ---
        # Pass the pre-ranked, pre-scored candidates
        ours_sel = selector.select(query, candidates, tokenizer)

        # print(ours_sel)
        
        # --- Metrics ---
        stats["baseline_red"].append(calculate_redundancy(selector, baseline_sel))
        stats["ours_red"].append(calculate_redundancy(selector, ours_sel))
        stats["baseline_count"].append(len(baseline_sel))
        stats["ours_count"].append(len(ours_sel))
        
        if baseline_sel:
            stats["baseline_avg_score"].append(np.mean([c['score'] for c in baseline_sel]))
        if ours_sel:
            stats["ours_avg_score"].append(np.mean([c['score'] for c in ours_sel]))

        count += 1
        
        # [Optional] Clear index to save RAM (though ad-hoc index is small)
        retriever.clear_index()

    # 3. Report
    print("\n" + "="*40)
    print(" EXPERIMENT RESULTS (Dense Retrieval)")
    print("="*40)
    
    b_mean_red = np.mean(stats['baseline_red'])
    o_mean_red = np.mean(stats['ours_red'])
    red_reduction = (b_mean_red - o_mean_red) / b_mean_red * 100 if b_mean_red > 0 else 0
    
    print(f"Redundancy (Lower is better):")
    print(f"  Baseline: {b_mean_red:.4f}")
    print(f"  Ours:     {o_mean_red:.4f}")
    print(f"  Improvement: {red_reduction:.2f}%")
    
    print("-" * 20)
    print(f"Avg Item Count (Information Density):")
    print(f"  Baseline: {np.mean(stats['baseline_count']):.2f}")
    print(f"  Ours:     {np.mean(stats['ours_count']):.2f}")
    
    print("-" * 20)
    print(f"Avg Raw Cosine Score:")
    print(f"  Baseline: {np.mean(stats['baseline_avg_score']):.4f}")
    print(f"  Ours:     {np.mean(stats['ours_avg_score']):.4f}")
    print("  (Note: Ours is expected to be slightly lower, trading marginal relevance for diversity)")

if __name__ == "__main__":
    run_batch_test()