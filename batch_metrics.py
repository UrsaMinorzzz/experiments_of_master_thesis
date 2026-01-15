import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
import re

# [Phase B] Imports
from selector import BudgetAwareSelector
from retriever import DenseRetriever # Your new class

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

def run_batch_test():
    # 1. Setup
    SAMPLE_SIZE = 100 
    BUDGET = 2048
    MODEL_NAME = "codesage/codesage-small-v2"
    
    # [Change 1] Instantiate Dense Retriever
    # Ensure you have a GPU visible or it will fall back to CPU (slow)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = DenseRetriever(model_name=MODEL_NAME, device=device)
    
    selector = BudgetAwareSelector(alpha=1.0, beta=1.0, token_budget=BUDGET)
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