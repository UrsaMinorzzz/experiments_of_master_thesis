import numpy as np
from datasets import load_dataset
from selector import BudgetAwareSelector
from rank_bm25 import BM25Okapi
import re

# ==========================================
# 1. 辅助工具：快速算分
# ==========================================
class Scorer:
    """为了给 Selector 提供 relevance score，我们现场算一下 BM25"""
    def tokenize(self, text):
        return [t for t in re.findall(r'\w+|[^\w\s]', text) if t.strip()]

    def score_candidates(self, query, candidates_text):
        tokenized_corpus = [self.tokenize(doc) for doc in candidates_text]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = self.tokenize(query)
        return bm25.get_scores(query_tokens)

# ==========================================
# 2. 实验主逻辑
# ==========================================
def run_experiment():
    print(">>> [Step 1] Loading CrossCodeEval Data...")
    # 使用 train split (前面验证过它其实包含了测试数据)
    ds = load_dataset("ZHENGRAN/cross_code_eval_python", split="train", streaming=True)
    
    # 我们的 Selector 设置
    # 预算设为 512 tokens (典型的本地模型上下文限制)
    budget = 512
    selector = BudgetAwareSelector(alpha=1.0, beta=1.0, token_budget=budget)
    
    scorer = Scorer()
    
    # 简单的 Mock Tokenizer (计算长度用)
    class LenTokenizer:
        def encode(self, text): return text.split() # 粗略估计
    tokenizer = LenTokenizer()

    print(f">>> [Step 2] Processing Samples (Budget: {budget} tokens)...")
    print("="*60)

    # 我们处理前 3 个样本看看效果
    for i, sample in enumerate(ds):
        if i >= 3: break
        
        print(f"\nProcessing Query #{i} (Task ID: {sample['metadata']['task_id']})")
        
        query = sample['prompt']
        
        # 提取候选集 (Raw List)
        # 注意：dataset 里这一层是 {'list': [...]}
        retrieved_data = sample['crossfile_context_retrieval']
        if not retrieved_data or 'list' not in retrieved_data:
            print("  -> No retrieved context found, skipping.")
            continue
            
        raw_chunks = retrieved_data['list'] # List of dicts
        
        # 准备 Candidates 格式
        candidates = []
        candidate_texts = [item['retrieved_chunk'] for item in raw_chunks]
        
        # 现场打分 (Assign Scores)
        # 因为数据集里给的可能是无序的或者只有 chunks
        if not candidate_texts:
            continue
            
        scores = scorer.score_candidates(query, candidate_texts)
        
        for idx, (chunk_obj, score) in enumerate(zip(raw_chunks, scores)):
            candidates.append({
                "id": f"c_{idx}",
                "code": chunk_obj['retrieved_chunk'],
                "filename": chunk_obj['filename'],
                "score": float(score) # BM25 score
            })
            
        # -------------------------------------------------
        # 对决开始！
        # -------------------------------------------------
        
        # Strategy A: Baseline (Naive Top-K)
        # 既然我们重新算了 BM25 分数，我们就按分数排序，填满预算
        candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)
        baseline_selected = []
        current_tokens = 0
        
        for c in candidates_sorted:
            c_len = len(tokenizer.encode(c['code']))
            if current_tokens + c_len <= budget:
                baseline_selected.append(c)
                current_tokens += c_len
            # Naive 策略通常遇到一个塞不进去就不再尝试小的了，或者继续尝试
            # 这里假设它继续尝试填满
            
        # Strategy B: Our Budget-Aware Selector
        # 注意：Selector 内部会自动根据 score 归一化并计算冗余
        our_selected = selector.select(query, candidates, tokenizer)
        
        # -------------------------------------------------
        # 结果分析
        # -------------------------------------------------
        b_ids = [c['id'] for c in baseline_selected]
        o_ids = [c['id'] for c in our_selected]
        
        print(f"  [Baseline] Count: {len(b_ids)} | IDs: {b_ids}")
        print(f"  [Ours]     Count: {len(o_ids)} | IDs: {o_ids}")
        
        # 关键指标：Jaccard Redundancy within selected context
        # 我们希望 Ours 选出的集合内部冗余度更低
        def calc_set_redundancy(selected_list):
            if len(selected_list) < 2: return 0.0
            total_j = 0
            count = 0
            for x in range(len(selected_list)):
                for y in range(x+1, len(selected_list)):
                    s1 = set(selector._tokenize_to_set(selected_list[x]['code']))
                    s2 = set(selector._tokenize_to_set(selected_list[y]['code']))
                    total_j += selector._jaccard_similarity(s1, s2)
                    count += 1
            return total_j / count if count else 0.0

        b_red = calc_set_redundancy(baseline_selected)
        o_red = calc_set_redundancy(our_selected)
        
        print(f"  [Metric] Avg Redundancy: Baseline={b_red:.4f} vs Ours={o_red:.4f}")
        
        if o_red < b_red:
            print("  -> ✅ WIN: Selector Reduced Redundancy!")
        elif len(o_ids) > len(b_ids):
             print("  -> ✅ WIN: Selector fit MORE chunks (Higher Density)!")
        else:
            print("  -> (Tie or Context is already clean)")

if __name__ == "__main__":
    run_experiment()