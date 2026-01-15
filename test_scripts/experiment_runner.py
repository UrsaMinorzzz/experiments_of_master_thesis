import json
import numpy as np
from typing import List, Dict
from selector import BudgetAwareSelector

# 模拟一个 tokenizer，实际跑的时候换成 huggingface 的 AutoTokenizer
class SimpleTokenizer:
    def encode(self, text):
        return text.split() # 暂时用 split 模拟 token 消耗

    def decode(self, tokens):
        return " ".join(tokens)

class ExperimentRunner:
    def __init__(self, dataset_path: str = None):
        self.tokenizer = SimpleTokenizer()
        # 初始化我们的 Selector
        self.selector = BudgetAwareSelector(
            alpha=1.0,  # Relevance
            beta=0.5,   # Structure (假设)
            token_budget=512 # 严格限制上下文预算
        )
        self.dataset = self._load_dataset(dataset_path)

    def _load_dataset(self, path):
        # 实际代码中，这里会加载 RepoEval 的 jsonl 文件
        # 现在我们 Mock 一条真实难度的数据
        return [
            {
                "query_id": "test_case_01",
                "query_context": "from utils import Logger\ndef process_data(data):\n    log = Logger()\n    # TODO: validate data",
                "ground_truth_context": "def validate(d):\n    if not d: raise ValueError", 
                # 假设 Retrieve 拿到了 5 个候选，其中包含大量重复
                "retrieved_candidates": [
                    # Candidate 1: 高分，但只是 Logger 的定义，相关但不完全
                    {"id": "c1", "code": "class Logger:\n    def log(self, msg): print(msg)", "score": 0.9},
                    
                    # Candidate 2: 完美的 Ground Truth (目标)
                    {"id": "c2", "code": "def validate(d):\n    if not d: raise ValueError", "score": 0.88},
                    
                    # Candidate 3: 恶心的重复 (Copy of c2 with comments) -> 应该被剔除
                    {"id": "c3", "code": "# Validation Logic\ndef validate(d):\n    if not d: raise ValueError", "score": 0.87},
                    
                    # Candidate 4: 又是重复 (Copy of c1) -> 应该被剔除
                    {"id": "c4", "code": "class Logger:\n    def log(self, msg): pass", "score": 0.85},
                    
                    # Candidate 5: 低分但 unique 的信息 (比如其他相关工具) -> 应该被保留填补预算
                    {"id": "c5", "code": "def clean_string(s):\n    return s.strip()", "score": 0.6}
                ]
            }
        ]

    def run_comparison(self):
        print(f"{'='*20} EXPERIMENT START {'='*20}")
        print(f"Token Budget: {self.selector.budget}")
        
        for entry in self.dataset:
            print(f"\n>>> Processing Query: {entry['query_id']}")
            candidates = entry['retrieved_candidates']
            query = entry['query_context']
            
            # --- Baseline: Naive Top-K (Greedy by Score) ---
            # 简单的按分数排序，填满预算为止
            baseline_selected = []
            current_tokens = 0
            sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            
            for c in sorted_candidates:
                c_len = len(self.tokenizer.encode(c['code']))
                if current_tokens + c_len <= self.selector.budget:
                    baseline_selected.append(c)
                    current_tokens += c_len
            
            # --- Ours: Budget-Aware Selector ---
            our_selected = self.selector.select(query, candidates, self.tokenizer)
            
            # --- Analysis ---
            self._print_results(baseline_selected, our_selected)

    def _print_results(self, baseline, ours):
        b_ids = [c['id'] for c in baseline]
        o_ids = [c['id'] for c in ours]
        
        print(f"Baseline Selection: {b_ids}")
        print(f"Ours Selection:     {o_ids}")
        
        # 简单分析：我们需要看到 Ours 选了 c2 和 c5 (多样性)，而 Baseline 选了 c1, c2, c3 (重复)
        # 这里用 Set 比较
        b_set = set(b_ids)
        o_set = set(o_ids)
        
        if "c3" in b_set and "c3" not in o_set:
            print(" -> WIN: Successfully removed redundant candidate c3")
        
        if "c5" in o_set and "c5" not in b_set:
            print(" -> WIN: Successfully included unique candidate c5 (Lower score but high gain)")

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_comparison()