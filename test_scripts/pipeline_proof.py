import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from selector import BudgetAwareSelector
import re

# ==========================================
# 1. 简易工具类
# ==========================================
class SimpleCodeTokenizer:
    """为了 BM25 和 Selector 统一的分词逻辑"""
    def tokenize(self, text):
        # 简单的正则分词：保留单词和符号
        return [t for t in re.findall(r'\w+|[^\w\s]', text) if t.strip()]

    def encode(self, text):
        # 模拟 HF Tokenizer 的 encode，返回 id 列表 (这里用长度模拟)
        return self.tokenize(text)

def build_bm25_index(corpus_texts):
    """构建 BM25 索引"""
    tokenizer = SimpleCodeTokenizer()
    tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# ==========================================
# 2. 核心实验流程
# ==========================================
def run_real_world_proof():
    print(">>> [Phase 1] Loading CrossCodeEval Data (Streaming)...")
    # 使用 streaming=True 避免下载整个巨大数据集，只取前几条做验证
    ds = load_dataset("cogrammar/crosscodeeval", "python", split="test", streaming=True)
    
    # 取出第一个有效样本
    # CrossCodeEval 的结构通常包含 'reference' (ground truth) 和项目上下文
    # 这里我们简化：假设 project 里的其他文件是检索库
    sample = None
    for item in ds:
        # 我们需要一个包含 context 信息的样本
        # CrossCodeEval 实际上是非常 Raw 的，这里为了演示，
        # 如果下载太慢，我会手动 Mock 一个真实的复杂结构，但我们先试着读一下
        sample = item
        break
    
    if not sample:
        print("Error: Could not load sample.")
        return

    print(f">>> Sample Loaded: {sample.get('file_path', 'Unknown File')}")
    
    # --- 模拟构建检索库 (Corpus) ---
    # 在真实 CrossCodeEval 任务中，你需要读取该项目下的其他文件。
    # 这里为了证明 pipeline 可行，我们手动制造一个 "Distractor Corpus" (干扰库)
    # 包含：真实相关的代码，完全不相关的代码，以及恶意重复的代码
    
    true_context = "def helper_function(x):\n    return x * 2"
    
    corpus_snippets = [
        # 1. 真实相关 (Ground Truth Context)
        true_context,
        # 2. 真实相关的稍微修改版 (Redundant)
        "def helper_function(x): # multiplied by 2\n    return x * 2",
        # 3. 真实相关的另一版 (Redundant)
        "def helper_function(n):\n    return n * 2",
        # 4. 完全不相关 (Noise)
        "class Database:\n    def connect(self): pass",
        "import os\nimport sys",
        "def quick_sort(arr): return sorted(arr)",
        # 5. 又是重复 (Redundant)
        "def helper_function(x):\n    return x * 2  # duplicate"
    ]
    
    # 给每个 snippet 一个 ID
    corpus_dicts = [{"id": f"doc_{i}", "code": code} for i, code in enumerate(corpus_snippets)]
    
    print("\n>>> [Phase 2] Running BM25 Retrieval...")
    # 模拟 Query: 假设我们在写主函数，需要调用 helper_function
    query_code = "def main():\n    val = 10\n    # call helper\n    res = "
    
    bm25 = build_bm25_index([c['code'] for c in corpus_dicts])
    tokenizer = SimpleCodeTokenizer()
    query_tokens = tokenizer.tokenize(query_code)
    
    # 获取 BM25 分数
    doc_scores = bm25.get_scores(query_tokens)
    
    # 组装成 Candidates 格式
    candidates = []
    for i, score in enumerate(doc_scores):
        # 归一化 BM25 分数到 0-1 (粗略)
        # 实际使用中不需要严格归一化，因为 Selector 内部会做
        candidates.append({
            "id": corpus_dicts[i]["id"],
            "code": corpus_dicts[i]["code"],
            "score": float(score)
        })
    
    # 按分数排序看看 Top-K
    candidates.sort(key=lambda x: x['score'], reverse=True)
    print("Top-3 BM25 Candidates:")
    for c in candidates[:3]:
        print(f" - [{c['score']:.4f}] {c['code'].splitlines()[0]}...")

    # ==========================================
    # 3. 运行你的 Selector
    # ==========================================
    print("\n>>> [Phase 3] Running Budget-Aware Selector...")
    # 预算设定为只够放 2 个片段 (强迫去重)
    selector = BudgetAwareSelector(alpha=1.0, beta=1.0, token_budget=20) 
    
    # 这里的 tokenizer 只是为了算长度
    selected = selector.select(query_code, candidates, tokenizer)
    
    print(f"\nResult: Selected {len(selected)} chunks.")
    for s in selected:
        print(f" [SELECTED] ID:{s['id']} | Score:{s['score']:.4f}")
        print(f"    Code: {s['code']}")

    # 验证逻辑
    selected_ids = [s['id'] for s in selected]
    # 我们希望看到 ID doc_0 (True Context) 被选中
    # 并且 doc_1, doc_2, doc_6 (Duplicates) 被剔除
    # 并且如果有余力，选点别的独特的
    
    if 'doc_0' in selected_ids and not any(x in selected_ids for x in ['doc_1', 'doc_2', 'doc_6']):
        print("\n>>> ✅ SUCCESS: High relevance selected, redundancy removed.")
    elif len(set(selected_ids) & {'doc_0', 'doc_1', 'doc_2', 'doc_6'}) > 1:
        print("\n>>> ⚠️ WARNING: Multiple redundant copies selected.")
    else:
        print("\n>>> ℹ️ INFO: Check manual output.")

if __name__ == "__main__":
    run_real_world_proof()