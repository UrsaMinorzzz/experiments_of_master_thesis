from selector import BudgetAwareSelector
import re

# 1. Mock Tokenizer
class MockTokenizer:
    def encode(self, text):
        # 简单模拟：假设平均每个单词算 1.3 个 token
        # 这里只为了计算长度消耗
        return text.split()

# 2. 构造测试数据
query = "def matrix_multiply(a, b):\n    result = []"

candidates = [
    # Case A: 标准紧凑写法 (The Anchor)
    {
        "id": "A (Standard)", 
        "code": "for i in range(len(a)):\n    val=a[i][k]*b[k][j]",
        "score": 0.95 
    },
    
    # Case B: 格式化/空格变体 (The Trap)
    # 对于 split() 来说，"val=a[i][k]" 和 "val = a [ i ] [ k ]" 是完全不同的字符串。
    # 但对于 Tokenizer + N-gram，它们的 token 序列是一样的。
    {
        "id": "B (Spaced)",
        "code": "for i in range( len( a ) ):\n    val = a [ i ] [ k ] * b [ k ] [ j ]",
        "score": 0.94
    }, 
    
    # Case C: 独特信息 (Unique)
    {
        "id": "C (Unique)",
        "code": "if len(a[0]) != len(b):\n    raise ValueError('Dimension Mismatch')",
        "score": 0.80
    }
]

def run_test():
    print(">>> [实验] Token-level N-gram Robustness Test")
    print(">>> 目的: 验证算法能否识别'格式不同但逻辑相同'的代码冗余\n")
    
    # 初始化：Token 预算设为极小，强迫做二选一
    # 假设 A 占 20，B 占 30 (因为空格多)，C 占 20。预算 50 只能选两个。
    selector = BudgetAwareSelector(alpha=1.0, beta=1.0, token_budget=50)
    tokenizer = MockTokenizer()
    
    # --- 关键验证: Jaccard 比较 ---
    print("--- Step 1: Jaccard 相似度诊断 ---")
    set_a = selector._tokenize_to_set(candidates[0]['code'], n=3)
    set_b = selector._tokenize_to_set(candidates[1]['code'], n=3)
    
    # 计算 A 和 B 的相似度
    jaccard_ab = selector._jaccard_similarity(set_a, set_b)
    
    print(f"Candidate A Tokens (Preview): {list(set_a)[:3]}...")
    print(f"Candidate B Tokens (Preview): {list(set_b)[:3]}...")
    print(f"Jaccard(A, B) Score: {jaccard_ab:.4f}")
    
    if jaccard_ab > 0.8:
        print(" -> ✅ Pass: 高相似度！Regex Tokenizer 成功忽略了空格差异。")
    elif jaccard_ab < 0.2:
        print(" -> ❌ Fail: 低相似度。Tokenizer 未生效，split() 被空格欺骗了。")
    else:
        print(" -> ⚠️ Warning: 相似度中等，N-gram size 可能需要调整。")

    # --- 运行选择流程 ---
    print("\n--- Step 2: 运行 Select ---")
    selected = selector.select(query, candidates, tokenizer)
    selected_ids = [c['id'] for c in selected]
    print(f"最终选中: {selected_ids}")
    
    # 预期结果：选中 A (最高分) 和 C (独特)。剔除 B (因为和 A 太像)。
    if "A (Standard)" in selected_ids and "B (Spaced)" not in selected_ids:
        print(" -> ✅ 成功: B 被判定为 A 的冗余项并被剔除。")
    elif "B (Spaced)" in selected_ids:
        print(" -> ❌ 失败: B 没有被剔除，冗余惩罚力度不足。")
    
    # 额外检查 Gamma
    raw_scores = [c['score'] for c in candidates]
    norm_scores = selector._normalize_scores(raw_scores)
    sets = [selector._tokenize_to_set(c['code']) for c in candidates]
    gamma = selector._calculate_adaptive_gamma(norm_scores, sets)
    print(f"\n[Debug] Adaptive Gamma: {gamma:.4f}")

if __name__ == "__main__":
    run_test()