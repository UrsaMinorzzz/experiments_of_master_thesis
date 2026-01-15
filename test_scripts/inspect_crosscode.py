from datasets import load_dataset

def inspect_crosscode():
    print(">>> [1/3] Loading ZHENGRAN/cross_code_eval_python (Split: TRAIN)...")
    try:
        # 修改点：split="test" -> split="train"
        ds = load_dataset("ZHENGRAN/cross_code_eval_python", split="train", streaming=True)
    except Exception as e:
        print(f"Critical Error: {e}")
        return

    print(">>> [2/3] Fetching first sample...")
    iterator = iter(ds)
    sample = next(iterator)
    
    print("\n" + "="*40)
    print(" DATASET STRUCTURE ANALYSIS")
    print("="*40)
    
    # 1. 打印所有字段名
    print(f"KEYS: {list(sample.keys())}")
    
    # 2. 智能推断
    # CrossCodeEval 的核心是：你需要补全的代码 (Query) 和 跨文件上下文 (Context)
    # 我们来看看它把上下文藏在哪个字段里了
    
    print("\n[Content Preview]:")
    for k, v in sample.items():
        # 转字符串，防止列表太长刷屏
        content = str(v)
        # 针对长代码做一个优雅的截断
        if len(content) > 200:
            content = content[:100] + " ... [Truncated] ... " + content[-50:]
        print(f" - {k}: {content}")
        
    print("="*40)

if __name__ == "__main__":
    inspect_crosscode()