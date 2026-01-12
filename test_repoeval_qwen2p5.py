# sample_repeval_openrouter.py
import os
import json
import time
import random
import argparse
from typing import Any, Dict, List, Optional

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_KEYS = ["prompt", "input", "instruction", "text", "query", "intent"]

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def pick_prompt(ex: Dict[str, Any]) -> str:
    for k in PROMPT_KEYS:
        v = ex.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
    # 有些数据会嵌套在 fields 里
    if "fields" in ex and isinstance(ex["fields"], dict):
        for k in PROMPT_KEYS:
            v = ex["fields"].get(k, None)
            if isinstance(v, str) and v.strip():
                return v
    raise KeyError(f"Cannot find prompt field in example keys={list(ex.keys())}")

def openrouter_chat(
    model: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: Optional[int] = 42,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Missing env var OPENROUTER_API_KEY")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # 下面两个头不是必须，但 OpenRouter 文档/社区常建议加，便于统计与溯源
        "X-Title": "RepoEval-10shot-sanity",
        "HTTP-Referer": "http://localhost",
    }

    system_msg = (
        "You are a code completion assistant. "
        "Complete the missing code. "
        "Output ONLY the code completion (no explanations, no markdown)."
    )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:1000]}")
    return r.json()

def extract_text(resp: Dict[str, Any]) -> str:
    # OpenAI-compatible: choices[0].message.content
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to RepoEval jsonl (or your prepared prompt jsonl)")
    ap.add_argument("--out", default="repeval_10_outputs.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--model", default="qwen/qwen2.5-coder-7b-instruct")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests (seconds)")
    args = ap.parse_args()

    random.seed(args.seed)
    data = load_jsonl(args.data)
    if args.shuffle:
        random.shuffle(data)

    subset = data[: args.n]
    print(f"Loaded {len(data)} examples, running {len(subset)} with model={args.model}")

    with open(args.out, "w", encoding="utf-8") as w:
        for i, ex in enumerate(subset):
            prompt = pick_prompt(ex)
            resp = openrouter_chat(
                model=args.model,
                user_prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed + i,
            )
            pred = extract_text(resp)

            rec = {
                "idx": i,
                "model": args.model,
                "prompt_key_used": next((k for k in PROMPT_KEYS if isinstance(ex.get(k), str) and ex.get(k).strip()), None),
                "prompt_preview": prompt[:200],
                "prediction": pred,
                "usage": resp.get("usage", None),
                "raw_id": resp.get("id", None),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[{i+1}/{len(subset)}] done | prompt_tokens={rec['usage'].get('prompt_tokens') if rec['usage'] else None} "
                  f"completion_tokens={rec['usage'].get('completion_tokens') if rec['usage'] else None}")
            time.sleep(args.sleep)

    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
