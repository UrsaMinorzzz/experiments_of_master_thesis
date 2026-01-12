import os, requests

key = os.environ["OPENROUTER_API_KEY"]
r = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {key}"}
)
r.raise_for_status()
models = r.json()["data"]

def find(q):
    q = q.lower()
    hits = [m for m in models if q in m["id"].lower() or q in m.get("name","").lower()]
    return [(m["id"], m.get("context_length")) for m in hits][:50]

print("codellama hits:", find("codellama"))
print("qwen2.5-coder hits:", find("qwen2.5-coder"))
print("starcoder2-15b-instruct hits:", find("starcoder"))
