import requests

API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "LOCAL-ONLY-KEY"
MODEL = "local/vllm"

# Build a long filler (≈100k tokens when tokenized)
filler = "FILLER TEXT. " * 6000

# Construct the request
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": f"{filler}\n\nAt the end of all this filler, please just reply with the word: SUCCESS. \n---\nNow: reply ONLY with SUCCESS."},
    ],
    "max_tokens": 8,
    "temperature": 0,
    "top_p": 1,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_tokens": 20
}

headers = {"Authorization": f"Bearer {API_KEY}"}

resp = requests.post(API_URL, json=payload, headers=headers)
resp.raise_for_status()
print(resp.json())
