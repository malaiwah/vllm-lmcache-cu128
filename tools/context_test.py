import requests

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"

# Build a long filler (≈100k tokens when tokenized)
filler = "FILLER TEXT. " * 24000

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

resp = requests.post(API_URL, json=payload)
resp.raise_for_status()
print(resp.json())
