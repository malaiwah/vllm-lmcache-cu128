# Import the requests library
import requests

# Define the API URL
API_URL = "http://localhost:8000/v1/responses"

# Define the API key
API_KEY = "LOCAL-ONLY-KEY"

# Define the model
MODEL = "local/vllm"

# Build a long filler (≈100k tokens when tokenized)
filler = "FILLER TEXT. " * 25000

# Construct the request payload
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

# Define the headers
headers = {"Authorization": f"Bearer {API_KEY}"}

# Send the POST request and handle the response
resp = requests.post(API_URL, json=payload, headers=headers)
resp.raise_for_status()
print(resp.json())
