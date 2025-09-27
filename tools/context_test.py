# Import the requests library
import requests

# Define the API URL
API_URL = "http://localhost:8000/v1/responses"

# Define the API key
API_KEY = "LOCAL-ONLY-KEY"

# Define the model
MODEL = "gpt-5"

# Define the instructions
INSTRUCTIONS = "You are a careful assistant."

# Define the input
INPUT = "Hello!"

# Construct the request payload
payload = {
    "model": MODEL,
    "instructions": INSTRUCTIONS,
    "input": INPUT
}

# Define the headers
headers = {"Authorization": f"Bearer {API_KEY}"}

# Send the POST request and handle the response
resp = requests.post(API_URL, json=payload, headers=headers)
resp.raise_for_status()
print(resp.json())
