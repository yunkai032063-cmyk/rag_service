import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3:8b"

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "請用一句話解釋什麼是 Retrieval Augmented Generation。"
        }
    ],
    "stream": False
}

response = requests.post(
    OLLAMA_URL,
    json=payload,
    timeout=120,
)
response.raise_for_status()
data = response.json()

print("LLM Response:")
print(data["message"]["content"])