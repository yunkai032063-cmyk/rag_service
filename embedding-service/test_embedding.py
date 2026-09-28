"""
This is a simple test to check if the embedding service is running and can generate embeddings for alist of texts.
Make sure that the embedding service is running at the specified URL before executing this code.
"""
import requests

URL = "http://127.0.0.1:8000/embed"

data = {
    "texts": [
        "台灣中央銀行負責制定貨幣政策。",
        "中央銀行透過貨幣政策影響經濟。",
        "我今天中午吃牛肉麵。",
    ],
    "input_type": "document",
}

response = requests.post(URL, json=data)
response.raise_for_status()
result = response.json()

print("Model:", result["model"])
print("Device:", result["device"])
print("Dimension:", result["dimension"])
for i, vector in enumerate(result["embeddings"]):
    print(
        f"text[{i}] -> dimension={len(vector)}, "
        f"first_values={vector[:5]}"
    )