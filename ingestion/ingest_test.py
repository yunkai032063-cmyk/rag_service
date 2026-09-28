"""
This is a simple test to check if the embedding service is running and can generate embeddings for a list of texts.
Make sure that the embedding service is running at the specified URL before executing this code.
"""
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

EMBEDDING_URL = "http://127.0.0.1:8000/embed"
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "documents"

documents = [
    {
        "id": 1,
        "text": "台灣中央銀行負責制定與執行貨幣政策。",
        "source": "example.txt",
    },
    {
        "id": 2,
        "text": "Qdrant 是一個向量資料庫，可以進行相似度搜尋。",
        "source": "example.txt",
    },
    {
        "id": 3,
        "text": "Python 可以透過 FastAPI 建立 HTTP API。",
        "source": "example.txt"
    },
]
texts = [doc["text"] for doc in documents]

response = requests.post(
    EMBEDDING_URL,
    json={
        "texts": texts,
        "input_type": "document",
    },
)
response.raise_for_status()
result = response.json()
vectors = result["embeddings"]

client = QdrantClient(url=QDRANT_URL)

points = []
for doc, vector in zip(documents, vectors):
    points.append(
        PointStruct(
            id=doc["id"],
            vector=vector,
            payload={
                "text": doc["text"],
                "source": doc["source"]
            },
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

print(f"Inserted {len(points)} documents.")