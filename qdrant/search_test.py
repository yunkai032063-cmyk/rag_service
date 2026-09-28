"""
This is a simple test to check if the search functionality is working correctly. 
It sends a query to the embedding service to generate an embedding for the question, 
and then queries the Qdrant vector database to retrieve the most relevant documents based on that embedding.
Make sure that both the embedding service and the Qdrant server are running at the specified URLs before executing this code.
"""
import requests
from qdrant_client import QdrantClient

EMBEDDING_URL = "http://127.0.0.1:8000/embed"
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "documents"

question = "中央銀行負責什麼？"

response = requests.post(
    EMBEDDING_URL,
    json={
        "texts": [question],
        "input_type": "query"
    },
)
# print(response.json())
response.raise_for_status()
query_vector = response.json()["embeddings"][0]

client = QdrantClient(url=QDRANT_URL)

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3,
    with_payload=True
).points

for result in results:
    print("=" * 60)
    print("Score:", result.score)
    print("Source:", result.payload["source"])
    print("Text:", result.payload["text"]) 