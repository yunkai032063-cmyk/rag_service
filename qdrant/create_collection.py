"""
This script creates a collection in Qdrant if it does not already exist.
Make sure that the Qdrant server is running at the specified URL before executing this code.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "documents"
VECTOR_SIZE = 1024

client = QdrantClient(url=QDRANT_URL)
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(f"Collection created: {COLLECTION_NAME}")
else:
    print(f"Collection already exists: {COLLECTION_NAME}")

print(client.get_collections())