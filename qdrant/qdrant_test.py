""" 
This is a simple test to check if the Qdrant client can connect to the Qdrant server and retrieve the list of collections. 
Make sure that the Qdrant server is running at the specified URL before executing this code.
"""
from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://127.0.0.1:6333"
)

print(client.get_collections())