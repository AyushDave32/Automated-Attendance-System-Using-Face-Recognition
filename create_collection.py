from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient("http://127.0.0.1:6333")
collection_name = "face_collection"
if client.collection_exists(collection_name):
    print(f"Deleting existing collection: {collection_name}")
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=128, distance=Distance.COSINE)
)
print(f"Collection '{collection_name}' created successfully!")

