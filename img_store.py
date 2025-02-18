import os
import numpy as np
import hnswlib
from deepface import DeepFace

DATASET_PATH = "dataset_faces"

# Initialize HNSW Index
dim = 512  # Embedding dimension (depends on DeepFace model)
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=1000, ef_construction=200, M=16)

face_db = {}
id_counter = 0

# Process each image in the dataset
for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)
    
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)

        # Extract face embedding
        embedding = DeepFace.represent(img_path, model_name="ArcFace", enforce_detection=False)
        if embedding:
            index.add_items(np.array([embedding[0]["embedding"]]), np.array([id_counter]))
            face_db[id_counter] = person  # Map ID to name
            id_counter += 1

# Save index
index.save_index("face_embeddings.bin")
np.save("face_db.npy", face_db)
