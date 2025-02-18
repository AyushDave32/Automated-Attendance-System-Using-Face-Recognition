import os
from deepface import DeepFace
import numpy as np

# Load images from the dataset folder
dataset_path = "dataset/Ayush"
face_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]

# List to store embeddings and face IDs
embeddings = []
face_db = {}  # Dictionary to map face ID to image filename

for idx, face_image in enumerate(face_images):
    try:
        # Generate embedding for each face image
        embedding = DeepFace.represent(face_image, model_name="ArcFace", enforce_detection=False)
        embeddings.append(embedding[0]["embedding"])
        
        # Add the image filename to the face_db dictionary
        face_db[idx] = face_image  # Mapping face ID (idx) to image filename
    except Exception as e:
        print(f"Error processing {face_image}: {e}")

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

# Save the embeddings and the face_db for future recognition
np.save("face_embeddings.npy", embeddings)
np.save("face_db.npy", face_db)

print("Embeddings and face_db saved successfully.")
