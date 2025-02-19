import cv2
import os
import numpy as np
from deepface import DeepFace
import joblib  # Library to save and load data

def generate_embeddings(image_paths):
    embeddings = []
    labels = []
    
    for image_path in image_paths:
        try:
            # Read the image
            img = cv2.imread(image_path)
            # Convert to RGB for DeepFace
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Generate the face embedding using DeepFace
            embedding = DeepFace.represent(img_rgb, model_name="ArcFace", enforce_detection=False)
            embeddings.append(embedding[0]["embedding"])

            # The label for each image could be its filename (person ID)
            label = os.path.basename(image_path).split('.')[0]  # Example: 'a.1.1.jpg' -> 'a.1'
            labels.append(label)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    return embeddings, labels

# Example Usage:
# Assuming you have face images in the "dataset/Ayush" folder
image_paths = [os.path.join("dataset/Ayush", f) for f in os.listdir("dataset/Ayush") if f.endswith('.jpg')]

# Generate embeddings
embeddings, labels = generate_embeddings(image_paths)

# Save the embeddings and labels for future use
joblib.dump(embeddings, 'embeddings.pkl')  # Save embeddings to a file
joblib.dump(labels, 'labels.pkl')  # Save labels to a file

# Print confirmation
print("Embeddings and labels saved successfully.")
