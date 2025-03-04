import cv2
import os
import numpy as np
from deepface import DeepFace
import joblib  # Library to save and load data

def generate_embeddings_from_folders(folder_paths):
    embeddings = []
    labels = []
    
    for folder in folder_paths:
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder, filename)
                
                try:
                    # Read the image
                    img = cv2.imread(image_path)
                    # Convert to RGB for DeepFace
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Generate the face embedding using DeepFace
                    embedding = DeepFace.represent(img_rgb, model_name="ArcFace", enforce_detection=False)
                    embeddings.append(embedding[0]["embedding"])

                    # Use folder name + filename as label to differentiate persons
                    person_name = os.path.basename(folder)  # Get the folder name (person's name)
                    image_label = f"{person_name}_{filename.split('.')[0]}"  # e.g., 'Ayush_a1'
                    labels.append(image_label)

                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
    
    return embeddings, labels

# List of dataset folders
dataset_folders = ["dataset/Ayush", "dataset/Yashvee"]  # Add multiple folders here

# Generate embeddings for all folders
embeddings, labels = generate_embeddings_from_folders(dataset_folders)

# Save the embeddings and labels for future use
joblib.dump(embeddings, 'embeddings.pkl')  # Save embeddings to a file
joblib.dump(labels, 'labels.pkl')  # Save labels to a file

# Print confirmation
print(f"Processed {len(embeddings)} images from {len(dataset_folders)} folders.")
print("Embeddings and labels saved successfully.")
