import os
import cv2
import numpy as np
from PIL import Image

def training(data_dir):
    dataset_path = "dataset/Ayush"
    face_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    faces = []
    ids = []

    for img in face_images:
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)  # Fix: Read image correctly
        if image is None:
            print(f"Skipping invalid image: {img}")
            continue  # Skip if image is not readable
        
        imageNP = np.array(image, 'uint8')  # Convert to NumPy array
        id = int(os.path.split(img)[1].split(".")[1])  # Extract ID from filename

        faces.append(imageNP)
        ids.append(id)

    ids = np.array(ids)  # Fix: Correctly convert IDs to NumPy array

    t_clf = cv2.face.LBPHFaceRecognizer_create()
    t_clf.train(faces, ids)
    t_clf.write("recog.xml")

training("dataset/Ayush")


# # List to store embeddings and face IDs
# embeddings = []
# face_db = {}  # Dictionary to map face ID to image filename

# for idx, face_image in enumerate(face_images):
#     try:
#         # Load the image
#         image = cv2.imread(face_image)

#         # Convert the image to grayscale
#         grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Generate embedding for each face image using DeepFace
#         embedding = DeepFace.represent(grayscale_image, model_name="ArcFace", enforce_detection=False)
#         embeddings.append(embedding[0]["embedding"])

#         # Add the image filename to the face_db dictionary
#         face_db[idx] = face_image  # Mapping face ID (idx) to image filename

#     except Exception as e:
#         print(f"Error processing {face_image}: {e}")

# # Convert embeddings to numpy array
# embeddings = np.array(embeddings)

# # Save the embeddings and the face_db for future recognition
# np.save("face_embeddings.npy", embeddings)
# np.save("face_db.npy", face_db)

# print("Embeddings and face_db saved successfully.")
