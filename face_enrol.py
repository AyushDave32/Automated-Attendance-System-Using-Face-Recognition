import faiss
import numpy as np
import pickle
import cv2
import os
import dlib
from deepface import DeepFace
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Define folder path containing images
folder_path = "./Database/Yash"

# Load YOLOv11 face detection model
model = YOLO('yolo11n.pt') 

# Initialize FAISS for face embeddings
embedding_dim = 512  
faiss_index = faiss.IndexFlatL2(embedding_dim)  
embedding_db = {}

# Load FAISS index if available
if os.path.exists("face_index.pkl"):
    with open("face_index.pkl", "rb") as f:
        embedding_db, faiss_index = pickle.load(f)

# Load Dlib's face detector and landmark predictor
dlib_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download this file from dlib
landmark_predictor = dlib.shape_predictor(predictor_path)

def align_face(img):
    """Detects eyes using Dlib and aligns the face."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = dlib_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)

        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Compute center and rotate
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        return aligned

    return img  # Return original if no face detected

# Process images
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(folder_path, filename)
        name = os.path.splitext(filename)[0]  

        # Load and convert image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face using YOLOv11
        results = model(img_rgb)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                face = img_rgb[y1:y2, x1:x2]  

                # Align face
                aligned_face = align_face(face)
                aligned_face = cv2.resize(aligned_face, (112, 112))  

                try:
                    embedding = DeepFace.represent(aligned_face, model_name="ArcFace",enforce_detection=False)
                    if embedding:
                        embedding_vector = np.array(embedding[0]["embedding"]).astype("float32").reshape(1, -1)
                        embedding_vector /= np.linalg.norm(embedding_vector)

                        faiss_index.add(embedding_vector)  
                        embedding_db[name] = embedding_vector
                        print(f"✅ Added {name} to FAISS database.")
                except Exception as e:
                    print(f"❌ Error processing {name}: {e}")

# Save FAISS index
with open("face_index.pkl", "wb") as f:
    pickle.dump((embedding_db, faiss_index), f)

print("🎯 Face embeddings saved in FAISS!")