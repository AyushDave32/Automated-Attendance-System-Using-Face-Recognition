import cv2
import time
import numpy as np
import faiss
import os
import dlib
from ultralytics import YOLO
from deepface import DeepFace
from huggingface_hub import hf_hub_download
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLOv8 face detection model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

# FAISS setup (512-dimensional embeddings)
embedding_dim = 512  
faiss_index = faiss.IndexFlatL2(embedding_dim)  
embedding_db = {}

# Load stored embeddings
if os.path.exists("face_index.pkl"):
    with open("face_index.pkl", "rb") as f:
        embedding_db, faiss_index = pickle.load(f)

print(f"Total faces in database: {faiss_index.ntotal}")

# Load Dlib detector and predictor
dlib_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  
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

    return img  

cap = cv2.VideoCapture(0)

cap.set(3, 640)  
cap.set(4, 480)  

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using YOLOv8
    results = model(img_rgb)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face = img_rgb[y1:y2, x1:x2]
            aligned_face = align_face(face)
            aligned_face = cv2.resize(aligned_face, (112, 112))

            try:
                embedding = DeepFace.represent(aligned_face, model_name="ArcFace", detector_backend="retinaface")

                if embedding:
                    embedding_vector = np.array(embedding[0]["embedding"]).astype("float32").reshape(1, -1)
                    embedding_vector /= np.linalg.norm(embedding_vector)

                    if faiss_index.is_trained and faiss_index.ntotal > 0:
                        distances, indices = faiss_index.search(embedding_vector, k=1)
                        closest_index = indices[0][0]
                        distance = distances[0][0]

                        name = list(embedding_db.keys())[closest_index] if distance < 1 else "Unknown1"
                    else:
                        name="unknown2"
                    print(f"Recognized: {name} (Distance: {distance:.2f})")
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "unknown3", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error: {e}")
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()