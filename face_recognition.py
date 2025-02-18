import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
from mtcnn import MTCNN
from deepface import DeepFace
import tensorflow as tf
import warnings

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLO model
model = YOLO('yolov11n-face.pt')

# Load MTCNN for face alignment
detector = MTCNN()

# Path to store dataset
DATASET_PATH = "dataset_faces"
os.makedirs(DATASET_PATH, exist_ok=True)

# Ask for person's name
user_name = input("Enter person's name: ")
user_folder = os.path.join(DATASET_PATH, user_name)
os.makedirs(user_folder, exist_ok=True)

img_count = 0  # Counter for images

def align_face_mtcnn(img):
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return img  # Return original if no face found

    keypoints = faces[0]['keypoints']
    left_eye, right_eye = keypoints['left_eye'], keypoints['right_eye']

    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    aligned = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    
    return aligned

cap = cv2.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using YOLO
    results = model(img_rgb)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract face
            face = img_rgb[y1:y2, x1:x2]
            
            if face.size == 0:
                continue  # Skip if face extraction fails
            
            # Align and resize face
            aligned_face = align_face_mtcnn(face)
            aligned_face = cv2.resize(aligned_face, (112, 112))

            # Save the face image
            img_path = os.path.join(user_folder, f"{user_name}_{img_count}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
            img_count += 1

            try:
                # Extract face embeddings
                embedding = DeepFace.represent(aligned_face, model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)

                if embedding:
                    embedding_vector = np.array(embedding[0]["embedding"]).astype("float32")
                    embedding_vector /= np.linalg.norm(embedding_vector)  # Normalize before storing


                    print(f"Image {img_count} saved: {img_path}")
            except Exception as e:
                print(f"Error extracting embedding: {e}")

            if img_count >= 20:  # Stop after collecting 20 images
                break

    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Face Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q") or img_count >= 20:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Face collection complete! {img_count} images saved for '{user_name}'.")
