import cv2
import numpy as np
import hnswlib
from deepface import DeepFace
from ultralytics import YOLO

# Load YOLO Model
model = YOLO('yolov11n-face.pt')

# Load HNSW Index
index = hnswlib.Index(space='cosine', dim=512)
index.load_index("face_embeddings.bin")
face_db = np.load("face_db.npy", allow_pickle=True).item()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Face detection

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]  # Crop detected face
            
            if face.size != 0:
                embedding = DeepFace.represent(face, model_name="ArcFace", enforce_detection=False)
                if embedding:
                    query_vector = np.array([embedding[0]["embedding"]])
                    labels, distances = index.knn_query(query_vector, k=1)

                    if distances[0][0] < 0.5:  # Threshold for match
                        name = face_db[labels[0][0]]
                        cv2.putText(frame, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
