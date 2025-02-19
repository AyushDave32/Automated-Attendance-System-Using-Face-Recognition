import cv2
import numpy as np
import hnswlib
from deepface import DeepFace
from ultralytics import YOLO

# Load YOLO Model for Face Detection
model = YOLO('yolov11n-face.pt')

# Load HNSW Face Embedding Index
dim = 512  # ArcFace embedding size
index = hnswlib.Index(space='cosine', dim=dim)
index.load_index("face_embeddings.bin")

# Load stored face labels and map them to image filenames
face_db = np.load("face_db.npy", allow_pickle=True).item()

# Assuming face_db contains {id: image_filename} format, if not, adjust as necessary.
id_to_image = face_db  # Directly using the face_db to map IDs to image filenames

# Open Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Face Detection

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]  # Crop detected face

            if face.size != 0:
                # Convert face to RGB
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                try:
                    # Extract face embedding
                    embedding = DeepFace.represent(face_rgb, model_name="ArcFace", enforce_detection=False)
                    if embedding:
                        query_vector = np.array([embedding[0]["embedding"]]).astype("float32")

                        # Normalize vector
                        query_vector /= np.linalg.norm(query_vector)

                        # Perform HNSW search
                        labels, distances = index.knn_query(query_vector, k=1)

                        # Recognize person if match is found
                        threshold = 0.6  # You can experiment with this threshold
                        if distances[0][0] < threshold:  # If distance is below the threshold
                            matched_image = id_to_image.get(labels[0][0], "Unknown image")
                        else:
                            matched_image = "No match"

                        # Display image filename or "No match" on screen
                        cv2.putText(frame, f"{matched_image}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                except Exception as e:
                    print(f"Error extracting embedding: {e}")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display results
    cv2.imshow("Real-Time Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
