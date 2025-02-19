import cv2
import numpy as np
from PIL import Image
import os
 
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
     
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
         
        id, pred = clf.predict(gray_img[y:y+h,x:x+w])
        confidence = int(100*(1-pred/300))
         
        if confidence>70:
            if id==1:
                cv2.putText(img, "Ayush", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
     
    return img
 
# loading classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("recog.xml")
 
video_capture = cv2.VideoCapture(1)
 
while True:
    ret, img = video_capture.read()
    img = draw_boundary(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
    cv2.imshow("face Detection", img)
     
    if cv2.waitKey(1)==13:
        break
video_capture.release()
cv2.destroyAllWindows()
# import cv2
# import numpy as np
# import hnswlib
# from deepface import DeepFace
# from ultralytics import YOLO

# # Load YOLO Model for Face Detection
# model = YOLO('yolov11n-face.pt')

# # Load HNSW Face Embedding Index
# dim = 512  # ArcFace embedding size
# index = hnswlib.Index(space='cosine', dim=dim)
# index.load_index("face_embeddings.bin")

# # Load stored face labels and map them to image filenames
# face_db = np.load("face_db.npy", allow_pickle=True).item()

# # Assuming face_db contains {id: image_filename} format, if not, adjust as necessary.
# id_to_image = face_db  # Directly using the face_db to map IDs to image filenames

# # Open Camera
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)  # Face Detection

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             face = frame[y1:y2, x1:x2]  # Crop detected face

#             if face.size != 0:
#                 # Convert face to RGB
#                 face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

#                 try:
#                     # Extract face embedding
#                     embedding = DeepFace.represent(face_rgb, model_name="ArcFace", enforce_detection=False)
#                     if embedding:
#                         query_vector = np.array([embedding[0]["embedding"]]).astype("float32")

#                         # Normalize vector
#                         query_vector /= np.linalg.norm(query_vector)

#                         # Perform HNSW search
#                         labels, distances = index.knn_query(query_vector, k=1)

#                         # Recognize person if match is found
#                         threshold = 0.6  # You can experiment with this threshold
#                         if distances[0][0] < threshold:  # If distance is below the threshold
#                             matched_image = id_to_image.get(labels[0][0], "Unknown image")
#                         else:
#                             matched_image = "No match"

#                         # Display image filename or "No match" on screen
#                         cv2.putText(frame, f"{matched_image}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#                 except Exception as e:
#                     print(f"Error extracting embedding: {e}")

#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Display results
#     cv2.imshow("Real-Time Face Recognition", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
