import cv2
import numpy as np
import dlib
import os
import pickle
from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO("yolov11n-face.pt")

# Export model to ONNX
model.export(format="onnx", dynamic=True)  # Will create a yolov11n-face.onnx file

    # Load dlib face recognition model
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    # File to store embeddings
ENCODINGS_FILE = "face_encodings.pkl"

    # Load known face encodings and names
known_face_encodings = []
known_face_names = []

def save_embeddings(embeddings, names, file_path=ENCODINGS_FILE):
    with open(file_path, "wb") as f:
        pickle.dump({"encodings": embeddings, "names": names}, f)

def load_embeddings(file_path=ENCODINGS_FILE):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]
    return [], []

def load_known_faces(image_folders):
    global known_face_encodings, known_face_names
    face_detector = dlib.get_frontal_face_detector()

    for folder in image_folders:
        for image_filename in os.listdir(folder):
            if image_filename.endswith((".jpg", ".png")):
                image_path = os.path.join(folder, image_filename)
                img = cv2.imread(image_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                faces = face_detector(rgb_img)
                for face in faces:
                    landmarks = sp(rgb_img, face)
                    face_descriptor = facerec.compute_face_descriptor(rgb_img, landmarks)

                    known_face_encodings.append(np.array(face_descriptor))
                    known_face_names.append(image_filename.split('.')[0])

    # Load stored embeddings or compute if not available
    known_face_encodings, known_face_names = load_embeddings()

    if not known_face_encodings:  # Compute and store only if no stored embeddings
        load_known_faces(["dataset/Ayush_1","dataset/Yash_1" ])
        save_embeddings(known_face_encodings, known_face_names)

    # Start webcam for real-time face recognition
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Run YOLO face detection

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                confidence = box.conf[0]  # Confidence score

                if confidence > 0.5:
                    cropped_face = frame[y1:y2, x1:x2]
                    rgb_cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    
                    # dlib face recognition
                    face_rect = dlib.rectangle(0, 0, cropped_face.shape[1], cropped_face.shape[0])
                    landmarks = sp(rgb_cropped_face, face_rect)
                    face_descriptor = facerec.compute_face_descriptor(rgb_cropped_face, landmarks)
                    face_descriptor = np.array(face_descriptor)

                    # Compare with known encodings
                    matches = [np.linalg.norm(face_descriptor - known) < 0.4 for known in known_face_encodings]
                    
                    if True in matches:
                        name = known_face_names[matches.index(True)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()