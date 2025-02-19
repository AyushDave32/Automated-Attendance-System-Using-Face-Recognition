import cv2
import os
import torch
from ultralytics import YOLO

# Load YOLOv11 face detection model
model = YOLO("yolov11n-face.pt")  # Ensure you have the correct model file

def crop_face(img):
    results = model(img)  # Perform face detection
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

            # Crop the detected face
            crop_face = img[y1:y2, x1:x2]
            return crop_face  # Return the first detected face
    return None

def gen_data():
    cap = cv2.VideoCapture(0)
    person_id = 1  # Start with person 1
    img_id = 0

    os.makedirs("dataset/Ayush", exist_ok=True)  # Ensure the dataset folder exists

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue
        
        face = crop_face(frame)  
        
        if face is not None:
            img_id += 1
            face = cv2.resize(face, (200, 200))  # Resize face image
            dataset_path = f"dataset/Ayush/p.{person_id}.{img_id}.jpg"
            cv2.imwrite(dataset_path, face)  # Save the face image in color (BGR)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

            cv2.imshow("Cropped Face", face)

            # If 200 images are captured for one person, change the person_id
            if img_id == 200:
                person_id += 1  # Switch to the next person
                img_id = 0  # Reset image id for the new person

            if cv2.waitKey(1) == 13:  # Press Enter to stop capturing
                break

    cap.release()
    cv2.destroyAllWindows()

# Run the dataset generation function
gen_data()

