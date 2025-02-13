# import cv2
# import time
# import logging
# from ultralytics import YOLO
# from deepface import DeepFace
# from huggingface_hub import hf_hub_download

# # Configure logging
# logging.basicConfig(level=logging.ERROR)

# # Download YOLO model
# model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
# model = YOLO(model_path)

# # Open webcam (0 for built-in, 1 for external)
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)  # Width
# cap.set(4, 480)  # Height

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     start_time = time.time()
#     results = model(frame)

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Crop face
#             face = frame[y1:y2, x1:x2]

#             # Recognize face
#             try:
#                 result = DeepFace.find(face, db_path="Database", model_name="ArcFace", enforce_detection=False)
#                 if result and len(result[0]) > 0:
#                     name = result[0]['identity'][0].split("\\")[-1]
#                     cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 else:
#                     cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             except Exception as e:
#                 logging.error(f"Face recognition error: {e}")

#     # Display FPS
#     fps = 1 / (time.time() - start_time)
#     cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#     # Show output
#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import os
import shutil
import random

# Set paths
train_images_dir = "dataset/images/train/images"
train_labels_dir = "dataset/images/train/labels"
val_images_dir = "dataset/images/valid/images"
val_labels_dir = "dataset/images/valid/labels"

# Ensure validation directories exist
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get list of image files (assuming images are .jpg and labels are .txt)
image_files = [f for f in os.listdir(train_images_dir) if f.endswith(".jpg")]

# Define percentage of files to move
val_split = 0.2  # Move 20% to validation set
num_to_move = int(len(image_files) * val_split)

# Randomly select files to move
val_images = random.sample(image_files, num_to_move)

# Move files
for img in val_images:
    label = img.replace(".jpg", ".txt")  # Assuming label file has the same name
    shutil.move(os.path.join(train_images_dir, img), os.path.join(val_images_dir, img))
    if os.path.exists(os.path.join(train_labels_dir, label)):
        shutil.move(os.path.join(train_labels_dir, label), os.path.join(val_labels_dir, label))

print(f"Moved {num_to_move} images and their labels to the validation set.")
    