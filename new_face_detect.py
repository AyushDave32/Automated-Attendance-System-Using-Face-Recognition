# import cv2
# import time
# import numpy as np
# import os
# from ultralytics import YOLO
# import face_recognition

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Load YOLO model
# model = YOLO('yolov11n-face.pt')

# # Path to known faces database
# database_path = "images/Ayush"  # Change this to your actual database folder

# # Load known face encodings
# known_face_encodings = []
# known_face_filenames = []

# for file in os.listdir(database_path):
#     img_path = os.path.join(database_path, file)
#     img = face_recognition.load_image_file(img_path)
#     encodings = face_recognition.face_encodings(img)

#     if len(encodings) > 0:
#         known_face_encodings.append(encodings[0])
#         known_face_filenames.append(file)  # Store full filename (e.g., ayush_1.jpg)

# # Open Webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         break

#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect faces using YOLO
#     results = model(img_rgb)

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             # Extract face ROI
#             face_roi = img_rgb[y1:y2, x1:x2]

#             # Encode face using face_recognition
#             face_encoding = face_recognition.face_encodings(face_roi)
#             if len(face_encoding) > 0:
#                 face_encoding = face_encoding[0]

#                 # Compare with known faces
#                 matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                 face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

#                 if True in matches:
#                     print("Face Matched!")
#                     cap.release()
#                     cv2.destroyAllWindows()
#                     exit()  # Exit program immediately

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os

# Load images and labels
import os
import cv2
import numpy as np

def load_images_and_labels(dataset_path):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        print(f"Checking directory: {person_path}")  # Debugging output
        if os.path.isdir(person_path):
            label_map[current_label] = person_name
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                print(f"Loading image: {image_path}")  # Debugging output

                # Ensure that the image path is correct and accessible
                if os.path.isfile(image_path):
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        print(f"Image loaded successfully: {image_path}")
                        print(f"Image shape: {img.shape}")  # Debugging output
                        img = cv2.resize(img, (200, 200))  # Resize to consistent size
                        images.append(img.flatten())  # Flatten the image
                        labels.append(current_label)
                    else:
                        print(f"Warning: Failed to load image {image_path}")
                else:
                    print(f"Error: {image_path} is not a valid file.")
            current_label += 1

    images = np.array(images)
    print(f"Loaded {images.shape[0]} images.")  # Debugging output
    return images, np.array(labels), label_map

def apply_pca(images, n_components=100):
    from sklearn.decomposition import PCA
    
    if images.size == 0:
        raise ValueError("No images to apply PCA on.")

    print(f"Applying PCA to {images.shape[0]} images...")
    pca = PCA(n_components=n_components)
    reduced_images = pca.fit_transform(images)
    
    return pca, reduced_images

def face_recognition_system(dataset_path):
    images, labels, label_map = load_images_and_labels(dataset_path)

    if images.size == 0:
        print("No images found, exiting...")
        return

    # Apply PCA to the images
    pca, reduced_images = apply_pca(images)

    print("PCA applied successfully.")
    # Further processing can continue here...

dataset_path = "dataset/Ayush/"  # Adjust this path as per your setup
face_recognition_system(dataset_path)



