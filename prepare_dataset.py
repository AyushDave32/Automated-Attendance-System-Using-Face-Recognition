import cv2
import os

# Initialize Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up webcam capture
cap = cv2.VideoCapture(0)

# Create a folder to store the dataset
dataset_path = "dataset/Ayush/"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize a counter for the number of images
count = 0

# Give the user time to get ready for the camera
print("Please look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale (Haar classifier works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop the detected face from the frame
        face = frame[y:y+h, x:x+w]
        
        # Save the face image to the dataset folder
        face_filename = os.path.join(dataset_path, f"face_{count}.jpg")
        cv2.imwrite(face_filename, face)
        
        # Increment the counter
        count += 1
        
        # Stop after collecting a fixed number of faces (e.g., 100 images)
        if count >= 1000:
            print("Dataset creation complete.")
            break
    
    # Display the frame with the detected face
    cv2.imshow("Face Capture", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
