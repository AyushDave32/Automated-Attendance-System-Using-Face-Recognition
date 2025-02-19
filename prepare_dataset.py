import cv2
import os

def gen_data():
    face_class = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def crop_face(img):
        gray_sc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_class.detectMultiScale(gray_sc, 1.3, 5)

        if len(faces) == 0:  # Fix: Correct way to check if faces were detected
            return None
        
        for (x, y, w, h) in faces:
            crop_face = img[y:y+h, x:x+w]
            return crop_face  # Fix: Return the first detected face
    
    cap = cv2.VideoCapture(0)
    id = 1
    img_id = 0

    while True:
        ret, frame = cap.read()
        face = crop_face(frame)  # Fix: Call the correct function
        
        if face is not None:
            img_id += 1
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            dataset_path = f"dataset/Ayush/a.{id}.{img_id}.jpg"
            cv2.imwrite(dataset_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

            cv2.imshow("Cropped face", face)

            if cv2.waitKey(1) == 13 or img_id == 250:  # Stop at 250 images
                break

    cap.release()
    cv2.destroyAllWindows()

# Run the dataset generation function
gen_data()
