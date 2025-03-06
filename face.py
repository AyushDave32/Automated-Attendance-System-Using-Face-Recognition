# face_recognition_module.py
import cv2
import numpy as np
import faiss
import os
import dlib
import pickle
import torch
import sqlite3
from deepface import DeepFace
from ultralytics.utils.ops import non_max_suppression
import onnxruntime as ort
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLOv11 ONNX model
onnx_path = "yolov11n-face.onnx"
ort_session = ort.InferenceSession(onnx_path)

# FAISS setup
embedding_dim = 512
faiss_index = faiss.IndexFlatL2(embedding_dim)
embedding_db = {}

if os.path.exists("face_index.pkl"):
    with open("face_index.pkl", "rb") as f:
        embedding_db, faiss_index = pickle.load(f)

print(f"Total faces in database: {faiss_index.ntotal}")

# Dlib setup
dlib_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(predictor_path)

# Global variables
detection_tracker = {}
logged_names = {}
cap = None
recognition_running = False

# SQLite setup
def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  date TEXT NOT NULL,
                  time TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS employees
                 (employee_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  email TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "mailto:ybalar500@gmail.com"
SMTP_PASSWORD = "vhgy unuz ffgr hukk"
FROM_EMAIL = SMTP_USER

def send_email(to_email, employee_name, timestamp):
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H:%M:%S")
    subject = f"Entry Notification - {date_str}"
    body = f"Dear {employee_name},\n\nYou entered on {date_str} at {time_str} for the first time today.\n\nBest regards,\nYour Security Team"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"Email sent to {to_email} for {employee_name}")
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")

# Utility Functions
def preprocess_onnx(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def postprocess_onnx(outputs, orig_h, orig_w):
    outputs = torch.from_numpy(outputs)
    outputs = non_max_suppression(outputs, 0.5, 0.5)
    detections = []
    scale_h, scale_w = orig_h / 640, orig_w / 640
    for output in outputs:
        if output is not None:
            for *xyxy, conf, cls in output:
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1, x2, y2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
                detections.append([x1, y1, x2, y2, float(conf), int(cls)])
    return detections

def align_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = dlib_detector(gray)
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        return aligned
    return img

def extract_embedding(face):
    if face is None or face.size == 0:
        print("Empty face crop detected.")
        return None
    try:
        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(face, model_name="Facenet512", enforce_detection=False)
        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], dict):
                embedding_vector = np.array(embedding[0].get("embedding", []), dtype=np.float32)
            else:
                embedding_vector = np.array(embedding, dtype=np.float32)
            if embedding_vector.size == embedding_dim:
                embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
                return embedding_vector.reshape(1, -1)
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
    return None

def log_to_db(name, timestamp):
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H:%M:%S")
    try:
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
                  (name, date_str, time_str))
        conn.commit()
        print(f"Logged to SQLite: {name} at {date_str} {time_str}")
    except sqlite3.Error as e:
        print(f"Error writing to database: {e}")
    finally:
        conn.close()

def get_employee_email(name):
    try:
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()
        print("name----------------", name)
        c.execute("SELECT email FROM employees WHERE name = ?", (name,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else None
    except sqlite3.Error as e:
        print(f"Error retrieving email: {e}")
        return None

async def store_embeddings(image_folder: str):
    global faiss_index, embedding_db
    if not os.path.exists(image_folder):
        raise Exception(f"Folder '{image_folder}' not found.")
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}, skipping.")
            continue
        img_h, img_w, _ = img.shape
        img_input = preprocess_onnx(img.copy())
        ort_inputs = {ort_session.get_inputs()[0].name: img_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        detections = postprocess_onnx(ort_outputs[0], img_h, img_w)
        for x1, y1, x2, y2, confidence, class_id in detections:
            if confidence > 0.7 and class_id == 0:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    print(f"Invalid bounding box for {filename}: ({x1}, {y1}, {x2}, {y2})")
                    continue
                face = img[y1:y2, x1:x2].copy()
                embedding = extract_embedding(face)
                if embedding is not None:
                    name = os.path.splitext(filename)[0]
                    faiss_index.add(embedding)
                    embedding_db[name] = embedding
                    print(f"Stored: {name} (Confidence: {confidence:.2f})")
    with open("face_index.pkl", "wb") as f:
        pickle.dump((embedding_db, faiss_index), f)
    print(f"Database updated. Total faces: {faiss_index.ntotal}")
    return {"status": "success", "total_faces": faiss_index.ntotal}

def recognize_live_task():
    global cap, recognition_running, detection_tracker, logged_names
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    try:
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()
        c.execute("SELECT DISTINCT date, name FROM attendance")
        for date, name in c.fetchall():
            if date not in logged_names:
                logged_names[date] = set()
            logged_names[date].add(name)
        conn.close()
    except sqlite3.Error as e:
        print(f"Error loading existing data: {e}")
    last_checked_date = None
    recognition_running = True
    while recognition_running:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error capturing frame.")
            break
        img_h, img_w, _ = frame.shape
        img_input = preprocess_onnx(frame.copy())
        ort_inputs = {ort_session.get_inputs()[0].name: img_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        detections = postprocess_onnx(ort_outputs[0], img_h, img_w)
        current_time = datetime.now()
        current_date_str = current_time.strftime("%Y-%m-%d")
        if last_checked_date != current_date_str:
            if current_date_str not in logged_names:
                logged_names[current_date_str] = set()
            last_checked_date = current_date_str
            print(f"Date changed to {current_date_str}. Switching to new day.")
        names_to_remove = []
        for name, data in list(detection_tracker.items()):
            if data["times"] and (current_time - data["times"][-1]) > timedelta(seconds=2) and data["count"] == 1:
                names_to_remove.append(name)
        for name in names_to_remove:
            del detection_tracker[name]
            print(f"Removed {name} from detection_tracker due to no detection in 2 seconds.")
        for x1, y1, x2, y2, confidence, class_id in detections:
            if confidence > 0.4 and class_id == 0:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 <= x1 or y2 <= y1:
                    print(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
                    continue
                face = frame[y1:y2, x1:x2].copy()
                aligned_face = align_face(face)
                aligned_face = cv2.resize(aligned_face, (112, 112))
                embedding = extract_embedding(aligned_face)
                if embedding is not None and faiss_index.ntotal > 0:
                    distances, indices = faiss_index.search(embedding, k=1)
                    closest_index = indices[0][0]
                    distance = distances[0][0]
                    name = list(embedding_db.keys())[closest_index] if distance < 0.4 else "Unknown"
                    if name.find("_") != -1:
                        name = name.split("_")[0]
                    print(f"Recognized: {name} (Distance: {distance:.2f}, Confidence: {confidence:.2f})")
                    if name != "Unknown":
                        if name not in detection_tracker:
                            detection_tracker[name] = {"count": 0, "times": []}
                        detection_tracker[name]["times"].append(current_time)
                        detection_tracker[name]["count"] += 1
                        two_sec_ago = current_time - timedelta(seconds=2)
                        detection_tracker[name]["times"] = [t for t in detection_tracker[name]["times"] if t > two_sec_ago]
                        detection_tracker[name]["count"] = len(detection_tracker[name]["times"])
                        if detection_tracker[name]["count"] > 3 and name not in logged_names[current_date_str]:
                            log_to_db(name, current_time)
                            logged_names[current_date_str].add(name)
                            email = get_employee_email(name)
                            print("email: ", email)
                            if email:
                                send_email(email, name, current_time)
                                print("email sent")
                            else:
                                print("email not sent - no email found")
                            detection_tracker[name] = {"count": 0, "times": []}
    cap.release()
    recognition_running = False

def capture_images(name: str):
    folder_path = f"Database/{name}"
    os.makedirs(folder_path, exist_ok=True)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise Exception("Error: Could not access the webcam.")
    num_photos = 100
    delay = 70
    captured_count = 0
    print(f"📸 Capturing {num_photos} images for '{name}'. Press 'q' to quit.")
    for i in range(1, num_photos + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to capture image {i}. Skipping...")
            continue
        image_path = os.path.join(folder_path, f"{name}_{i:03d}.jpg")
        cv2.imwrite(image_path, frame)
        captured_count += 1
        cv2.putText(frame, f"Photo {i}/{num_photos}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            print("🚪 Exiting early...")
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ {captured_count} photos saved in '{folder_path}'")
    return {"status": "success", "name": name, "photos_captured": captured_count, "folder_path": folder_path}

def add_employee_to_db(employee_id, name, email):
    try:
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO employees (employee_id, name, email) VALUES (?, ?, ?)",
                  (employee_id, name, email))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Employee {name} added/updated"}
    except sqlite3.Error as e:
        raise Exception(f"Database error: {e}")

if __name__ == "__main__":
    recognize_live_task()