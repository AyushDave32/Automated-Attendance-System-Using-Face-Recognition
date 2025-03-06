import cv2
import threading
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
import faiss
import os
import dlib
import pickle
import sqlite3
from deepface import DeepFace
import onnxruntime as ort
from datetime import datetime, timedelta
import asyncio
import time
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the RTSP stream sources
RTSP_STREAMS = [
    0,  # Laptop webcam
    'http://192.168.0.174:4747/video',  # Phone camera RTSP URL
]

app = FastAPI()

# Tkinter Dashboard as Home Page
class CameraDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RTSP Camera Dashboard - Home")
        self.root.geometry("800x600")

        self.labels = []
        for i in range(1):
            for j in range(2):
                label = Label(root, bg="black")
                label.grid(row=i, column=j, padx=10, pady=10)
                self.labels.append(label)

        self.threads = []
        for i, source in enumerate(RTSP_STREAMS):
            thread = threading.Thread(target=self.update_frame, args=(i, source))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def update_frame(self, index, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Camera {index} not accessible.")
            return

        def video_loop():
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (320, 240))
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.labels[index].after(10, self.update_image, index, img)

        threading.Thread(target=video_loop, daemon=True).start()

    def update_image(self, index, img):
        self.labels[index].config(image=img)
        self.labels[index].image = img

# FastAPI video streaming endpoints
def generate_frames(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/camera/{cam_id}")
def video_feed(cam_id: int):
    if cam_id < 0 or cam_id >= len(RTSP_STREAMS):
        return {"error": "Invalid camera ID"}
    return StreamingResponse(generate_frames(RTSP_STREAMS[cam_id]), media_type="multipart/x-mixed-replace; boundary=frame")

# Face Recognition Setup
onnx_path = "yolov11n-face.onnx"
ort_session = ort.InferenceSession(onnx_path)

embedding_dim = 512
faiss_index = faiss.IndexFlatL2(embedding_dim)
embedding_db = {}

if os.path.exists("face_index.pkl"):
    with open("face_index.pkl", "rb") as f:
        embedding_db, faiss_index = pickle.load(f)

print(f"Total faces in database: {faiss_index.ntotal}")

dlib_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(predictor_path)

detection_tracker = {}
logged_names = {}
cap = None
recognition_running = False

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

# Utility Functions without PyTorch
def preprocess_onnx(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def non_max_suppression_np(boxes, scores, iou_threshold=0.5):
    """NumPy-based NMS implementation"""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def postprocess_onnx(outputs, orig_h, orig_w):
    """Process ONNX output without PyTorch"""
    outputs = outputs[0]  # Assuming single output from ONNX
    boxes = outputs[:, :4]  # x1, y1, x2, y2
    scores = outputs[:, 4]  # Confidence scores
    classes = outputs[:, 5]  # Class IDs

    # Filter by confidence
    mask = scores > 0.5
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    if len(boxes) == 0:
        return []

    # Apply NMS
    keep = non_max_suppression_np(boxes, scores, iou_threshold=0.5)

    detections = []
    scale_h, scale_w = orig_h / 640, orig_w / 640
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx]
        conf = scores[idx]
        cls = classes[idx]
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

def is_face_straight(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    landmarks = landmark_predictor(gray, dlib.rectangle(0, 0, gray.shape[1], gray.shape[0]))
    nose = (landmarks.part(30).x, landmarks.part(30).y)
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    left_distance = abs(nose[0] - left_eye[0])
    right_distance = abs(nose[0] - right_eye[0])
    if left_distance / right_distance > 2 or right_distance / left_distance > 2:
        print("Skip the image due to looking sideways -------------------------")
        return False
    return True

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
        raise HTTPException(status_code=400, detail=f"Folder '{image_folder}' not found.")
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
                            if email:
                                send_email(email, name, current_time)
                            detection_tracker[name] = {"count": 0, "times": []}
    cap.release()
    recognition_running = False

def capture_images(name: str):
    folder_path = f"Database/{name}"
    os.makedirs(folder_path, exist_ok=True)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error: Could not access the webcam.")
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

class TrainRequest(BaseModel):
    folder_path: str

class CaptureRequest(BaseModel):
    name: str

class EmployeeRequest(BaseModel):
    employee_id: str
    name: str
    email: str

@app.post("/train")
async def train_endpoint(request: TrainRequest):
    result = await store_embeddings(request.folder_path)
    return result

@app.post("/recognize/start")
async def start_recognition(background_tasks: BackgroundTasks):
    global recognition_running
    if recognition_running:
        raise HTTPException(status_code=400, detail="Recognition is already running.")
    background_tasks.add_task(recognize_live_task)
    return {"status": "Recognition started"}

@app.post("/recognize/stop")
async def stop_recognition():
    global recognition_running, cap
    if not recognition_running:
        raise HTTPException(status_code=400, detail="Recognition is not running.")
    recognition_running = False
    if cap is not None:
        cap.release()
    return {"status": "Recognition stopped"}

@app.post("/capture")
async def capture_endpoint(request: CaptureRequest):
    try:
        result = capture_images(request.name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_employee")
async def add_employee(request: EmployeeRequest):
    try:
        conn = sqlite3.connect("attendance.db")
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO employees (employee_id, name, email) VALUES (?, ?, ?)",
                  (request.employee_id, request.name, request.email))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Employee {request.name} added/updated"}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    root = tk.Tk()
    dashboard = CameraDashboard(root)
    root.mainloop()