import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

RTSP_URL = "rtsp://camera.rtsp.stream:1935/live/stream"

def generate_frames():
    cap = cv2.VideoCapture(RTSP_URL)  # Open stream inside the function
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break  # Stop if no more frames

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame in HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()  # Release video capture when done

@app.get("/rtsp-stream")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




# import os
# import cv2
# import faiss
# import pickle
# import sqlite3
# from datetime import datetime
# from fastapi import FastAPI, BackgroundTasks, HTTPException
# import onnxruntime as ort
# from rtsp_stream import RTSPStream
# from attendance_db import log_to_db, load_embeddings, store_embeddings

# # FastAPI app
# app = FastAPI()

# # Load YOLOv11 ONNX model
# onnx_path = "yolov11n-face.onnx"
# ort_session = ort.InferenceSession(onnx_path)

# # FAISS setup
# embedding_dim = 512  # Matches FaceNet512 output
# embedding_db, faiss_index = load_embeddings()

# # RTSP Stream setup
# rtsp_stream = RTSPStream("rtsp://your_rtsp_stream_url")

# # Global variables
# recognition_running = False  # Flag to control live recognition
# logged_names = {}

# # Utility functions for face detection and recognition
# def preprocess_onnx(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (640, 640))
#     img = img / 255.0
#     img = img.transpose(2, 0, 1)
#     img = np.expand_dims(img, axis=0).astype(np.float32)
#     return img

# def postprocess_onnx(outputs, orig_h, orig_w):
#     # (same as previous implementation)
#     pass

# def extract_embedding(face):
#     # (same as previous implementation)
#     pass

# # API to start live recognition in the background
# @app.post("/recognize_live/")
# async def recognize_live(background_tasks: BackgroundTasks):
#     global recognition_running
#     if recognition_running:
#         return {"status": "already running"}
    
#     background_tasks.add_task(recognize_live_task)
#     return {"status": "started"}

# # Recognition task
# def recognize_live_task():
#     global recognition_running, rtsp_stream, embedding_db, faiss_index
#     recognition_running = True
#     if not rtsp_stream.start():
#         print("Error starting RTSP stream.")
#         return

#     try:
#         while recognition_running:
#             frame = rtsp_stream.read_frame()
#             if frame is None:
#                 break

#             img_h, img_w, _ = frame.shape
#             img_input = preprocess_onnx(frame)
#             ort_inputs = {ort_session.get_inputs()[0].name: img_input}
#             ort_outputs = ort_session.run(None, ort_inputs)
#             detections = postprocess_onnx(ort_outputs[0], img_h, img_w)

#             for x1, y1, x2, y2, conf, class_id in detections:
#                 if conf > 0.7 and class_id == 0:
#                     face = frame[y1:y2, x1:x2]
#                     embedding = extract_embedding(face)
#                     if embedding is not None:
#                         distances, indices = faiss_index.search(embedding, 1)
#                         if distances[0][0] < 0.5:
#                             name = list(embedding_db.keys())[indices[0][0]]
#                             current_date = datetime.now().strftime("%Y-%m-%d")
#                             if name not in logged_names.get(current_date, set()):
#                                 log_to_db(name, datetime.now())
#                                 if current_date not in logged_names:
#                                     logged_names[current_date] = set()
#                                 logged_names[current_date].add(name)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         rtsp_stream.stop()
#         recognition_running = False

# # API to store images into the database
# @app.post("/store_embeddings/")
# async def store_embeddings_endpoint(image_folder: str):
#     global embedding_db, faiss_index
#     for filename in os.listdir(image_folder):
#         img_path = os.path.join(image_folder, filename)
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Could not read {img_path}, skipping.")
#             continue
#         # Face detection and embedding extraction
#         embedding = extract_embedding(img)
#         if embedding is not None:
#             name = os.path.splitext(filename)[0]
#             faiss_index.add(embedding)
#             embedding_db[name] = embedding
#             print(f"Stored: {name}")
#     store_embeddings(embedding_db, faiss_index)
#     return {"status": "success", "total_faces": faiss_index.ntotal}
