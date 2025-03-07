# backend.py
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import threading
import cv2
from pydantic import BaseModel
from display import RTSP_STREAMS, CameraDashboard
from face import store_embeddings, recognize_live_task, recognition_running, cap  # Adjusted to 'face_recognition_module'
import tkinter as tk

app = FastAPI()

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
        cap = None
    return {"status": "Recognition stopped"}

@app.post("/capture")
async def capture_endpoint(request: CaptureRequest):
    from face import capture_images
    try:
        result = capture_images(request.name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_employee")
async def add_employee(request: EmployeeRequest):
    from face_recognition_module import add_employee_to_db
    try:
        result = add_employee_to_db(request.employee_id, request.name, request.email)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    root = tk.Tk()
    dashboard = CameraDashboard(root)
    root.mainloop()