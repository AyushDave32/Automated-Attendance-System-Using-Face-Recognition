# api_server.py
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import threading
import cv2
from pydantic import BaseModel
from dashboard import RTSP_STREAMS, CameraDashboard
from face_recognition_module import store_embeddings, recognize_live, recognition_running, cap, log_to_db
import tkinter as tk

app = FastAPI()

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

class TrainRequest(BaseModel):
    folder_path: str

@app.post("/train")
async def train_endpoint(request: TrainRequest):
    store_embeddings(request.folder_path)
    return {"status": "success", "total_faces": faiss_index.ntotal}

@app.post("/recognize/start")
async def start_recognition(background_tasks: BackgroundTasks):
    global recognition_running
    if recognition_running:
        raise HTTPException(status_code=400, detail="Recognition is already running.")
    background_tasks.add_task(recognize_live)
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

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    root = tk.Tk()
    dashboard = CameraDashboard(root)
    root.mainloop()