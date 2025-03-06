import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

# Temporary setup: Use webcam instead of RTSP
RTSP_STREAMS = [
    0,  # Laptop webcam (Replace later with "rtsp://your_camera_1_url")
    # "rtsp://your_camera_2_url",
    # "rtsp://your_camera_3_url",
    # "rtsp://your_camera_4_url",
]

def generate_frames(source):
    cap = cv2.VideoCapture(source)  # Handle both webcam (0) and RTSP URLs

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/camera/{cam_id}")
def video_feed(cam_id: int):
    if cam_id < 0 or cam_id >= len(RTSP_STREAMS):
        return {"error": "Invalid camera ID"}

    return StreamingResponse(generate_frames(RTSP_STREAMS[cam_id]), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
