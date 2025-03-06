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
