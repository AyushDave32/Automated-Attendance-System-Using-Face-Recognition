import cv2
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Define the RTSP stream sources
RTSP_STREAMS = [
    0,  # Laptop webcam (Replace later with "rtsp://your_camera_1_url")
    'http://192.168.0.174:4747/video',  # Example: Phone camera RTSP URL
]

app = FastAPI()

# Tkinter application for displaying the camera feed
class CameraDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RTSP Camera Dashboard")
        self.root.geometry("800x600")  # Set window size

        self.labels = []  # Store Tkinter labels for displaying video feeds

        # Create a grid layout for displaying video feeds
        for i in range(1):
            for j in range(2):
                label = Label(root, bg="black")  # Black background for missing feed
                label.grid(row=i, column=j, padx=10, pady=10)
                self.labels.append(label)

        # Start video threads for both laptop and phone cameras
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
                frame = cv2.resize(frame, (320, 240))  # Resize to fit the grid
                img = ImageTk.PhotoImage(Image.fromarray(frame))

                # Ensure updates happen in the Tkinter thread
                self.labels[index].after(10, self.update_image, index, img)

        threading.Thread(target=video_loop, daemon=True).start()

    def update_image(self, index, img):
        self.labels[index].config(image=img)
        self.labels[index].image = img

# FastAPI backend to stream video
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

@app.get("/camera/{cam_id}")
def video_feed(cam_id: int):
    if cam_id < 0 or cam_id >= len(RTSP_STREAMS):
        return {"error": "Invalid camera ID"}

    return StreamingResponse(generate_frames(RTSP_STREAMS[cam_id]), media_type="multipart/x-mixed-replace; boundary=frame")

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Run FastAPI backend in a background thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    # Initialize Tkinter GUI for camera dashboard
    root = tk.Tk()
    app = CameraDashboard(root)
    root.mainloop()
