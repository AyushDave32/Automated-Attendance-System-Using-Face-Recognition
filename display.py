import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading

# RTSP Stream URLs (Replace with your camera URLs)
RTSP_URLS = [
    "rtsp://your_camera_1_url",
    "rtsp://your_camera_2_url",
    "rtsp://your_camera_3_url",
    "rtsp://your_camera_4_url"
]

class RTSPDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RTSP Camera Dashboard")
        
        # Create Labels for 4 camera feeds
        self.labels = []
        for i in range(4):
            label = tk.Label(self.root, text=f"Camera {i+1}", bg="black", fg="white")
            label.grid(row=i//2, column=i%2, padx=10, pady=10)
            self.labels.append(label)
        
        # Start threads for each RTSP stream
        self.streams = []
        for i in range(4):
            thread = threading.Thread(target=self.update_frame, args=(i,))
            thread.daemon = True
            thread.start()
            self.streams.append(thread)
    
    def update_frame(self, index):
        cap = cv2.VideoCapture(RTSP_URLS[index])
        if not cap.isOpened():
            print(f"Error: Cannot open RTSP stream {index+1}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                continue  # Skip if frame not captured

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((400, 300))  # Resize for display
            img_tk = ImageTk.PhotoImage(img)

            # Update label with new frame
            self.labels[index].config(image=img_tk)
            self.labels[index].image = img_tk  # Keep reference to avoid garbage collection

        cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = RTSPDashboard(root)
    root.mainloop()
