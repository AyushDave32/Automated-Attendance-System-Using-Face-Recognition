import cv2
import threading
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# List of camera sources (replace with your phone's RTSP URL and laptop camera)
CAM_SOURCES = [0, 'rtsp://10.147.76.251:8080/video']  # Modify this with your phone camera URL

class CameraDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RTSP Camera Dashboard")
        self.root.geometry("800x600")  # Set window size

        self.labels = []  # Store Tkinter labels for displaying video feeds

        # Create 1x2 grid layout for two video feeds (one for laptop and one for phone)
        for i in range(1):
            for j in range(2):
                label = Label(root, bg="black")  # Black background for missing feed
                label.grid(row=i, column=j, padx=10, pady=10)
                self.labels.append(label)

        # Start video threads for both laptop and phone cameras
        self.threads = []
        for i, source in enumerate(CAM_SOURCES):
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


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraDashboard(root)
    root.mainloop()
