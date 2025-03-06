# dashboard.py
import cv2
import threading
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Define the RTSP stream sources
RTSP_STREAMS = [
    0,  # Laptop webcam
    'http://192.168.0.174:4747/video',  # Phone camera RTSP URL
]

class CameraDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RTSP Camera Dashboard")
        self.root.geometry("800x600")

        self.labels = []
        self.show_second_camera = False
        self.cap_second_camera = None

        for i in range(1):
            for j in range(2):
                label = Label(root, bg="black")
                label.grid(row=i, column=j, padx=10, pady=10)
                self.labels.append(label)

        self.toggle_button = Button(root, text="Show Second Camera", command=self.toggle_second_camera)
        self.toggle_button.grid(row=2, column=0, columnspan=2)

        self.threads = []
        for i, source in enumerate(RTSP_STREAMS):
            thread = threading.Thread(target=self.update_frame, args=(i, source))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def toggle_second_camera(self):
        self.show_second_camera = not self.show_second_camera
        if self.show_second_camera:
            self.toggle_button.config(text="Hide Second Camera")
            self.start_second_camera()
        else:
            self.toggle_button.config(text="Show Second Camera")
            self.stop_second_camera()

    def start_second_camera(self):
        if self.cap_second_camera is None:
            self.cap_second_camera = cv2.VideoCapture(RTSP_STREAMS[1])
            if not self.cap_second_camera.isOpened():
                print("Second camera not accessible.")
                return
            thread = threading.Thread(target=self.update_frame, args=(1, RTSP_STREAMS[1]))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def stop_second_camera(self):
        if self.cap_second_camera is not None:
            self.cap_second_camera.release()
            self.cap_second_camera = None
            self.labels[1].config(image=None)
            self.labels[1].image = None

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
                if index == 1 and not self.show_second_camera:
                    continue
                self.labels[index].after(10, self.update_image, index, img)

        threading.Thread(target=video_loop, daemon=True).start()

    def update_image(self, index, img):
        self.labels[index].config(image=img)
        self.labels[index].image = img

if __name__ == "__main__":
    root = tk.Tk()
    dashboard = CameraDashboard(root)
    root.mainloop()