# display.py
import tkinter as tk
from tkinter import Label, Button, messagebox
from tkinter import simpledialog  # Import simpledialog explicitly
import threading
import cv2
from PIL import Image, ImageTk
import requests

# Define the RTSP stream sources (for reference, not auto-started)
RTSP_STREAMS = [
    0,  # Laptop webcam
    'http://192.168.0.174:4747/video',  # Phone camera RTSP URL
]

class CameraDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Dashboard")
        self.root.geometry("800x600")
        self.root.configure(bg="white")  # Blank white background

        self.labels = []
        self.show_second_camera_flag = False
        self.cap_first_camera = None
        self.cap_second_camera = None
        self.threads = []

        # Create blank white placeholders for potential video feeds
        for i in range(1):
            for j in range(2):
                label = Label(root, bg="white", width=40, height=20)  # White, no video by default
                label.grid(row=i, column=j, padx=10, pady=10)
                self.labels.append(label)

        # Buttons frame
        button_frame = tk.Frame(root, bg="white")
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)

        # Define buttons
        self.camera_button = Button(button_frame, text="Camera", command=self.start_first_camera, bg="lightgray")
        self.camera_button.grid(row=0, column=0, padx=5)

        self.train_button = Button(button_frame, text="Train", command=self.train_model, bg="lightgray")
        self.train_button.grid(row=0, column=1, padx=5)

        self.recognize_start_button = Button(button_frame, text="Recognize Start", command=self.start_recognition, bg="lightgray")
        self.recognize_start_button.grid(row=0, column=2, padx=5)

        self.recognize_stop_button = Button(button_frame, text="Recognize Stop", command=self.stop_recognition, bg="lightgray")
        self.recognize_stop_button.grid(row=0, column=3, padx=5)

        self.capture_button = Button(button_frame, text="Capture", command=self.capture_images, bg="lightgray")
        self.capture_button.grid(row=0, column=4, padx=5)

        self.add_employee_button = Button(button_frame, text="Add Employee", command=self.add_employee, bg="lightgray")
        self.add_employee_button.grid(row=0, column=5, padx=5)

        self.toggle_second_camera_button = Button(button_frame, text="Show Second Camera", command=self.toggle_second_camera, bg="lightgray")
        self.toggle_second_camera_button.grid(row=0, column=6, padx=5)

    def start_first_camera(self):
        """Start the first camera feed."""
        if self.cap_first_camera is None:
            self.cap_first_camera = cv2.VideoCapture(RTSP_STREAMS[0])
            if not self.cap_first_camera.isOpened():
                messagebox.showerror("Error", "First camera not accessible.")
                self.cap_first_camera = None
                return
            thread = threading.Thread(target=self.update_frame, args=(0, self.cap_first_camera))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def toggle_second_camera(self):
        """Toggle the second camera feed on/off."""
        self.show_second_camera_flag = not self.show_second_camera_flag
        if self.show_second_camera_flag:
            self.toggle_second_camera_button.config(text="Hide Second Camera")
            self.start_second_camera()
        else:
            self.toggle_second_camera_button.config(text="Show Second Camera")
            self.stop_second_camera()

    def start_second_camera(self):
        """Start the second camera feed."""
        if self.cap_second_camera is None:
            self.cap_second_camera = cv2.VideoCapture(RTSP_STREAMS[1])
            if not self.cap_second_camera.isOpened():
                messagebox.showerror("Error", "Second camera not accessible.")
                self.cap_second_camera = None
                return
            thread = threading.Thread(target=self.update_frame, args=(1, self.cap_second_camera))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def stop_second_camera(self):
        """Stop the second camera feed."""
        if self.cap_second_camera is not None:
            self.cap_second_camera.release()
            self.cap_second_camera = None
            self.labels[1].config(image=None)
            self.labels[1].image = None

    def update_frame(self, index, cap):
        """Update the video frame for the given camera."""
        def video_loop():
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (400,400))
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                if index == 1 and not self.show_second_camera_flag:
                    continue
                self.labels[index].after(10, self.update_image, index, img)
            cap.release()

        threading.Thread(target=video_loop, daemon=True).start()

    def update_image(self, index, img):
        """Update the label with the new frame."""
        self.labels[index].config(image=img)
        self.labels[index].image = img

    def train_model(self):
        """Trigger the training process via API."""
        folder_path = simpledialog.askstring("Input", "Enter folder path for training images:")
        if folder_path:
            try:
                response = requests.post("http://localhost:8000/train", json={"folder_path": folder_path})
                if response.status_code == 200:
                    messagebox.showinfo("Success", "Training completed: " + response.text)
                else:
                    messagebox.showerror("Error", f"Training failed: {response.text}")
            except requests.RequestException as e:
                messagebox.showerror("Error", f"API call failed: {e}")

    def start_recognition(self):
        """Start the recognition process via API."""
        try:
            response = requests.post("http://localhost:8000/recognize/start")
            if response.status_code == 200:
                messagebox.showinfo("Success", "Recognition started.")
            else:
                messagebox.showerror("Error", f"Start recognition failed: {response.text}")
        except requests.RequestException as e:
            messagebox.showerror("Error", f"API call failed: {e}")

    def stop_recognition(self):
        """Stop the recognition process via API."""
        try:
            response = requests.post("http://localhost:8000/recognize/stop")
            if response.status_code == 200:
                messagebox.showinfo("Success", "Recognition stopped.")
            else:
                messagebox.showerror("Error", f"Stop recognition failed: {response.text}")
        except requests.RequestException as e:
            messagebox.showerror("Error", f"API call failed: {e}")

    def capture_images(self):
        """Trigger image capture via API."""
        name = simpledialog.askstring("Input", "Enter name for capturing images:")
        if name:
            try:
                response = requests.post("http://localhost:8000/capture", json={"name": name})
                if response.status_code == 200:
                    messagebox.showinfo("Success", "Images captured: " + response.text)
                else:
                    messagebox.showerror("Error", f"Capture failed: {response.text}")
            except requests.RequestException as e:
                messagebox.showerror("Error", f"API call failed: {e}")

    def add_employee(self):
        """Add an employee via API."""
        employee_id = simpledialog.askstring("Input", "Enter employee ID:")
        name = simpledialog.askstring("Input", "Enter employee name:")
        email = simpledialog.askstring("Input", "Enter employee email:")
        if employee_id and name and email:
            try:
                response = requests.post("http://localhost:8000/add_employee", 
                                        json={"employee_id": employee_id, "name": name, "email": email})
                if response.status_code == 200:
                    messagebox.showinfo("Success", "Employee added: " + response.text)
                else:
                    messagebox.showerror("Error", f"Add employee failed: {response.text}")
            except requests.RequestException as e:
                messagebox.showerror("Error", f"API call failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    dashboard = CameraDashboard(root)
    root.mainloop()