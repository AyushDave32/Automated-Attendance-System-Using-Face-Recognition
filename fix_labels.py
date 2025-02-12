import os

def fix_labels(label_dir):
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, "r") as f:
                lines = f.readlines()
            fixed_lines = []
            for line in lines:
                values = list(map(float, line.strip().split()))
                if all(0 <= v <= 1 for v in values[1:]):  # Ignore class index
                    fixed_lines.append(line)
            if fixed_lines:
                with open(file_path, "w") as f:
                    f.writelines(fixed_lines)
            else:
                print(f"Removed corrupt label file: {file_path}")
                os.remove(file_path)

fix_labels("C:/Users/Ayush/Downloads/face_recog_attendance/wider_yolo_dataset/labels/val/")
