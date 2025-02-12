import os
import random
import shutil

# Define paths
SAVE_DIR = "wider_yolo_dataset"
IMAGE_DIR = os.path.join(SAVE_DIR, "images/train")
LABEL_DIR = os.path.join(SAVE_DIR, "labels/train")

VAL_IMAGE_DIR = os.path.join(SAVE_DIR, "images/val")
VAL_LABEL_DIR = os.path.join(SAVE_DIR, "labels/val")

# Create validation directories
os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

# List all images
all_images = os.listdir(IMAGE_DIR)
random.shuffle(all_images)  # Shuffle to randomize selection

# Define split ratio (80% train, 20% val)
split_ratio = 0.8
split_index = int(len(all_images) * split_ratio)

train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Move validation images and labels
for img in val_images:
    img_path = os.path.join(IMAGE_DIR, img)
    label_path = os.path.join(LABEL_DIR, img.replace(".jpg", ".txt"))  # YOLO format label

    # Move files to validation set
    shutil.move(img_path, os.path.join(VAL_IMAGE_DIR, img))
    shutil.move(label_path, os.path.join(VAL_LABEL_DIR, img.replace(".jpg", ".txt")))

print("✅ Dataset successfully split into train and validation sets!")
