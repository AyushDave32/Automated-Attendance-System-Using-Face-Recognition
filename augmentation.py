import cv2
import numpy as np
import os
import albumentations as A
from glob import glob

# Define augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5)
])

# Path to original images
input_folder = "images/Ayush/"
output_folder = "augmented_images/"
os.makedirs(output_folder, exist_ok=True)

# Process each image
image_paths = glob(os.path.join(input_folder, "*.jpg"))  # Change extension if needed

for img_path in image_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(5):  # Generate 5 augmented versions per image
        augmented = augmentations(image=img)["image"]
        output_path = os.path.join(output_folder, f"{os.path.basename(img_path).split('.')[0]}_aug{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

print(f"Augmented images saved in {output_folder}")
