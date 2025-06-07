# utils/dataset.py

import os
import cv2
import numpy as np
from config import DATASET_DIR, IMAGE_EXTENSIONS
from utils.preprocessing import extract_landmarks

def load_dataset():
    X, y = []
    y = []
    for class_name in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for image_file in os.listdir(class_dir):
            if image_file.lower().endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(class_dir, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    landmarks = extract_landmarks(image)
                    if landmarks is not None:
                        X.append(landmarks)
                        y.append(class_name)
    return np.array(X), np.array(y)
