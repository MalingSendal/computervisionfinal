# main.py

import os
import cv2
import numpy as np
from config import DATASET_DIR
from utils.model import train_model
from utils.recognizer import recognize_sibi_sign

def show_menu():
    menu_img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(menu_img, "SIBI Hand Sign Recognition System", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(menu_img, "1. Train model on existing dataset", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(menu_img, "2. Start real-time SIBI recognition", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(menu_img, "3. Exit", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(menu_img, "Press 1, 2 or 3 to select option", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(menu_img, "Press ESC to close window", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("SIBI Recognition Menu", menu_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            cv2.destroyWindow("SIBI Recognition Menu")
            train_model()
            return 1
        elif key == ord('2'):
            cv2.destroyWindow("SIBI Recognition Menu")
            recognize_sibi_sign()
            return 2
        elif key == ord('3') or key == 27:
            cv2.destroyWindow("SIBI Recognition Menu")
            return 3

def main():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        info_img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(info_img, f"Created {DATASET_DIR} folder", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_img, "Add SIBI images in a/, b/, c/ folders", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Information", info_img)
        cv2.waitKey(5000)
        cv2.destroyWindow("Information")

    while True:
        choice = show_menu()
        if choice == 3:
            break

if __name__ == "__main__":
    main()
