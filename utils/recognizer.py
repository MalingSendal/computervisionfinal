# utils/recognizer.py

import os
import time
import pickle
import numpy as np
import cv2
from config import MODEL_PATH
from utils.preprocessing import extract_landmarks
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def recognize_sibi_sign():
    if not os.path.exists(MODEL_PATH):
        error_img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(error_img, "No trained model found!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Error", error_img)
        cv2.waitKey(3000)
        cv2.destroyWindow("Error")
        return

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    sentence = ""
    last_sign = ""
    last_time = time.time()
    last_added_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        landmarks = extract_landmarks(frame)
        current_time = time.time()
        confidence = 0.0

        if landmarks is not None:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            prediction = model.predict([landmarks])[0]
            confidence = np.max(model.predict_proba([landmarks]))

            if confidence > 0.7:
                if prediction != last_sign or (current_time - last_added_time) > 1.0:
                    sentence += prediction
                    last_sign = prediction
                    last_added_time = current_time
                cv2.putText(frame, f"SIBI Sign: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif confidence > 0.3:
                cv2.putText(frame, "Unrecognized sign", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                last_sign = ""
            else:
                cv2.putText(frame, "Not a sign", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                last_sign = ""
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            last_sign = ""

        if current_time - last_time > 5 and (len(sentence) == 0 or not sentence.endswith(" ")):
            sentence += " "
            last_sign = ""
            last_added_time = current_time
        if current_time - last_time > 15 and len(sentence.strip()) > 0:
            result_img = np.zeros((200, 800, 3), dtype=np.uint8)
            cv2.putText(result_img, "Recognized Sentence:", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(result_img, sentence.strip(), (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Result", result_img)
            cv2.waitKey(5000)
            cv2.destroyWindow("Result")
            sentence = ""
            last_sign = ""
            last_time = current_time

        cv2.putText(frame, f"Sentence: {sentence}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow('SIBI Sign Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
