# utils/model.py

import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from config import MODEL_PATH
from utils.dataset import load_dataset

def train_model():
    status_img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(status_img, "Loading dataset...", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Training Status", status_img)
    cv2.waitKey(1)

    X, y = load_dataset()

    if len(X) == 0:
        cv2.putText(status_img, "Error: No training data found!", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Training Status", status_img)
        cv2.waitKey(3000)
        cv2.destroyWindow("Training Status")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale', probability=True))
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    cv2.putText(status_img, f"Accuracy: {accuracy:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(status_img, "Model saved!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Training Status", status_img)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    cv2.waitKey(3000)
    cv2.destroyWindow("Training Status")
    return model
