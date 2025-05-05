#dependecies needed:
# pip install opencv-python mediapipe scikit-learn numpy

import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, 
                      min_detection_confidence=0.5)

# Configuration
DATASET_DIR = 'datasets'
MODEL_PATH = 'asl_model.pkl'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def extract_landmarks(image):
    """Extract hand landmarks from an image"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
    return None

def load_dataset():
    """Load dataset from image folders"""
    X = []
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

def train_model():
    """Train and save the ASL recognition model"""
    # Create a black image for status display
    status_img = np.zeros((300, 600, 3), dtype=np.uint8)
    
    cv2.putText(status_img, "Loading dataset...", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Training Status", status_img)
    cv2.waitKey(1)
    
    X, y = load_dataset()
    
    if len(X) == 0:
        cv2.putText(status_img, "Error: No training data found!", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(status_img, f"Check {DATASET_DIR}/a/, {DATASET_DIR}/b/, etc.", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Training Status", status_img)
        cv2.waitKey(3000)
        cv2.destroyWindow("Training Status")
        return None
    
    cv2.putText(status_img, f"Loaded {len(X)} samples", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(status_img, f"Training on {len(set(y))} classes", (50, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Training Status", status_img)
    cv2.waitKey(1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    cv2.putText(status_img, f"Accuracy: {accuracy:.2f}", (50, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(status_img, "Model saved!", (50, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Training Status", status_img)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    cv2.waitKey(3000)
    cv2.destroyWindow("Training Status")
    return model

def recognize_asl_sign():
    """Real-time ASL sign recognition from webcam"""
    if not os.path.exists(MODEL_PATH):
        error_img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(error_img, "No trained model found!", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(error_img, "Please train the model first", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Error", error_img)
        cv2.waitKey(3000)
        cv2.destroyWindow("Error")
        return
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        landmarks = extract_landmarks(frame)
        
        if landmarks is not None:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            prediction = model.predict([landmarks])[0]
            confidence = np.max(model.predict_proba([landmarks]))
            
            cv2.putText(frame, f"ASL Sign: {prediction}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('ASL Sign Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def show_menu():
    """Display graphical menu for the application"""
    menu_img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw menu title
    cv2.putText(menu_img, "ASL Hand Sign Recognition System", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # Draw menu options
    cv2.putText(menu_img, "1. Train model on existing dataset", (50, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(menu_img, "2. Start real-time ASL recognition", (50, 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(menu_img, "3. Exit", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw instructions
    cv2.putText(menu_img, "Press 1, 2 or 3 to select option", (50, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(menu_img, "Press ESC to close window", (50, 350), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    cv2.imshow("ASL Recognition Menu", menu_img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            cv2.destroyWindow("ASL Recognition Menu")
            train_model()
            return 1
        elif key == ord('2'):
            cv2.destroyWindow("ASL Recognition Menu")
            recognize_asl_sign()
            return 2
        elif key == ord('3') or key == 27:  # 27 is ESC key
            cv2.destroyWindow("ASL Recognition Menu")
            return 3

def main():
    """Main program loop"""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        # Show popup message about dataset folder
        info_img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(info_img, f"Created {DATASET_DIR} folder", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_img, "Add ASL images in a/, b/, c/ folders", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Information", info_img)
        cv2.waitKey(3000)
        cv2.destroyWindow("Information")
    
    while True:
        choice = show_menu()
        if choice == 3:
            break

if __name__ == "__main__":
    main()