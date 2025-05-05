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

# initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, 
                      min_detection_confidence=0.5)

# config
DATASET_DIR = 'datasets'  # folder a/, b/, c/, etc.
MODEL_PATH = 'asl_model.pkl'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def extract_landmarks(image):
    """Extract hand landmarks from an image"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # extract landmarks (x,y,z coordinates)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
    return None

def load_dataset():
    """Load dataset from image folders"""
    X = []
    y = []
    
    # each subfolder represents a class (a, b, c, etc.)
    for class_name in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # process each image in the class folder
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
    print("Loading dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("Error: No valid training data found!")
        print(f"Please ensure your dataset is structured as: {DATASET_DIR}/a/, {DATASET_DIR}/b/, etc.")
        return None
    
    print(f"Loaded {len(X)} samples for {len(set(y))} classes")
    print("Classes found:", sorted(set(y)))
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # create and train model pipeline
    print("Training model...")
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    )
    model.fit(X_train, y_train)
    
    # evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.2f}")
    
    # save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {MODEL_PATH}")
    return model

def recognize_asl_sign():
    """Real-time ASL sign recognition from webcam"""
    # load trained model
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Please train the model first.")
        return
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Get class labels from dataset folders
    class_labels = sorted([d for d in os.listdir(DATASET_DIR) 
                         if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get landmarks
        landmarks = extract_landmarks(frame)
        
        # Draw hand landmarks if detected
        if landmarks is not None:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Predict ASL sign
            prediction = model.predict([landmarks])[0]
            confidence = np.max(model.predict_proba([landmarks]))
            
            # Display prediction
            cv2.putText(frame, f"ASL Sign: {prediction}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('ASL Sign Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main program interface"""
    print("ASL Hand Sign Recognition System")
    print("1. Train model on existing dataset")
    print("2. Start real-time ASL recognition")
    print("3. Exit")
    
    while True:
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            recognize_asl_sign()
        elif choice == '3':
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    # Create datasets folder if it doesn't exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Created {DATASET_DIR} folder. Please add your ASL image datasets in folders a/, b/, c/, etc.")
    
    main()