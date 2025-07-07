# SIBI Hand Sign Recognition System

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMp1C7pAAvzoAxFwkgn7mzE8k-JSBW5qJITA&s" alt="Logo SIBI" width="200"/>
</p>

## 📖 Project Description

This project aims to develop a system that can detect and translate Indonesian Sign Language (SIBI) using Computer Vision. The system is designed to assist communication for people with disabilities who use sign language and to support learning SIBI.

## ✨ Features

- **Hand Detection**: Detects hands and hand gestures using computer webcam.
- **Sign Classification**: Classifies SIBI hand signs into corresponding letters (A-Z, 0-9).
- **Real-time Recognition**: Recognizes and displays the detected sign in real time.
- **Model Training**: Train your own model using your dataset of hand sign images.
- **User Interface**: Simple graphical menu for training and recognition.

## 🗂️ Project Structure

```
asl_datasets/                        # Folders for each sign class containing cropped hand images
  a/,
  b/,
  ...,
  z/,
  0/,
  ...,
  9/
vision.py                        # Main application code
sibi_model.pkl                   # Trained model (generated after training)
README.md                        # Project documentation
```

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MalingSendal/computervisionfinal.git
   cd computervisionfinal
   ```

2. **Install Dependencies**
   ```bash
   pip install opencv-python mediapipe==0.10.5 scikit-learn numpy
   ```

3. **Prepare Dataset**
   - Place cropped hand sign images in the appropriate subfolders under `asl_datasets/` (e.g., `asl_datasets/a/`, `asl_datasets/b/`, ..., `asl_datasets/0/`, etc.).
   - Each folder should contain images for that specific sign.

## 🏃‍♂️ How to Run

1. **Start the Application**
   ```bash
   python3 vision.py
   ```

2. **Menu Options**
   - **1**: Train the model using the images in `asl_datasets/`.
   - **2**: Start real-time SIBI sign recognition using your webcam.
   - **3** or **ESC**: Exit the application.

## 🛠️ How It Works

- Uses [MediaPipe](https://google.github.io/mediapipe/) to extract hand landmarks from images.
- Trains an SVM classifier (with scikit-learn) on the extracted landmarks.
- Recognizes hand signs in real time and displays the predicted letter/number and confidence score.

## 📊 Project Progress

| Phase | Description                              | Status           | Progress   |
|-------|------------------------------------------|------------------|------------|
| 1     | Dataset Collection & Preprocessing       | ✅ Completed      | 100%       |
| 2     | Hand Sign Detection Model Development    | 🟡 In Progress   | 70%        |
| 3     | Text Translation Implementation          | 🟡 In Progress   | 100%         |
| 4     | Testing & Validation                     | ⏳ Not Started   | 100%         |

### Dataset Size

- **2515** hand sign images across all classes.

## 📷 Example

![Example](https://user-images.githubusercontent.com/yourusername/example-sibi-demo.gif)

## 🤝 Contributing

Contributions are more than welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.

---

**Developed by Rendy for better SIBI accessibility.**
