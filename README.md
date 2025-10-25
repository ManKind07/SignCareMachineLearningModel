# 🧠 Static Sign Language Detection with MediaPipe and Virtual Webcam

This project captures **static (non-moving) sign language gestures** from a webcam, classifies them in real-time, and streams the annotated video feed to a **virtual webcam** for use in applications like **Zoom, Google Meet, or Microsoft Teams**.

It uses **Google's MediaPipe** for high-fidelity hand and landmark tracking, and a **Random Forest classifier** for a lightweight and fast machine learning model.



---

## 🚀 Features

- **Real-Time Classification:** Detects and classifies 10 different static signs in real-time.  
- **Two-Hand Support:** Recognizes signs requiring one or two hands via landmark padding.  
- **Virtual Webcam Output:** Uses `pyvirtualcam` to create a new “webcam” device for video conferencing apps.  
- **Lightweight Model:** Employs `scikit-learn`’s Random Forest — fast, CPU-only.  
- **Customizable:** Collect and train on your own set of static gestures.

---

## ⚙️ How It Works — The 4-Script Pipeline

### 1️⃣ `collect_imgs.py` — Data Collection  
- Uses **OpenCV** to access your webcam.  
- Loops through each class (10 by default, from `'0'` to `'9'`).  
- For each class:
  - Prompts you to press `Q` to begin.  
  - Rapidly saves **500 images** to `data/0`, `data/1`, etc.  

---

### 2️⃣ `create_datasets.py` — Feature Extraction  
- Loads all captured images.  
- Uses **MediaPipe** to detect hand landmarks.  
- Normalizes and pads data (84 features total).  
- Saves all landmark vectors + labels into `data.pickle`.  

---

### 3️⃣ `train_classifier.py` — Model Training  
- Loads `data.pickle`.  
- Splits data: **80% training, 20% testing**.  
- Trains a `RandomForestClassifier`.  
- Evaluates accuracy on unseen data.  
- Saves trained model as `model.p`.  

---

### 4️⃣ `Main.py` — Real-Time Application  
- Loads your trained `model.p`.  
- Runs MediaPipe live on webcam feed.  
- Predicts a sign for each frame.  
- Draws hand skeleton + label (e.g., *“Thank You”*).  
- Streams annotated feed to a **virtual webcam**.

---

## ⚡ Quick Start

### 🧩 Prerequisites
- Python **3.8+**  
- A working **webcam**  
- A **virtual camera driver**  
  - **Windows:** Install [OBS Studio Virtual Cam Plugin](https://obsproject.com/).  
  - **macOS:** No extra driver required.  
  - **Linux:**  
    ```bash
    sudo apt install v4l2loopback-dkms
    ```

---

### 📥 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SignCareMachineLearningModel.git
cd SignCareMachineLearningModel
