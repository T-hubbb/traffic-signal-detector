# 🚦 Traffic Sign Detection & Classification

A traffic sign classification system built with Python, CNNs, and OpenCV.
Trained on the GTSRB dataset (43 classes, 50,000+ images) and deployable both
as a desktop app for image uploads and as a web app.

**Test Accuracy: 90%**

---


https://github.com/user-attachments/assets/791fa1ce-b4e9-41b2-b5de-567fc766cb3c



## 🎯 What it does

- Classifies 43 types of traffic signs from an uploaded image
- Trained from scratch using a Convolutional Neural Network (CNN)
- Handles class imbalance via image augmentation
- Desktop GUI (`local_app.py`, built with Tkinter): choose an image file and instantly see the predicted sign class
- Also deployable as a web app (`app.py`) for browser-based image upload
---

## 🏗️ Architecture
The model is a Convolutional Neural Network trained from scratch (no transfer learning):

Input (32×32×3 RGB image)
→ Conv2D (32 filters, 3×3) + ReLU → MaxPooling2D
→ Conv2D (64 filters, 3×3) + ReLU → MaxPooling2D
→ Dropout (0.25)
→ Flatten
→ Dense (256) + ReLU → Dropout (0.5)
→ Dense (43) + Softmax

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Model | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Dataset | [GTSRB](http://benchmark.ini.rub.de/) |

---

## 🚀 Run Locally

```bash
git clone https://github.com/T-hubbb/traffic-signal-detector
cd traffic-signal-detector
pip install -r requirements.txt

# Webcam (real-time)
python local_app.py

# Upload image (web app)
python app.py
```

---

## 📊 Sample Output

![Detection Output](output.png)

---

## 🔮 What I'd improve next

- Containerise with Docker for one-command deployment
- Add confidence scores to predictions ("92% — Stop Sign")
- Replace with MobileNet for faster inference on edge/embedded devices
- Train on full GTSRB with better augmentation pipeline

---

## 📁 Project Structure

| File | Purpose |
|---|---|
| `Traffic_signal_model_training.ipynb` | Full training pipeline |
| `model.h5` | Saved trained model |
| `app.py` | Web app (image upload) |
| `local_app.py` | Local webcam detection |
| `test.py` | Evaluation & metrics |
| `output.png` | Sample prediction output |
| `requirements.txt` | All dependencies |
