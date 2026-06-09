<img width="1919" height="828" alt="image" src="https://github.com/user-attachments/assets/86c93e3d-6a76-4a4d-adda-1a928260299f" /># CNN-Based Image Authenticity Detection

A Deep Learning-based Computer Vision project that classifies images as **Real** or **AI-Generated** using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## Project Overview

With the increasing use of AI image generation tools, distinguishing between authentic and synthetic images has become a significant challenge. This project aims to address that problem by developing a CNN-based image classification system capable of automatically identifying whether an image is real or AI-generated.

The model is trained on a dataset containing both real and AI-generated images and can perform predictions on unseen images.

---

## Problem Statement

AI-generated images are becoming increasingly realistic, making manual detection difficult. This project leverages Deep Learning techniques to automate the classification process and improve image authenticity detection.

---

## Objectives

- Build a CNN-based image classification model
- Distinguish between real and AI-generated images
- Perform image preprocessing and feature extraction
- Train and evaluate a deep learning model
- Save and reuse the trained model for prediction

---

## Technologies Used

### Programming Language
- Python

### Frameworks & Libraries
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-Learn

### Development Environment
- Google Colab
- Jupyter Notebook

### Deployment
- Streamlit
- Ngrok

---

## Dataset

### Source
Kaggle Dataset

### Classes
- Real Images
- AI-Generated Images

The dataset was organized into two categories for binary image classification.

---

## Methodology

### 1. Data Collection
Collected real and AI-generated images from a publicly available dataset.

### 2. Data Preprocessing
Performed preprocessing using OpenCV:
- Image resizing
- Pixel normalization
- Label encoding
- Conversion to NumPy arrays

### 3. Dataset Splitting
The dataset was divided into:
- Training Set
- Testing Set

using Scikit-Learn's train_test_split() function.

### 4. CNN Model Development
A Convolutional Neural Network was designed using TensorFlow and Keras consisting of:

- Convolution Layers
- Max Pooling Layers
- Flatten Layer
- Dense Layers
- Output Layer

### 5. Model Training
The model was trained to learn image features and patterns that distinguish real images from AI-generated images.

### 6. Model Evaluation
Model performance was evaluated on unseen test images.

### 7. Model Saving
The trained model was saved as:

```bash
ai_image_detector.keras
```

### 8. Prediction
The trained model can predict whether a given image is:

- Real Image
- AI-Generated Image

---

## CNN Architecture

```text
Input Image
      ↓
Convolution Layer
      ↓
Max Pooling Layer
      ↓
Convolution Layer
      ↓
Max Pooling Layer
      ↓
Flatten Layer
      ↓
Dense Layer
      ↓
Output Layer
```

---

## Features

- Real vs AI-generated image classification
- Image preprocessing pipeline
- CNN-based Deep Learning model
- Model training and evaluation
- Saved model for future inference
- Streamlit integration support

---

## Screenshots

### Model Training

<img width="1919" height="865" alt="image" src="https://github.com/user-attachments/assets/b168366c-a820-4888-bed4-c19ffd63582f" />


### Prediction Output

### For Ai image
<img width="1919" height="849" alt="image" src="https://github.com/user-attachments/assets/4e1df13c-895f-4d87-8dff-0a2e8caa0920" />

### For Real image
<img width="1919" height="849" alt="image" src="https://github.com/user-attachments/assets/3447f7f7-29e1-4f14-9cd7-52565e2e0d64" />


---

## Project Structure

```text
cnn-image-authenticity-detection/
│
├── app.py
├── ai_image_detector.keras
├── requirements.txt
├── README.md
│
└── Screenshots/
    ├── home.png
    ├── real_prediction.png
    └── ai_prediction.png
```

---

## Future Enhancements

- Increase dataset size
- Improve model accuracy
- Support multiple AI image generators
- Cloud deployment
- Explainable AI integration

---

## Learning Outcomes

Through this project, I gained practical experience in:

- Computer Vision
- Deep Learning
- CNN Architecture
- TensorFlow & Keras
- OpenCV
- Data Preprocessing
- Model Evaluation
- AI Model Deployment

---

## Author

**Logesh N A**

B.Tech Artificial Intelligence and Machine Learning

Saveetha Engineering College

GitHub: https://github.com/Logesh051
