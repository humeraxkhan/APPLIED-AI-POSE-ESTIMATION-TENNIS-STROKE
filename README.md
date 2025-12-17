# APPLIED-AI-POSE-ESTIMATION-TENNIS-STROKE
# Pose-Based Tennis Stroke Recognition

This project implements a pose-based action recognition system to classify tennis strokes
from short video clips using computer vision and deep learning techniques.

Rather than relying on raw RGB video frames, the approach focuses on human pose keypoints
extracted from video data and models temporal motion patterns using a recurrent neural
network.

---

## Project Objective

The objective of this project is to automatically classify tennis strokes into the
following four categories:

- Forehand  
- Backhand  
- Serves  
- NoStroke  

By modeling pose dynamics over time, the system aims to robustly distinguish between
different stroke types while reducing background and appearance-related noise.

---

## Methodology Overview

1. **Pose Extraction**  
   Human pose keypoints are extracted from each video frame using **MediaPipe Pose**.
   Each frame yields 33 body landmarks represented by x and y coordinates, resulting
   in a 66-dimensional feature vector.

2. **Sequence Construction**  
   Each video is converted into a fixed-length sequence of 40 frames. Longer sequences
   are truncated, and shorter ones are zero-padded to ensure compatibility with temporal
   models.

3. **Data Preprocessing**  
   Normalization is applied on a per-sample basis to avoid data leakage. Data augmentation
   is applied only to the training set to improve generalization.

4. **Modeling**  
   A baseline multilayer perceptron is trained from scratch for comparison. The primary
   model is a two-layer **PyTorch LSTM** network designed to capture temporal dependencies
   in pose sequences.

5. **Evaluation**  
   Model performance is evaluated using classification accuracy and confusion matrices
   on a held-out test set.

---

## Technologies Used

- Python  
- PyTorch (≥ 2.1.0)  
- MediaPipe Pose (≥ 0.10.0)  
- OpenCV (≥ 4.10.0)  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  


---

## Project Structure

