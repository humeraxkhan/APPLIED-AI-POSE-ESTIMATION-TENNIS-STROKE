# Tennis Stroke Classification using LSTM and Pose Estimation


##  Overview

This project implements an end-to-end deep learning pipeline for classifying tennis strokes from video clips. The system uses **MediaPipe Pose** for extracting human pose keypoints and an **LSTM neural network** for temporal sequence classification.

### Problem Statement
Given a video clip of a tennis player, classify the stroke type into one of four categories:
-  **Forehand**
- **Backhand** 
-  **Serves**
-  **NoStroke** (no action)

### Key Achievements
-  **80% Test Accuracy** with LSTM model
- **100% Test Accuracy** with baseline MLP
- Fully reproducible experiments with seed control
- Comprehensive data augmentation pipeline
- real-time inference capability

---

##  Features

- **Pose Extraction**: Automated keypoint extraction using MediaPipe
- **Data Augmentation**: Horizontal flipping and Gaussian noise
- **Advanced LSTM Architecture**: Two-layer LSTM with dropout regularization
- **Training Controls**: Early stopping, learning rate scheduling, model checkpointing
- **Baseline Comparison**: Simple MLP model for performance benchmarking
- **Hyperparameter Tuning**: Systematic search over learning rates and batch sizes
- **Visualization Tools**: Confusion matrix, training curves, pose overlay

---

---

##  Requirements

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional (CUDA-compatible for faster training)

### Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.11.4 | Programming language |
| PyTorch | ≥ 2.5.1 | Deep learning framework |
| OpenCV | ≥ 4.12.0 | Video processing |
| MediaPipe | ≥ 0.10.14 | Pose estimation |
| scikit-learn | ≥ 1.7.2 | ML utilities |
| NumPy | ≥ 1.26.4 | Numerical computing |
| Matplotlib | ≥ 3.8.3 | Visualization |
| tqdm | latest | Progress bars |

---

##  Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/tennis-stroke-classification.git
cd tennis-stroke-classification
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv tennis_env
source tennis_env/bin/activate  # On Windows: tennis_env\Scripts\activate

# Or using conda
conda create -n tennis_env python=3.11
conda activate tennis_env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```python
python -c "import torch; import cv2; import mediapipe as mp; print('✓ All packages installed successfully')"
```

### Step 5: Install Jupyter (if needed)
```bash
pip install jupyter notebook
jupyter notebook
```

---

##  Dataset

### Dataset Composition
The dataset contains **100 video clips** (MP4 format) organized into 4 categories:

| Category | Videos | Duration | Description |
|----------|--------|----------|-------------|
| Forehand | 20 | ~3-5s | Forehand stroke execution |
| Backhand | 20 | ~3-5s | Backhand stroke execution |
| Serves | 20 | ~3-5s | Service motion |
| NoStroke | 40 | ~3-5s | No specific stroke (imbalanced) |

### Dataset Split (Stratified)
```
Total: 100 videos
├── Training:   60 videos (60%) → After augmentation: 120 samples
├── Validation: 20 videos (20%)
└── Test:       20 videos (20%)

Split with random_state=42 for reproducibility
```

### Class Distribution in Splits

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Forehand | 12 | 4 | 4 | 20 |
| Backhand | 12 | 4 | 4 | 20 |
| Serves | 12 | 4 | 4 | 20 |
| NoStroke | 24 | 8 | 8 | 40 |

### Data Format
- **Input**: MP4 video files
- **Processed**: NumPy arrays (.npy) containing pose keypoints
- **Shape**: (num_frames, 66) where 66 = 33 landmarks × 2 (x, y)
- **Sequence Length**: Fixed at 40 frames

---

## Quick Start

### Option 1: Run Complete Pipeline
```bash
# Open Jupyter Notebook
jupyter notebook notebooks/APPLIED_AI_MAIN_PROJECTFINAL.ipynb

# Run all cells in order (Runtime → Run All)
```

### Option 2: Step-by-Step Execution

#### Extract Pose Keypoints
```python
# Update paths to your dataset location
DATASET_PATH = r"C:\path\to\VideoDataset\VideoDataset"
OUTPUT_PATH = r"C:\path\to\TennisKeypoints"

# Run cells [22] - Extracts keypoints from all videos
# Output: .npy files in TennisKeypoints/ folder
# Time: ~5-6 minutes for 100 videos
```

####Load and Preprocess Data
```python
# Run cells [36], [44], [45], [46]
# - Loads keypoint sequences
# - Creates train/val/test splits
# - Applies data augmentation
# - Normalizes features
```

#### Train LSTM Model
```python
# Run cells [48], [49], [62], [72]
# - Initializes LSTM model
# - Sets up training controls
# - Trains with early stopping
# - Saves best model weights
```

#### Evaluate and Visualize
```python
# Run cell [73] - Test set evaluation
# Run cell [78] - Confusion matrix
# Run cell [86] - Sample predictions with pose visualization
```

---

##  Methodology

### 1. Pose Extraction Pipeline

```python
# MediaPipe Pose Configuration
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5
)

# Extract 33 body landmarks (66 features: x, y)
# Frame skip = 2 (process every 2nd frame for stability)
```

**MediaPipe Landmarks** (33 points):
- Face: nose, eyes, ears, mouth
- Upper body: shoulders, elbows, wrists
- Core: hips
- Lower body: knees, ankles
- Hands: pinky, index, thumb

### 2. Data Preprocessing

**Sequence Processing:**
1. Load .npy keypoint files
2. **Padding**: If frames < 40 → pad with zeros
3. **Trimming**: If frames > 40 → keep first 40
4. **Per-sample normalization**: (x - min) / (max - min)
5. **Global standardization**: (x - μ_train) / σ_train

**Normalization Formula:**
```python
# Step 1: Per-sample min-max
data_norm = (data - data.min()) / (data.max() - data.min())

# Step 2: Global z-score using training statistics
mu = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_normalized = (X - mu) / std
```

### 3. Data Augmentation (Training Only)

Applied to training set **after** splitting:

```python
# Horizontal Flip
x_flip[:, ::2] = 1 - x_flip[:, ::2]  # Flip x-coordinates

# Gaussian Noise
noise = np.random.normal(0, 0.01, shape)
x_noisy = np.clip(x_flip + noise, 0, 1)
```

**Result**: 60 → 120 training samples (2x increase)

### 4. Model Training Strategy

```python
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, 
    patience=5, min_lr=1e-6
)

# Early Stopping
patience = 10  # Stop if no improvement for 10 epochs

# Batch Size
batch_size = 8

# Loss Function
criterion = nn.CrossEntropyLoss()
```

---

##  Model Architecture

### LSTM Classifier (Main Model)

```
Input: (batch_size, 40, 66)
    ↓
┌─────────────────────────────────────┐
│ LSTM Layer 1                        │
│ - Input size: 66                    │
│ - Hidden size: 128                  │
│ - Bidirectional: False              │
│ - Parameters: ~100K                 │
└─────────────────────────────────────┘
    ↓
Dropout(p=0.3)
    ↓
┌─────────────────────────────────────┐
│ LSTM Layer 2                        │
│ - Input size: 128                   │
│ - Hidden size: 64                   │
│ - Parameters: ~50K                  │
└─────────────────────────────────────┘
    ↓
Take Last Timestep: [:, -1, :]
    ↓
Dropout(p=0.3)
    ↓
┌─────────────────────────────────────┐
│ Fully Connected 1                   │
│ - Input: 64 → Output: 64            │
│ - Activation: ReLU                  │
│ - Parameters: 4,160                 │
└─────────────────────────────────────┘
    ↓
Dropout(p=0.3)
    ↓
┌─────────────────────────────────────┐
│ Output Layer                        │
│ - Input: 64 → Output: 4             │
│ - Parameters: 260                   │
└─────────────────────────────────────┘
    ↓
Output: (batch_size, 4) [Logits]
```

**Total Parameters**: ~154K

### Baseline MLP (Comparison Model)

```
Input: (batch_size, 40, 66)
    ↓
Flatten → (batch_size, 2640)
    ↓
FC(2640 → 64) + ReLU + Dropout(0.3)
    ↓
FC(64 → 32) + ReLU
    ↓
FC(32 → 4)
    ↓
Output: (batch_size, 4) [Logits]
```

**Key Difference**: MLP ignores temporal order, treating all frames as independent features.

---

## 📈 Results

### Final Performance Comparison

| Model | Architecture | Train Acc | Val Acc | Test Acc | Parameters |
|-------|-------------|-----------|---------|----------|------------|
| **LSTM** | 2-layer LSTM | 96.67% | 85.00% | **80.00%** | ~154K |
| **Baseline MLP** | 3-layer FC | 95.83% | 95.00% | **100.00%** | ~170K |

### Training Dynamics

**LSTM Model:**
```
Epoch 1: Train Loss=0.1611, Train Acc=96.67%, Val Loss=0.3815, Val Acc=85.00%
Early stopping triggered (no improvement for 10 epochs)
```

**Key Observation**: Model converged extremely fast (1 epoch), indicating:
- Effective data augmentation
- Potential dataset simplicity
- Risk of overfitting to validation set

### Confusion Matrix (LSTM - Test Set)

```
                Predicted
              FH   BH   Srv  NS
Actual FH  [  4    0    0    0  ]  100%
       BH  [  0    3    0    1  ]   75%
       Srv [  0    0    4    0  ]  100%
       NS  [  0    1    2    5  ]   62.5%
```

**Per-Class Performance:**
- Forehand: 4/4 correct (100%)
- Backhand: 3/4 correct (75%) - 1 misclassified as NoStroke
- Serves: 4/4 correct (100%)
- NoStroke: 5/8 correct (62.5%) - most errors occur here

### Hyperparameter Tuning Results

Systematic search over 6 configurations (30 epochs each):

| LR | Batch Size | Best Val Acc | Rank |
|----|------------|--------------|------|
| 0.001 | 4 | **75.0%** | 🥇 1st |
| 0.001 | 8 | 45.0% | 3rd |
| 0.0003 | 8 | 50.0% | 2nd |
| 0.0007 | 4 | 40.0% | 4th |
| 0.0007 | 8 | 40.0% | 4th |
| 0.0003 | 4 | 40.0% | 4th |

**Optimal Configuration**: `LR=0.001, Batch Size=4`

---

##  Usage Examples

### Example 1: Inference on New Video

```python
import cv2
import numpy as np
import torch
import mediapipe as mp

# Load trained model
model = LSTMClassifier(input_dim=66, num_classes=4)
model.load_state_dict(torch.load('models/tennis_lstm_best.pth'))
model.eval()

# Extract pose from video
def predict_stroke(video_path):
    # Extract keypoints
    keypoints = extract_keypoints_from_video(video_path)
    
    # Preprocess
    seq = preprocess_sequence(keypoints, seq_length=40)
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(seq_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    
    categories = ["Forehand", "Backhand", "Serves", "NoStroke"]
    return categories[pred_class], confidence

# Use
stroke, conf = predict_stroke("new_video.mp4")
print(f"Predicted: {stroke} (Confidence: {conf:.2%})")
```

### Example 2: Visualize Predictions

```python
# Load video frame and overlay prediction
frame = load_video_frame(video_path, frame_idx=0)
stroke, conf = predict_stroke(video_path)

# Draw pose skeleton
pose_frame = draw_pose_landmarks(frame)

# Add prediction text
cv2.putText(pose_frame, f"{stroke}: {conf:.2%}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2)

# Display
plt.imshow(pose_frame)
plt.title("Pose Detection & Classification")
plt.show()
```

### Example 3: Batch Processing

```python
# Process multiple videos
video_dir = "test_videos/"
results = []

for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_dir, video_file)
        stroke, conf = predict_stroke(video_path)
        results.append({
            'video': video_file,
            'prediction': stroke,
            'confidence': conf
        })

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('predictions.csv', index=False)
print(df)
```

---

##  Reproducibility

All experiments are **fully reproducible** using seed control:

### Seeds Used
```python
import random
import numpy as np
import torch

# Set all seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Additional PyTorch settings for determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Reproduction Steps

1. **Dataset Split**: Use `random_state=42` in `train_test_split()`
2. **Pose Extraction**: Process with `frame_skip=2`
3. **Augmentation**: Apply with `np.random.seed(42)` before augmentation
4. **Training**: Use provided hyperparameters exactly
5. **Evaluation**: Load `tennis_lstm_best.pth` weights

### Expected Variance
Due to random initialization, expect test accuracy within **±2%** of reported values.

---

## ⚠️ Limitations

### Current Limitations

1. **Small Dataset** 
   - Only 100 videos total
   - Limited diversity in players, courts, conditions
   - May not generalize to professional matches

2. **Single Camera Angle**
   - All videos captured from back of court
   - Cannot handle side-view or front-view footage

3. **Simplified Features**
   - Only x, y coordinates used (z-depth ignored)
   - No velocity or acceleration features
   - No temporal derivatives

4. **Class Imbalance**
   - NoStroke has 2× samples of other classes
   - May bias model towards predicting NoStroke

5. **Binary Coordinates**
   - MediaPipe provides x, y, z, visibility
   - Current model only uses x, y (66 features)
   - Missing 50% of available information

6. **Frame Skip Trade-off**
   - Skip=2 improves stability but loses temporal detail
   - May miss subtle motion differences

7. **Fixed Sequence Length**
   - Padding/trimming to 40 frames
   - May lose important motion at start/end
   - Not adaptive to video duration

---

## 🚀 Future Work



### Long-Term Vision (3-6 months)

7. **Multi-Modal Learning**
   - Combine pose + raw video + audio
   - Ensemble multiple models
   - Late fusion of predictions

8. **Real-Time Application**
   - Optimize for mobile deployment (TensorFlow Lite)
   - Edge device inference (NVIDIA Jetson)
   - Live stroke classification during play

9. **Advanced Analytics**
   - Stroke quality assessment (good vs bad technique)
   - Player style profiling
   - Temporal segmentation (detect exact stroke boundaries)
   - Trajectory prediction

10. **Synthetic Data Generation**
    ```python
    # Use GANs to generate synthetic poses
    from diffusion_models import PoseGenerator
    generator = PoseGenerator()
    synthetic_poses = generator.generate(class_label='forehand', n=100)
    ```

---

##  Acknowledgments

### Libraries and Frameworks
- **PyTorch**: Deep learning framework - [pytorch.org](https://pytorch.org/)
- **MediaPipe**: Pose estimation - [mediapipe.dev](https://mediapipe.dev/)
- **OpenCV**: Computer vision utilities - [opencv.org](https://opencv.org/)
- **scikit-learn**: Machine learning tools - [scikit-learn.org](https://scikit-learn.org/)

### Academic References
This project was completed as part of:
- **Course**: COMP41790 – Applied AI in Vision & Imaging
- **Institution**: University College Dublin
- **Year**: 2025

### Inspiration
- MediaPipe Pose: [arXiv:2006.10204](https://arxiv.org/abs/2006.10204)
- LSTM Networks: Hochreiter & Schmidhuber (1997)
- Sports Action Recognition: Survey by Ramanathan et al.

---





## 📚 Additional Resources

### Tutorials
- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [MediaPipe Pose Guide](https://google.github.io/mediapipe/solutions/pose)
- [Time Series Classification](https://github.com/timeseriesAI/tsai)

### Related Papers
1. **Pose Estimation**: "BlazePose: On-device Real-time Body Pose tracking"
2. **Action Recognition**: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
3. **LSTM**: "Long Short-Term Memory" - Hochreiter & Schmidhuber

### Datasets
- **Tennis Dataset**: UCF101 Sports Action subset
- **General Actions**: Kinetics-400, Kinetics-700
- **Pose Dataset**: MPII Human Pose, COCO Keypoints

---

## Project Checklist

Before submission, ensure:

- [ ] All code runs without errors
- [ ] requirements.txt is complete
- [ ] Model weights are saved (tennis_lstm_best.pth)
- [ ] README.md is comprehensive
- [ ] Jupyter notebook has clear comments
- [ ] Reproducibility is verified (random seeds)
- [ ] Results match reported values (±2%)
- [ ] No dataset files uploaded (only code)
- [ ] Video presentation is recorded (5 min max)
- [ ] All paths are relative or configurable

---

## 📊 Quick Reference

### Key Hyperparameters
```python
SEQUENCE_LENGTH = 40
FEATURE_DIM = 66
NUM_CLASSES = 4
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
DROPOUT = 0.3
LSTM_HIDDEN_1 = 128
LSTM_HIDDEN_2 = 64
EARLY_STOP_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
```

### File Paths
```python
DATASET_PATH = "data/VideoDataset/VideoDataset"
KEYPOINTS_PATH = "data/TennisKeypoints"
MODEL_PATH = "models/tennis_lstm_best.pth"
```

### Class Labels
```python
categories = ["Forehand", "Backhand", "Serves", "NoStroke"]
label_map = {0: "Forehand", 1: "Backhand", 2: "Serves", 3: "NoStroke"}
```


