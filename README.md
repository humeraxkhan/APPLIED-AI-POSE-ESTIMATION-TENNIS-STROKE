# Tennis Stroke Classification using LSTM & Pose Estimation

## Project Overview

This project implements an end-to-end deep learning pipeline for automated tennis stroke classification from video data. Using MediaPipe Pose for human keypoint extraction and LSTM neural networks for temporal sequence modeling, the system accurately identifies four types of tennis actions: Forehand, Backhand, Serves, and NoStroke.

### Key Features
- Pose-based approach extracting 33 body landmarks (66 x,y coordinates) using MediaPipe
- 2-layer LSTM architecture capturing motion dynamics across 40-frame sequences
- Data augmentation with horizontal flipping and Gaussian noise
- Robust training with early stopping, learning rate scheduling, and model checkpointing
- Comprehensive evaluation including baseline comparison and hyperparameter tuning

### Architecture Pipeline

```
Video Input → MediaPipe Pose → Keypoint Extraction → Sequence Processing
                                                            ↓
Test Results ← Model Inference ← LSTM Classifier ← Data Augmentation
```

### Results Summary

| Model | Architecture | Test Accuracy | Parameters |
|-------|-------------|---------------|------------|
| LSTM (Main) | 2-layer LSTM | 80.0% | ~154K |
| Baseline MLP | 3-layer FC | 100.0% | ~170K |

**Dataset**: 100 videos (Forehand: 20, Backhand: 20, Serves: 20, NoStroke: 40)  
**Split**: 60% Train / 20% Validation / 20% Test (stratified with random_state=42)  
**Training**: Converged in 1 epoch with early stopping

---

## Requirements

### Software Dependencies
- Python >= 3.11.4
- PyTorch >= 2.5.1
- OpenCV >= 4.12.0
- MediaPipe >= 0.10.14
- scikit-learn >= 1.7.2
- NumPy, Matplotlib, tqdm

See `requirements.txt` for complete list.

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, cv2, mediapipe as mp; print('Installation successful')"

# Launch Jupyter Notebook
jupyter notebook APPLIED_AI_MAIN_PROJECTFINAL.ipynb
```

---

## How to Run

### Dataset Preparation

Place your video dataset in this structure:
```
VideoDataset/VideoDataset/
├── Forehand/
├── Backhand/
├── Serves/
└── NoStroke/
```

### Execution Steps

Open `APPLIED_AI_MAIN_PROJECTFINAL.ipynb` and run cells in sequence:

#### Phase 1: Setup & Verification (Cells 2, 11, 14-21)
- Imports packages and verifies versions
- Extracts dataset and confirms structure

#### Phase 2: Pose Extraction (Cell 22) - Takes ~5-6 minutes
Update these paths first:
```python
DATASET_PATH = r"C:\Users\YourName\Downloads\VideoDataset\VideoDataset"
OUTPUT_PATH = r"C:\Users\YourName\Downloads\TennisKeypoints"
```
- Processes all 100 videos with MediaPipe Pose
- Extracts 33 body landmarks per frame (66 features)
- Saves keypoint sequences as .npy files

#### Phase 3: Data Loading & Preprocessing (Cells 36, 40, 44)
- Loads extracted keypoint sequences
- Pads/trims all sequences to 40 frames
- Creates stratified 60/20/20 train/val/test split

#### Phase 4: Data Augmentation (Cells 45, 46)
- Applies horizontal flip and Gaussian noise to training set
- Doubles training samples from 60 to 120
- Normalizes using training set statistics

#### Phase 5: Model Training (Cells 47-49, 62, 72)
- Initializes 2-layer LSTM (128 → 64 hidden units)
- Trains with Adam optimizer (lr=1e-3, batch_size=8)
- Applies early stopping and LR scheduling
- Saves best model to `tennis_lstm_best.pth`

Expected output:
```
Epoch [1/100] | Train Loss: 0.1611, Train Acc: 0.9667 | Val Loss: 0.3815, Val Acc: 0.8500
Early stopping triggered
```

#### Phase 6: Model Evaluation (Cell 73)
- Evaluates on test set
- Output: `FINAL TEST ACCURACY: 80.00%`

#### Phase 7: Results Visualization (Cells 76, 78, 86)
- Generates confusion matrix
- Visualizes predictions with pose overlays

### Optional Components

**Baseline Model** (Cells 79-82): Train simple MLP for comparison  
**Hyperparameter Tuning** (Cell 86): Test different learning rates and batch sizes

---


## File Structure

```
project/
├── APPLIED_AI_MAIN_PROJECTFINAL.ipynb  # Main notebook
├── requirements.txt                     # Dependencies
├── README.md                            # This file
├── tennis_lstm_best.pth                 # Model weights (generated)
├── VideoDataset/VideoDataset/           # Input videos
└── TennisKeypoints/                     # Extracted keypoints (generated)
```

---

## Reproducibility

All experiments use fixed random seeds for reproducibility:
```python
np.random.seed(42)
random_state=42  # for train_test_split
```

Expected variance: Results may vary ±2% due to random weight initialization.

---

## Troubleshooting

**MediaPipe Installation Error**
```bash
pip install --upgrade mediapipe
```

**CUDA/GPU Not Detected**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Path Errors (Windows)**
```python
# Use raw strings:
DATASET_PATH = r"C:\Users\Name\VideoDataset"
# Or forward slashes:
DATASET_PATH = "C:/Users/Name/VideoDataset"
```



