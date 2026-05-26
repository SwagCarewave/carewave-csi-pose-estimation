# CareWave CSI Pose Estimation

WiFi CSI(Channel State Information)-based Human Pose Estimation and Skeleton Reconstruction using Conv1D-LSTM Sequence Learning.

---

# Overview

This project aims to reconstruct human body skeletons using only WiFi CSI signals without cameras or wearable devices.

Unlike traditional action classification systems that only predict labels such as “fall” or “walking”, this project directly predicts human joint coordinates `(x, y)` and visualizes them as a skeleton structure.

The final goal is:

```text
CSI Signal
→ Deep Learning Model
→ Human Pose Coordinates
→ Skeleton Visualization
```

---

# Project Pipeline

```text
Human Action
→ ESP32-S3 CSI Collection
→ CSI Feature Extraction
→ MediaPipe Pose Label Generation
→ Sequence Dataset Construction
→ Conv1D + LSTM Pose Regression
→ Skeleton Prediction
→ Visualization
```

---

# Dataset

## Collected Actions

- standing
- sitting
- sit-stand
- walking
- hand wave
- arm movement
- body rotation
- lying
- fall actions

Total:
- 22 action videos
- ~2 minutes per action

---

# Final Dataset Structure

```text
X shape: (10657, 15, 12)
Y shape: (10657, 26)
```

Meaning:

```text
10657 → total sequence samples
15    → sequence length
12    → CSI feature count
26    → pose coordinate count
```

The model predicts the final pose coordinates from 15 consecutive CSI feature frames.

---

# Pose Labels

Pose labels were generated using MediaPipe Pose.

Selected joints:

- nose
- shoulders
- elbows
- wrists
- hips
- knees
- ankles

Each joint uses normalized `(x, y)` coordinates.

Total output dimension:

```text
13 joints × 2 coordinates = 26 outputs
```

---

# Model Architecture

Final architecture:

```text
CSI Sequence
→ Conv1D
→ BatchNormalization
→ LSTM(128)
→ LSTM(64)
→ Dense Layers
→ Pose Coordinates (26)
```

Main improvements:
- Sequence-based learning
- Conv1D temporal feature extraction
- Huber Loss
- Learning rate scheduling
- Prediction smoothing

---

# Sequence Learning Structure

Previous approach:

```text
Single CSI feature
→ Predict current pose
```

Improved approach:

```text
15 consecutive CSI frames
→ Predict final pose coordinates
```

This allows the model to learn temporal human motion flow rather than static CSI patterns.

Example:

```text
standing
→ body movement
→ falling motion
→ ground pose
```

The model learns the full CSI transition sequence.

---

# Performance

## Final Evaluation

```text
Test MAE : 0.00736
Test MSE : 0.000152
```

Compared to the previous single-frame model:

```text
MAE reduced by approximately 29%
(0.0104 → 0.00736)
```

---

# Visualization

The system visualizes:

1. Original video
2. Real MediaPipe pose
3. CSI-predicted pose

This allows direct comparison between actual human pose and CSI-based predicted skeleton.

---

# Main Files

```text
batch_extract_pose.py
→ Extract MediaPipe pose labels from videos

build_sequence_dataset_all.py
→ Construct CSI sequence dataset

train_pose_model_sequence.py
→ Train Conv1D-LSTM pose regression model

evaluate_pose_model_sequence.py
→ Evaluate joint/action MAE

video_pose_compare_all.py
→ Visualize predicted skeletons
```

---

# Technologies

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- WiFi CSI
- ESP32-S3

---

# Future Work

- Real-time CSI streaming
- Web dashboard skeleton rendering
- Transformer-based architecture
- Multi-person pose estimation
- Real-time fall detection

---

# Author

CareWave Project Team  
SwagCarewave
