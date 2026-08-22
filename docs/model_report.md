# GESCOM Model and Implementation Report

**Project:** GESCOM ISL Non-Manual Feature Translation Dashboard  
**Report date:** 22 August 2026

## 1. Objective

GESCOM captures facial, hand, and upper-body movements from a live webcam and predicts one of the labeled Indian Sign Language sentence classes. The system converts the detected prediction into text and can optionally speak the result using text-to-speech.

## 2. Dataset

The local ISL_CSLRT_Corpus contains:

- 492 labeled video samples
- 101 sentence classes
- 687 raw video files in the complete corpus directory
- 492 extracted feature sequences used for training
- 30 temporal steps per sequence
- 534 features per temporal step

The GitHub repository excludes the raw dataset and generated feature cache because they are large local files. They must be available locally before preprocessing or retraining.

## 3. Feature Pipeline

Each video frame is processed by MediaPipe landmark models:

- Face landmarker: facial mesh and iris-related landmarks
- Hand landmarker: left and right hand landmarks
- Pose landmarker: upper-body pose landmarks

The resulting 534-dimensional vector contains:

| Feature group | Dimensions |
| --- | ---: |
| Hand-crafted facial and head cues | 18 |
| Normalized selected face landmarks | 354 |
| Normalized left-hand landmarks | 63 |
| Normalized right-hand landmarks | 63 |
| Normalized upper-body pose landmarks | 36 |
| **Total per frame** | **534** |

The system normalizes face and body coordinates relative to the nose, eyes, wrist, or shoulder center. This reduces sensitivity to camera distance and signer position.

The temporal model receives a tensor shaped:

```text
(batch_size, 30 frames, 534 features)
```

## 4. Models

### 4.1 BiLSTM with Attention

The primary live webcam model is a PyTorch two-layer bidirectional LSTM followed by self-attention and a dense classification head.

```text
30 x 534 input
    -> 2-layer bidirectional LSTM
    -> 256-dimensional temporal output
    -> self-attention pooling
    -> dense layer: 256 -> 128
    -> 101-class output
```

The trained live model is loaded from:

```text
models/best_model.pth
```

The model is used by [app.py](../app.py) through [inference.py](../ml/inference.py).

### 4.2 XGBoost Benchmark

XGBoost was tested as a traditional machine-learning comparison. The sequence was summarized using mean, standard deviation, minimum, maximum, and average frame-to-frame movement, producing 2,670 fixed features.

XGBoost is stored in:

```text
models/xgboost_model.json
```

It is not the live webcam model because it performed worse than the BiLSTM on the available corpus.

### 4.3 Regression

Regression is not appropriate for this task. The system predicts one class from 101 sentence labels, so this is a multiclass classification problem.

## 5. Verified Results

Both models were evaluated using the `full` split. This split uses the same samples for training and monitoring, so these values are not independent real-world test scores.

| Model | Top-1 accuracy | Top-3 accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| BiLSTM + Attention | 40.24% | 53.25% | 0.4551 |
| XGBoost | 27.85% | 39.23% | 0.3056 |

The BiLSTM is currently the better model for temporal gesture recognition. Its measured CPU inference speed in the evaluation script was approximately 1.27 ms per sample, although MediaPipe landmark extraction is the main cost during live webcam use.

## 6. Live Webcam Workflow

1. The browser requests webcam permission using `getUserMedia()`.
2. The live video is shown in the browser.
3. Frames are captured at up to approximately 10 frames per second.
4. Frames are sent temporarily to the Flask `/predict` endpoint.
5. MediaPipe extracts the 534 features.
6. The backend maintains a 30-frame temporal buffer.
7. The BiLSTM predicts the current sentence class.
8. Frontend smoothing reduces flickering predictions.
9. Stable predictions are added to the translation text output.

The webcam preview is mirrored for the user, but the frame sent to inference keeps the original orientation so left and right landmarks match the training data.

## 7. Beginner Setup and Training Steps

### Step 1: Open the project

```powershell
cd E:\anuj\ISL_NonManual_AI
```

Place the dataset beside the project:

```text
E:\anuj\ISL_CSLRT_Corpus
E:\anuj\ISL_NonManual_AI
```

### Step 2: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3: Inspect the dataset

```powershell
python ml/dataset_inspection.py
```

Confirm that the metadata contains the expected samples and classes.

### Step 4: Extract features

```powershell
python ml/preprocessing.py
```

Confirm that `data/features/` contains one `.npy` file for each valid labeled sample.

### Step 5: Train the main model

```powershell
python ml/train.py --model temporal --split full --epochs 35
```

The important outputs are:

```text
models/best_model.pth
models/final_model_temporal.pth
models/history_temporal.json
models/labels.json
```

### Step 6: Evaluate the model

```powershell
python ml/evaluate.py --model temporal --split full
```

The metrics are written to:

```text
models/evaluation_temporal.json
```

### Step 7: Start the webcam app

```powershell
python app.py
```

Open `http://127.0.0.1:5000`, allow camera access, click **Start Camera**, and keep the face and hands visible.

### Step 8: Optional XGBoost comparison

```powershell
python ml/xgboost_model.py --split full --estimators 50
```

This creates `models/evaluation_xgboost.json` and `models/xgboost_model.json`.

## 8. Limitations and Recommendations

- There are only about 4 to 5 examples per class for 101 classes.
- The full split is not an independent test and can overstate generalization.
- The current model can still confuse similar gestures.
- Lighting, camera angle, distance, hand visibility, and signer variation affect webcam results.
- The most valuable improvement is collecting more webcam-style samples for each class.
- A signer-independent evaluation should use separate signers for training, validation, and testing.
- The model should be retrained after adding varied lighting, backgrounds, distances, and left/right orientations.

## 9. Main Project Files

- [config.py](../config.py): paths, feature dimensions, and hyperparameters
- [feature_extraction.py](../ml/feature_extraction.py): MediaPipe extraction and normalization
- [preprocessing.py](../ml/preprocessing.py): video-to-feature conversion
- [dataset_loader.py](../ml/dataset_loader.py): labels and dataset splits
- [model.py](../ml/model.py): BiLSTM and baseline architectures
- [train.py](../ml/train.py): PyTorch training loop
- [evaluate.py](../ml/evaluate.py): accuracy and evaluation report
- [xgboost_model.py](../ml/xgboost_model.py): optional XGBoost benchmark
- [app.py](../app.py): Flask live inference server
- [frontend/script.js](../frontend/script.js): webcam capture and prediction display
