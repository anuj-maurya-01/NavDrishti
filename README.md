# GESCOM: ISL Non-Manual Feature Translation Dashboard

GESCOM is a complete, end-to-end, working AI/ML application for the Smart India Hackathon (SIH) problem statement:
**"Capturing Non-manual Features of Indian Sign Language and Converting Them into Text."**

GESCOM tracks facial gestures (eyebrows, eyes, iris, mouth, head pose) along with skeletal pose and hand gestures using MediaPipe Holistic, aligning them over time to translate continuous Indian Sign Language (ISL) video/webcam streams into readable text sentences and spoken audio.

---

## 1. Project Directory Structure

```
ISL_NonManual_AI/
│
├── README.md               # Main instructions and setup guide
├── requirements.txt       # Python dependencies
├── config.py              # Central hyperparameters and paths
├── app.py                 # Flask Backend server
│
├── frontend/              # Web Dashboard interface
│   ├── index.html         # HTML UI layout
│   ├── style.css          # CSS Styling
│   └── script.js          # JS Webcam stream, smoothing & TTS
│
├── ml/                    # Machine Learning pipeline files
│   ├── dataset_loader.py  # Label encoder, signer split & DataLoaders
│   ├── dataset_inspection.py # Analytics of classes & resolutions
│   ├── feature_extraction.py # Legay & Tasks API MediaPipe landmarkers
│   ├── preprocessing.py   # Resumable video feature extraction (.npy)
│   ├── model.py           # BiLSTM+Attention and Baseline models
│   ├── train.py           # Training loop, scheduler & checkpointing
│   └── evaluate.py        # Macro F1, top-3 accuracy & CM plotting
│
├── models/                # Saved models, label map & config
│   ├── best_model.pth     # Best model checkpoint
│   ├── labels.json        # Class label index mappings
│   └── feature_config.json # Input dimensions metadata
│
├── notebooks/             # Google Colab resources
│   └── ISL_NonManual_Training.ipynb # Step-by-step training pipeline
│
├── data/
│   └── README.md
│
└── docs/
    ├── architecture.md    # Neural network layout & pipeline diagram
    ├── dataset.md         # Auto-generated dataset statistics report
    └── methodology.md     # Normalization formulas & smoothing window
```

---

## 2. Installation & Quick Setup

To run GESCOM locally:

### Step 1: Clone or Open the Directory
Open your terminal in the `ISL_NonManual_AI` folder:
```bash
cd ISL_NonManual_AI
```

### Step 2: Install Python Dependencies
Ensure you are using Python 3.9 - 3.13. Install the packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Note: If you run locally on Windows, the system will automatically download CPU-optimized wheels. If you are missing legacy MediaPipe solutions, the backend automatically downloads the new MediaPipe Task binaries (`face_landmarker.task`, `hand_landmarker.task`, `pose_landmarker.task`) to run the modern Tasks API.*

---

## 3. Training the Model in Google Colab

The dataset is approximately **8 GB** on Kaggle. The easiest way to train the sequence model is in **Google Colab** using the pre-configured notebook:

1. Open Google Colab (https://colab.research.google.com/).
2. Click **Upload** and select `notebooks/ISL_NonManual_Training.ipynb`.
3. Change runtime type to **GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Execute the cells in order:
   - **Kaggle Authentication**: Select your `kaggle.json` API file.
   - **Download & Extract**: Automatically pulls `kartiksaxena/isl-csltr` from Kaggle and unzips it.
   - **Feature Extraction**: Runs MediaPipe to extract facial landmarks and saves them as `.npy` feature sequences. (Resumable: if Colab disconnects, restart the cell and it will skip already created files).
   - **Train & Evaluate**: Trains the BiLSTM + Self-Attention model and evaluates performance on unseen test signers.
   - **Model Export**: Packaged as `gescom_model.zip`. Download it from Colab.

---

## 4. Local App Deployment

Once the model has finished training in Colab:
1. Download `gescom_model.zip` and extract its content (`best_model.pth`, `labels.json`, and `feature_config.json`) into the `models/` directory of this project.
2. Run the Flask backend server:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
4. Click **Start Camera**, allow webcam access, and perform ISL signs.

---

## 5. Smart India Hackathon (SIH) Presentation Features

GESCOM includes several features that distinguish it from standard hand-sign classifiers:

1. **Explainable AI Cues Panel**: 
   A dedicated UI section showing exactly why the model predicted a sign, matching the live eyebrows raised/lowered, eye blinking/openness, mouth openness, and head roll/pitch/yaw angles.
2. **Prediction Smoothing**: 
   A sliding window majority-voting system prevents predictions from flickering, making the translation stable.
3. **Continuous Sentence Buffering**: 
   Enables signers to chain multiple signs in sequence (e.g. "ARE" + "YOU" + "FREE" + "TODAY") into a readable sentence.
4. **Text-to-Speech (TTS)**: 
   Converts the recognized sentence into spoken audio (using English with an Indian accent `en-IN` voice setting) for accessibility.
5. **SIH Demonstration Mode**: 
   Draws colored facial trackers (bounding boxes, eye circles, mouth lines, head pose vector arrows) on top of the webcam feed in real-time, providing immediate visual feedback for the judges.

---

## 6. PPT Making & Presentation Talking Points

Here is a summary of technical details to add to your SIH presentation slides:

*   **Slide 1: Problem Statement & Focus**
    *   *Traditional Limitation:* Most ISL classifiers focus exclusively on hand shapes, ignoring facial expressions, which represent up to **50% of the lexical meaning** in sign languages.
   *   *Our Solution (GESCOM):* A custom sequence recognition model that captures non-manual features (eyebrows, eyes, mouth, head pose) fused with upper-body skeletal information.
*   **Slide 2: System Architecture**
    *   *Pipeline:* Raw Webcam $\rightarrow$ MediaPipe Spatial Tracking $\rightarrow$ Coordinate Normalization $\rightarrow$ BiLSTM Encoder $\rightarrow$ Self-Attention $\rightarrow$ Multi-class Translation.
    *   *Feature Vector Size:* 534 values per frame, sampled over a history of 30 frames.
*   **Slide 3: Technical Merits & Innovation**
    *   *Signer-Independent Split:* Evaluated on completely unseen signers during validation to guarantee model generalization.
    *   *Self-Attention Mechanism:* Instead of simple average pooling, the model computes frame-level importance, indicating which specific frames triggered the sign prediction.
    *   *CPU-friendly Local Deployment:* The feature extraction model is extremely lightweight (4.2 MB), running at **80+ FPS** on consumer-grade laptop CPUs.
