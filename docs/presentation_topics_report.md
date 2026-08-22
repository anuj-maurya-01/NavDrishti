# Presentation Topics and Project Knowledge Report

## 1. Project Overview

GESCOM is an Indian Sign Language (ISL) recognition system that translates non-manual facial and body movements into text. The project focuses on live webcam-based recognition of ISL sentence classes using a sequence model.

The main goal is to capture:

- facial expressions
- eye and mouth movements
- head pose
- hand gestures
- upper-body pose

and convert them into meaningful text output in real time.

---

## 2. Topics You Should Know for the Presentation

### 2.1 Problem Statement

The project addresses a real problem: most sign language recognition systems focus only on hand gestures, but in ISL, non-manual features also carry important meaning.

Examples of non-manual features:

- eyebrow position
- eye openness or blinking
- mouth shape
- head movement
- facial expression
- body posture

These features are important because they help distinguish signs that look similar when judged only by hand movement.

### 2.2 Indian Sign Language (ISL)

Need to explain:

- ISL is a visual language used by Deaf and hard-of-hearing communities in India.
- It is not just a direct translation of spoken English.
- Sign language depends on both manual and non-manual features.
- Sentence recognition is more complex than single-word classification.

### 2.3 Why Non-Manual Features Matter

This is one of the most important presentation points.

Traditional models often classify signs using only hand shapes and gestures. But in real sign communication:

- expressions can change meaning
- mouth movement can indicate emphasis or phrase structure
- head pose can indicate questions, affirmation, or emotion
- eye gaze and eyebrow motion help convey intent

So, the project includes these features to improve recognition quality.

### 2.4 Dataset

The dataset used is the ISL_CSLRT_Corpus, which contains sentence-level ISL samples.

Important dataset points:

- local corpus in the project directory
- multiple classes of ISL sentence labels
- video-based recordings
- frames processed into landmark-based features
- sequence data used for temporal learning

Presentation note:

> The project is trained on a sentence-level ISL corpus and not only on static hand images.

### 2.5 Data Preprocessing

Before training, raw videos are processed to extract useful features.

This step includes:

- frame extraction from videos
- resizing and normalization
- detection of landmarks
- cleanup of invalid or low-quality frames
- conversion of video sequences into feature arrays

This helps the model learn meaningful motion patterns instead of raw pixel data.

### 2.6 MediaPipe Landmark Detection

One of the most important technologies used in the project is MediaPipe.

The project uses MediaPipe to detect:

- face landmarks
- hand landmarks
- pose landmarks

This gives coordinates for different parts of the face and body in each frame.

Why it matters:

- it reduces dependence on raw image pixels
- it captures geometry and motion more robustly
- it is lightweight and suitable for real-time applications

### 2.7 Feature Extraction

For every frame, the system extracts a feature vector.

In this project, the per-frame vector size is around 534 dimensions.

This includes:

- facial and head cues
- selected face landmark coordinates
- normalized left-hand landmarks
- normalized right-hand landmarks
- normalized pose landmarks

The model uses a sequence of 30 frames, therefore each input sample is shaped like:

- 30 frames x 534 features

This means the model learns time-dependent motion, not only static position.

### 2.8 Normalization

Normalization is critical in gesture recognition.

The system normalizes features relative to reference points such as:

- nose
- eyes
- wrists
- shoulders

This helps reduce error due to:

- changing camera distance
- different body positions
- variable signer height
- small lateral shifts in the frame

This makes the model better at handling real-world webcam input.

### 2.9 Temporal Sequence Modeling

Sign language is dynamic. The meaning depends on how movements change over time.

So the project uses sequence learning instead of single-frame classification.

The model processes a sequence of frames and learns:

- motion over time
- transition between signs
- temporal dependencies

This is why a recurrent model is used instead of a simple traditional image classifier.

### 2.10 BiLSTM (Bidirectional Long Short-Term Memory)

BiLSTM is a strong model for sequential data.

Why used here:

- it understands previous and future context in a sequence
- it remembers long-term dependencies
- it is effective for gesture motion sequences

The model processes the 30-frame feature sequence and learns spatial-temporal patterns.

### 2.11 Attention Mechanism

The project uses an attention layer after the BiLSTM.

This helps the model identify which frames are more important for the final result.

Why this matters:

- not all frames carry equal meaning
- some frames show the action start or peak movement
- attention improves decision-making

In presentation, explain attention as:

> The model learns which moments in the sequence matter the most for recognizing the sign.

### 2.12 Classification Layer

After feature learning, the model passes the information to a dense final layer.

This layer predicts one of the sign classes.

The final problem is a multiclass classification task, because the system predicts one label from many possible ISL sentence classes.

### 2.13 Live Webcam Pipeline

The project is not only a trained model; it is a full app.

Live flow:

1. Webcam captures frames
2. Frames are sent to the backend
3. MediaPipe extracts landmarks
4. Features are normalized
5. 30-frame buffer is created
6. Sequence model predicts the class
7. Prediction smoothing is applied
8. Final output is shown to the user

This makes it a real end-to-end AI application.

### 2.14 Prediction Smoothing

A common issue in live recognition is flickering: the model may change prediction often between similar classes.

The project uses smoothing to reduce this effect.

This helps by:

- reducing unstable intermediate predictions
- making output look consistent
- improving user experience in real-time translation

### 2.15 Text-to-Speech (TTS)

The system can speak the recognized output.

This is useful for accessibility and demonstration.

It helps users hear the translated sentence and validates the result in real time.

### 2.16 Evaluation Metrics

You should know the standard metrics used to evaluate the model.

Important ones:

- Accuracy
- Top-3 accuracy
- Macro F1-score

Why these metrics matter:

- accuracy shows overall correctness
- top-3 gives a softer measure when similar classes exist
- macro F1 is useful in multiclass settings because it balances minority and majority classes

### 2.17 Model Comparison

The project also compared the main temporal model with XGBoost.

This is useful as a benchmark:

- BiLSTM + Attention performed better on sequence data
- XGBoost was tested on summarized temporal features
- the temporal deep learning model was selected as the live system model

This shows the project is not only using a model but validating it against another approach.

### 2.18 Challenges and Limitations

This is important for a presentation because it shows awareness of real-world issues.

Main challenges:

- limited dataset size
- many classes but few samples per class
- variations in lighting
- camera angle and distance problems
- hand visibility problems
- differences between signers
- chance of confusion between similar gestures

This makes the project realistic and demonstrates engineering maturity.

### 2.19 Deployment Architecture

The deployed project uses:

- Python for model logic and backend
- OpenCV for video processing
- MediaPipe for landmark extraction
- PyTorch for deep learning
- Flask for the backend server
- JavaScript + HTML + CSS for the frontend UI

This is a complete AI product pipeline from webcam input to prediction output.

### 2.20 Future Improvements

In a presentation, mention what can be improved next.

Examples:

- larger and more diverse ISL dataset
- signer-independent evaluation
- more variation in lighting, background, and distance
- better temporal smoothing
- use of transformer-based models
- multi-camera or 3D pose estimation
- deployment with a stronger production pipeline

---

## 3. Important Presentation-Friendly Summary

A strong 1-minute explanation for your presentation could be:

> GESCOM is a real-time Indian Sign Language recognition system that uses webcam input, MediaPipe landmark extraction, and a temporal BiLSTM with attention to detect non-manual facial and body features. The model learns motion over 30-frame sequences, normalizes body points for consistency, and predicts ISL sentence classes in real time. The system is implemented using Python, MediaPipe, PyTorch, Flask, and a browser-based frontend, and it also includes prediction smoothing and text-to-speech for user-friendly output.

---

## 4. Key Technical Terms to Remember

- Indian Sign Language (ISL)
- Non-manual features
- MediaPipe Holistic
- Landmark extraction
- Feature normalization
- Sequence modeling
- BiLSTM
- Attention mechanism
- Multiclass classification
- Temporal buffer
- Webcam inference
- Prediction smoothing
- Evaluation metrics
- Text-to-speech
- Flask deployment

---

## 5. Presentation Structure You Can Follow

### Slide 1: Title and Problem
- Introduction to ISL recognition
- Importance of non-manual features

### Slide 2: Proposed Solution
- Webcam-based real-time system
- MediaPipe + sequence model pipeline

### Slide 3: Dataset and Features
- ISL corpus
- 534-dimensional frame features
- 30-frame sequence input

### Slide 4: Model Architecture
- BiLSTM + attention
- Classification head
- Why this is suitable for temporal gestures

### Slide 5: Web App Workflow
- Camera input
- extraction
- prediction
- output and TTS

### Slide 6: Results and Evaluation
- metrics
- model comparison
- strengths and limitations

### Slide 7: Challenges and Future Scope
- dataset limitations
- real-world robustness
- future improvements

---

## 6. Final Takeaway

This project is valuable because it combines computer vision, sequence modeling, and real-time deployment into a practical sign-language translation system. It is especially strong because it captures both manual and non-manual gestures instead of relying only on hand movements.

That makes the project relevant, technically sound, and presentation-ready for academic or hackathon evaluation.
