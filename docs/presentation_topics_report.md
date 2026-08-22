# GESCOM Project Knowledge and Presentation Report

## 1. Project Overview

GESCOM is a real-time Indian Sign Language (ISL) recognition system designed to translate sign gestures into readable text. The project is focused on capturing non-manual features such as facial expression, eyebrow movement, eye motion, head pose, mouth shape, and upper-body posture, along with hand gestures. The final system takes webcam input, extracts meaningful motion features, and predicts the corresponding ISL sentence class.

The main objective is to build a practical AI-based communication system that can help Deaf and hard-of-hearing people by converting sign language into text and speech. This makes the project socially relevant and technically meaningful.

---

## 2. Problem Statement

Most traditional sign-language recognition systems focus only on hand shapes and hand movement. That is an incomplete solution for Indian Sign Language because facial and body expressions carry crucial meaning.

For example:

- two signs may use similar hand gestures
- head pose may change the sentence meaning
- eye and mouth movement can add emphasis or expression
- eyebrows and facial muscles may indicate emotion or grammar

Therefore, the project argues that sign language understanding should not rely only on hand data. A better system must consider both manual and non-manual features.

The project addresses this by combining:

- face landmarks
- hand landmarks
- pose landmarks
- temporal sequence learning

---

## 3. Why This Project Matters

This project matters because:

- ISL is a visual communication system used by a large community in India
- sign language recognition can improve accessibility
- communication gaps still exist between Deaf communities and non-signers
- such a system can support education, public services, and healthcare settings
- real-time sign translation is a challenging but highly valuable AI problem

This is not only a model-building project; it is a complete AI application with real-world impact.

---

## 4. Domain Knowledge You Must Understand

### 4.1 Indian Sign Language (ISL)

ISL is not a direct translation of spoken English. It has its own grammar, context, expression patterns, and sentence construction. A sign can depend on:

- hand orientation
- finger position
- movement path
- facial expression
- head movement
- body posture

This means that recognition must work beyond single static poses. It must understand temporal patterns and context.

### 4.2 Non-Manual Features

These are features outside the hands that carry important meaning.

Examples:

- eyebrow movement
- eye openness or blinking
- mouth movement
- head orientation and tilt
- nose and facial muscle movement
- upper-body posture

These features often help distinguish text meanings that could otherwise be ambiguous.

### 4.3 Manual Features

Manual features refer to hand-related cues, such as:

- finger position
- palm orientation
- motion trajectory
- hand shape

The project combines manual features with non-manual features to build a richer representation of sign language.

---

## 5. Dataset and Corpus

The project uses the local dataset stored at:

- E:\anuj\ISL_CSLRT_Corpus

This corpus includes sentence-level ISL samples and is used for training a sequence-based recognition model. The report and project pipeline indicate that:

- around 492 labeled feature samples were used in the extraction pipeline
- around 101 sentence classes were involved
- videos are converted into sequence data for temporal learning
- each sample is represented by a sequence of frames

Important point:

The dataset is not just static images. It is a video sequence dataset, which is essential because gesture recognition depends on temporal motion.

---

## 6. Data Preprocessing Pipeline

Raw videos are processed before training. This step is critical because the model must learn patterns from structured landmarks instead of noisy raw video frames.

The preprocessing pipeline includes:

- extracting frames from the video
- detecting hand, face, and body landmarks
- normalizing landmark positions
- removing inconsistent or low-quality frames
- converting the sequence into a time-based feature matrix

This process turns unstructured video content into a consistent format that the model can learn from.

---

## 7. MediaPipe and Landmark Detection

The project uses MediaPipe, a widely used computer-vision framework for landmark extraction.

The system detects:

- face landmarks
- left-hand landmarks
- right-hand landmarks
- pose landmarks

This is highly important because it gives structured coordinates for the full body and face rather than using full-image pixels. Landmark-based data is more compact, interpretable, and efficient for real-time recognition.

MediaPipe helps in:

- extracting geometry from the face and hands
- tracking motion across time
- reducing dependence on raw image pixels
- making the system lighter and faster

---

## 8. Feature Extraction Details

For each frame, the system extracts a feature vector. The report states that the feature vector size is approximately 534 dimensions per frame.

The feature breakdown is:

- facial and head cues: 18
- selected face landmarks: 354
- left-hand landmarks: 63
- right-hand landmarks: 63
- pose landmarks: 36

Total = 534 features per frame.

The temporal model does not process only one frame. It receives a sequence of 30 frames for each sample, so the input shape becomes:

- 30 frames × 534 features

This is the core idea behind the model: learning motion across time, not just static posture.

---

## 9. Feature Normalization

Normalization is a very important part of this project.

Why is it necessary?

- different people have different heights
- camera distance changes
- body position may change across videos
- face and hand location may shift inside the frame
- lighting and orientation may vary

The system reduces these variations by normalizing coordinates relative to reference points such as:

- nose
- eyes
- wrists
- shoulders

This makes the model less sensitive to camera variation and signer position, improving real-world stability.

---

## 10. Temporal Modeling Concept

Sign language is a temporal process. The meaning depends on how gestures evolve over a sequence of frames.

For example:

- hand movement direction matters
- facial expression may appear at a specific moment
- sequence speed or timing matters
- transitions between poses matter more than the pose itself

A static image model would lose this temporal information. That is why the project uses a sequence-learning model instead of a simple image classifier.

---

## 11. Model Architecture: BiLSTM + Attention

The main model is a Bidirectional Long Short-Term Memory (BiLSTM) network with an attention layer.

### 11.1 BiLSTM

A BiLSTM processes the sequence in both directions:

- forward direction: past to future
- backward direction: future to past

This helps the model capture motion dependencies from both sides of the sequence.

### 11.2 Attention Layer

The attention mechanism helps the model decide which frames are more important for the final recognition.

This is helpful because:

- not all frames carry equal meaning
- some frames represent the start or peak of the sign
- some frames are noisy or transitional

Attention allows the model to focus on the most relevant frames instead of treating all frames equally.

### 11.3 Final Classification Layer

After temporal features are processed, the model sends the information to a dense classification layer. This predicts one output label from the known list of ISL sentence classes.

Therefore, the task is a multiclass classification problem.

---

## 12. Why Not Use Regression?

Regression is not appropriate for this project because the system is not predicting a continuous number.

It is predicting a class label such as:

- I am hungry
- How are you?
- Thank you
- Do not hurt me

This means the system predicts among a fixed set of classes. That is classification, not regression.

---

## 13. Training and Evaluation Setup

The project trains the model using the processed feature sequences. Model evaluation includes metrics such as:

- accuracy
- top-3 accuracy
- macro F1-score

These metrics help measure how well the classifier works across multiple classes.

The project also compares the main model against a baseline classical approach called XGBoost.

---

## 14. XGBoost Comparison

XGBoost was tested as a benchmark model.

It used summarized temporal statistics such as:

- mean
- standard deviation
- minimum
- maximum
- movement-related features

This produced a larger fixed-length feature set for traditional machine learning. However, the deep-learning sequence model performed better on the available data.

This comparison is useful because it shows that the chosen temporal BiLSTM is not just a default choice but the better-performing approach for this problem.

---

## 15. Verified Results

The project report states that the main model achieved better results than XGBoost on the available dataset.

Approximate results:

- BiLSTM + Attention: around 40.24% top-1 accuracy
- XGBoost: around 27.85% top-1 accuracy

Top-3 and Macro F1 values also support the same conclusion. This proves that the sequence-based model is a stronger solution for temporal gesture recognition.

However, the report also clearly states that the current evaluation is not a fully independent signer-independent test and that the dataset size is still limited.

---

## 16. Live Webcam Workflow

The project is designed to run in real time from a webcam. The app flow is:

1. webcam captures a frame
2. the frame is sent to the backend
3. MediaPipe extracts landmarks
4. feature vectors are computed and normalized
5. the system keeps a 30-frame buffer
6. the BiLSTM predicts the current class
7. the prediction is smoothed to avoid flickering
8. the final output is displayed on the screen
9. optional speech output is generated

This is the complete live inference pipeline.

---

## 17. Prediction Smoothing

In live webcam use, models often flicker between different labels. This happens because consecutive frames may produce slightly different predictions.

To solve that, the system uses prediction smoothing, which helps by:

- reducing rapid label changes
- making output more stable
- improving the readability of the translation
- creating a better demonstration experience

This is a very important practical design choice for a real-time recognition app.

---

## 18. Text-to-Speech (TTS)

The system is also capable of speaking the recognized output. This is useful for:

- accessibility
- live demonstration
- validation of output by hearing the predicted sentence
- user experience improvement

This adds an end-user-friendly layer that makes the project more complete.

---

## 19. Technology Stack

The project uses a standard but effective stack:

- Python for model logic and backend
- OpenCV for video processing
- MediaPipe for landmark extraction
- PyTorch for deep learning
- Flask for the server and API layer
- JavaScript, HTML, and CSS for the frontend UI

This gives a complete AI application pipeline from webcam camera input to final text output.

---

## 20. Main Project Files

The most important project files are:

- config.py: stores parameters such as sequence length, input size, thresholds, and file paths
- app.py: backend Flask server for live inference
- frontend/script.js: webcam capture and UI logic
- ml/feature_extraction.py: landmark extraction and feature creation
- ml/preprocessing.py: video-to-feature conversion
- ml/dataset_loader.py: dataset arrangement and loading logic
- ml/model.py: BiLSTM and other model definitions
- ml/train.py: training script
- ml/evaluate.py: evaluation script
- ml/xgboost_model.py: benchmark model

These files collectively define the full machine-learning pipeline.

---

## 21. Challenges and Limitations

The project is strong, but it also has real limitations. In a presentation, these should be mentioned honestly.

Main limitations:

- dataset size is relatively small for 101 classes
- there are only a few training examples per class
- lighting conditions affect accuracy
- camera angle and distance affect recognition
- hand visibility may be poor in some cases
- different signers create variation in style and speed
- similar signs may get confused

This shows the project is realistic and not oversold.

---

## 22. Future Improvements

The project has clear future directions:

- collect a larger and more diverse dataset
- perform signer-independent validation
- use more variation in background, lighting, and camera setup
- improve temporal smoothing
- test transformer-based models
- use 3D pose estimation or multi-camera setups
- deploy the system more robustly for real-world use

These improvements would make the model more practical and generalizable.

---

## 23. Presentation-Ready Summary

A strong one-minute explanation of the project is:

> GESCOM is a real-time Indian Sign Language recognition system that uses webcam input, MediaPipe landmark extraction, and a temporal BiLSTM with attention to identify ISL gestures. The model learns from sequences of normalized facial, hand, and pose features over 30 frames, enabling it to classify sentence-level signs in real time. The system is built with Python, PyTorch, Flask, and a browser-based frontend, and it includes prediction smoothing and text-to-speech for improved usability and accessibility.

---

## 24. Key Technical Terms to Remember

These are the terms you should know for the presentation or viva:

- Indian Sign Language (ISL)
- Non-manual features
- Manual features
- MediaPipe
- Landmark extraction
- Feature normalization
- Temporal sequence modeling
- BiLSTM
- Attention mechanism
- Multiclass classification
- Sequence length
- Webcam inference
- Prediction smoothing
- Text-to-speech
- Flask deployment
- Evaluation metrics

---

## 25. Best Slide Structure for Presentation

### Slide 1: Title and Motivation
- What is ISL?
- Why is it important?
- What problem does the project solve?

### Slide 2: Proposed Solution
- Webcam-based sign recognition
- Use of face, hand, and pose features
- Real-time classification system

### Slide 3: Dataset and Feature Extraction
- ISL_CSLRT_Corpus
- 30-frame temporal sequence
- 534 features per frame

### Slide 4: Model Architecture
- BiLSTM + Attention
- Why sequence modeling is needed
- Why non-manual features matter

### Slide 5: Real-Time App Workflow
- camera input
- landmark extraction
- buffer creation
- prediction
- final text output

### Slide 6: Results and Evaluation
- accuracy, top-3, macro F1
- comparison with XGBoost
- strengths and improvements

### Slide 7: Challenges and Future Work
- dataset limitations
- generalization issues
- future improvements and deployment potential

---

## 26. Final Takeaway

GESCOM is a meaningful AI project because it combines computer vision, temporal learning, and accessibility. It does not just detect hand gestures; it tries to understand the full visual language of sign communication by considering face, body, and motion over time.

This makes the project both technically challenging and socially useful. It is a strong example of an applied AI system that connects real-world communication needs with machine learning research.

---

## 27. Short Viva / Interview Answer

If someone asks, “What is your project about?” you can answer:

> This project is an Indian Sign Language recognition system that uses webcam-based live input and MediaPipe landmark extraction to analyze facial, hand, and upper-body movements. The extracted features are fed into a BiLSTM with attention model, which learns the temporal pattern of the sign and predicts the correct sentence label. The system outputs text and can also convert it into speech, making it useful for accessibility and communication support.

This is the cleanest and most complete explanation to remember.

---

## 28. Topics You Must Learn for This Project

These are the exact topics you should study before giving the presentation or viva.

### 28.1 Computer Vision
- image and video processing
- frames from video
- feature extraction from images
- landmark detection
- pose and face tracking

### 28.2 Indian Sign Language (ISL)
- meaning of sign language
- manual vs non-manual features
- difference between static and dynamic gestures
- sentence-level sign recognition

### 28.3 MediaPipe
- face landmarks
- hand landmarks
- pose landmarks
- landmark-based representation
- why MediaPipe is useful in real-time applications

### 28.4 Feature Engineering
- feature vector creation
- normalized body coordinates
- face, hand, and pose features
- sequence-based feature representation
- why normalization is important

### 28.5 Temporal Sequence Learning
- time-series data in sign language
- sequence length
- relation between successive frames
- understanding motion over time

### 28.6 LSTM and BiLSTM
- what is RNN
- what is LSTM
- why BiLSTM is better for sequential gesture data
- learning past and future context

### 28.7 Attention Mechanism
- why some frames matter more than others
- attention in sequence classification
- importance of focus on relevant sign frames

### 28.8 Deep Learning Basics
- neural network structure
- optimizer
- loss function
- training loop
- validation and testing

### 28.9 Classification Concepts
- multiclass classification
- labels and class mapping
- confusion between similar signs
- model output interpretation

### 28.10 Evaluation Metrics
- accuracy
- top-3 accuracy
- macro F1-score
- confusion matrix
- model comparison

### 28.11 Real-Time Inference
- webcam streaming
- frame capture
- buffering 30 frames
- live prediction
- flicker reduction through smoothing

### 28.12 Flask and Web App Integration
- backend server
- API endpoints
- sending input from frontend to backend
- receiving outputs and displaying results

### 28.13 Frontend Basics
- webcam access using JavaScript
- capturing camera frames
- handling live UI updates
- displaying prediction text

### 28.14 Text-to-Speech (TTS)
- why TTS is useful
- converting recognized text into speech
- accessibility use case

### 28.15 Data Preprocessing and Cleaning
- frame extraction
- data normalization
- removing poor-quality samples
- preparing data for training

### 28.16 Model Comparison and Benchmarking
- BiLSTM vs XGBoost
- why benchmarking is important
- choosing the better model

### 28.17 Challenges and Limitations
- small dataset
- lighting changes
- signer variation
- similar-looking signs
- real-world robustness

These are the most essential topics to learn, understand, and explain in your project presentation.
