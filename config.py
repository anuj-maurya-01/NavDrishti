import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if running in Google Colab
IS_COLAB = os.path.exists("/content")

if IS_COLAB:
    DATASET_PATH = "/content/ISL_CSLRT_Corpus"
    FEATURES_DIR = "/content/features"
    MODELS_DIR = "/content/models"
else:
    # Local fallback
    DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "ISL_CSLRT_Corpus"))
    FEATURES_DIR = os.path.abspath(os.path.join(BASE_DIR, "data", "features"))
    MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "models"))

# Data structure files
EXCEL_DETAILS_PATH = os.path.join(DATASET_PATH, "corpus_csv_files", "ISL_CSLRT_Corpus details.xlsx")
FRAME_DETAILS_PATH = os.path.join(DATASET_PATH, "corpus_csv_files", "ISL_CSLRT_Corpus_frame_details.xlsx")
WORD_DETAILS_PATH = os.path.join(DATASET_PATH, "corpus_csv_files", "ISL_CSLRT_Corpus_word_details.xlsx")
GLOSS_CSV_PATH = os.path.join(DATASET_PATH, "corpus_csv_files", "ISL Corpus sign glosses.csv")

# Make sure output directories exist
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Temporal processing configuration
SEQUENCE_LENGTH = 30  # Number of frames per temporal sequence
FRAME_SAMPLING_MODE = "uniform"  # "uniform" or "interpolate" or "skip"

# Model Hyperparameters
NUM_CLASSES = 101  # There are 101 sentences in the dataset
INPUT_DIM = 534    # Feature dimension per frame (18 custom features + 354 face + 63 L hand + 63 R hand + 36 pose)
HIDDEN_DIM = 128
LSTM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 1e-3

# Real-time Web Application Configuration
CONFIDENCE_THRESHOLD = 0.60
STABILITY_THRESHOLD = 4   # Minimum consecutive frames predicted as the same sign to trigger update
SMOOTHING_WINDOW = 8       # Window size for prediction history majority vote

# Landmark config indices (MediaPipe Face Mesh)
# Selected key face landmarks (120 total) to save memory and avoid overfitting
SELECTED_FACE_INDICES = [
    # Eyebrows (left: 70,63,105,66,107,55,65,52,53,46 | right: 300,293,334,296,336,285,295,282,283,276)
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    # Eyes (left contour)
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # Eyes (right contour)
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # Mouth (outer lips)
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 95,
    # Mouth (inner lips)
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    # Nose & Bridge
    1, 2, 98, 327, 4, 5, 6, 197, 195, 168,
    # Jaw / Chin outline (subset)
    0, 17, 18, 200, 152, 377, 400, 396, 175, 171, 148, 136, 150, 176, 140, 142
]

# Pose landmark keypoints (shoulders, elbows, wrists, hands)
POSE_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
