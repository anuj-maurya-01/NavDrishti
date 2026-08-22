# Data Directory

This directory is used for caching extracted feature files.

## Caching
- **Extracted Features Path**: `data/features/*.npy`
- Large video files from the 8 GB Kaggle dataset (`ISL_CSLRT_Corpus`) should be stored outside the repository directory or kept locally.
- Extracted features are represented as aligned `(30, 534)` float32 numpy arrays, which are extremely lightweight (approx. 64 KB per video) and are used directly during training.
