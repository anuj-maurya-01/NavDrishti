# Methodology - GESCOM

Project GESCOM employs a rigorous, multi-stage scientific approach to capture and translate continuous Indian Sign Language (ISL) gestures, focusing specifically on non-manual cues.

---

## 1. Spatial Normalization (Invariance Space)

To ensure that variables such as camera distance, aspect ratio, signer height, and minor lateral body movements do not affect prediction accuracy, all extracted coordinates undergo nose-centered scale-invariant projection.

Given a coordinate vector for landmark $i$ at frame $t$:
\[p_i = (x_i, y_i, z_i)\]

1. **Centering**: Shift the coordinate space origin to the **nose tip** (Landmark 1):
   \[p'_{i} = p_i - p_{1}\]
   *Effect: After this operation, $p'_{1} = (0.0, 0.0, 0.0)$, meaning the origin is locked to the signer's face center.*

2. **Scaling**: Compute the Euclidean **inter-ocular distance** $d_{eye}$ between outer eye corners (Landmark 33 and Landmark 263):
   \[d_{eye} = \| p'_{263} - p'_{33} \|_2\]
   Scale the centered landmarks relative to the face size:
   \[p''_{i} = \frac{p'_i}{d_{eye}}\]
   *Effect: Normalizes coordinates to a face-relative scale, ensuring identical vector ranges whether the user sits 1 meter or 3 meters from the webcam.*

3. **Hand & Pose Normalization**:
   - Hand landmarks are centered on the **wrist** (Landmark 0) and scaled by $d_{eye}$.
   - Pose landmarks (shoulder, elbows, wrists) are centered on the **shoulder midpoint** ($p_{shoulder}$) and scaled by $d_{eye}$.
   - This ensures all feature coordinate scales remain consistent with the facial mesh features.

---

## 2. Temporal Alignment (Uniform Decimation)

Videos in the corpus contain between $70$ and $186$ frames. To train a constant-tensor recurrent model, all frames are resampled to exactly $30$ timesteps (`SEQUENCE_LENGTH = 30`).

Given a video with $N$ total frames, we sample frames at indices $I$:
\[I_j = \text{round}\left( j \cdot \frac{N - 1}{29} \right) \quad \text{for } j \in [0, 29]\]
If frame decoding fails or the face is briefly occluded at index $I_j$, the system carries forward the last valid face frame features (forward-fill) to prevent zero-padding shocks inside the BiLSTM sequence memory.

---

## 3. Generalization Splitting (Signer-Independent)

Evaluating model accuracy by randomly splitting frames or individual videos leads to severe **data leakage** because the same signer appears in both the train and test sets, allowing the network to memorize individual facial shapes rather than lexical expressions.

To prove genuine model utility to the SIH judges, we split by **Disjoint Signer Groups**:
- **Train Split**: Signers 1, 2, 3, and 4.
- **Validation Split**: Signer 5.
- **Test Split**: Signer 6 (and Signer 7 if available).

Evaluating on Signer 6 tests the model's accuracy on a completely unseen user, demonstrating its real-world generalization performance.

---

## 4. Prediction Smoothing & Temporal Committing

To prevent text flickering in live webcam streams, we implement a sliding window decision tree:

```
[ Predicted Gloss Stream ] 
       │
       ▼
  ( Buffer 8 ) ── Store the last 8 predictions in a sliding window
       │
       ▼
 [ Majority Vote ] ── Identify the gloss with the highest frequency (f_max)
       │
       ▼
 f_max >= 4 ? ── Yes ──► confidence >= 0.60 ? ── Yes ──► Same as last committed?
       │                                                      │
       No                                                     No ──► COMMIT WORD!
       │                                                      │
       ▼                                                      ▼
  [ Ignore ]                                             [ Ignore ]
```

1. **Sliding Window**: A frontend buffer keeps track of the last 8 frame predictions (`SMOOTHING_WINDOW = 8`).
2. **Majority Voting**: The class with the highest frequency is selected.
3. **Commit Threshold**: The word is committed to the final sentence only if:
   - It is stable for $4$ consecutive window updates (`STABILITY_THRESHOLD = 4`).
   - The model's prediction confidence is $\ge 60\%$ (`CONF_THRESHOLD = 0.60`).
   - It differs from the last word in the committed sentence (preventing duplicate stuttering).

---

## 5. Web Speech Synthesis

Once a word is committed, it is appended to the sentence buffer and sent to the browser's **SpeechSynthesis API** for immediate text-to-speech feedback:
- Runs client-side (no network latency or server load).
- Voice locale set to `en-IN` (English with Indian accent) to ensure natural pronunciation for Indian Sign Language translations in the hackathon context.
