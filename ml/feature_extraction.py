import cv2
import numpy as np
import math
import sys
import os
import urllib.request

# Ensure config is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# Attempt to import legacy solutions, detect if we need the Tasks API fallback
try:
    import mediapipe as mp
    # Test if solutions is available
    _ = mp.solutions.holistic
    USE_LEGACY_API = True
except (AttributeError, ImportError, ModuleNotFoundError):
    USE_LEGACY_API = False
    print("[Info] MediaPipe legacy solutions API not available. Using Tasks API fallback.")

class FeatureExtractor:
    def __init__(self, static_image_mode=False):
        self.use_legacy = USE_LEGACY_API
        
        # Standard 3D head model points for pose estimation (solvePnP)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye outer corner
            (225.0, 170.0, -135.0),      # Right eye outer corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)

        if self.use_legacy:
            import mediapipe as mp
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=static_image_mode,
                model_complexity=1,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            # Initialize Tasks API models
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Setup paths for .task files
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            self.face_task_path = os.path.join(config.MODELS_DIR, "face_landmarker.task")
            self.hand_task_path = os.path.join(config.MODELS_DIR, "hand_landmarker.task")
            self.pose_task_path = os.path.join(config.MODELS_DIR, "pose_landmarker.task")
            
            # URLs for downloading if missing
            self._download_task_files_if_missing()
            
            # Initialize Landmarkers
            base_face = python.BaseOptions(model_asset_path=self.face_task_path)
            options_face = vision.FaceLandmarkerOptions(
                base_options=base_face,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)
            
            base_hand = python.BaseOptions(model_asset_path=self.hand_task_path)
            options_hand = vision.HandLandmarkerOptions(
                base_options=base_hand,
                num_hands=2
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
            
            base_pose = python.BaseOptions(model_asset_path=self.pose_task_path)
            options_pose = vision.PoseLandmarkerOptions(
                base_options=base_pose,
                output_segmentation_masks=False
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

    def _download_task_files_if_missing(self):
        downloads = [
            ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", self.face_task_path),
            ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", self.hand_task_path),
            ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", self.pose_task_path)
        ]
        for url, dest in downloads:
            if not os.path.exists(dest):
                print(f"Downloading MediaPipe Task file: {os.path.basename(dest)} ...")
                try:
                    urllib.request.urlretrieve(url, dest)
                    print(f"Successfully downloaded {os.path.basename(dest)}")
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                    raise e

    def close(self):
        if self.use_legacy:
            self.holistic.close()
        else:
            self.face_landmarker.close()
            self.hand_landmarker.close()
            self.pose_landmarker.close()

    def _calculate_ear(self, landmarks, eye_indices):
        p1 = np.array(landmarks[eye_indices[0]])
        p2 = np.array(landmarks[eye_indices[4]])
        p3 = np.array(landmarks[eye_indices[5]])
        p4 = np.array(landmarks[eye_indices[8]])
        p5 = np.array(landmarks[eye_indices[12]])
        p6 = np.array(landmarks[eye_indices[13]])
        
        d_vert1 = np.linalg.norm(p2 - p6)
        d_vert2 = np.linalg.norm(p3 - p5)
        d_horiz = np.linalg.norm(p1 - p4)
        
        if d_horiz < 1e-6:
            return 0.0
        return (d_vert1 + d_vert2) / (2.0 * d_horiz)

    def _estimate_head_pose(self, landmarks_3d, width, height):
        pts_idx = [1, 152, 33, 263, 61, 291]
        image_points = np.array([
            [landmarks_3d[i][0] * width, landmarks_3d[i][1] * height] for i in pts_idx
        ], dtype=np.float32)
        
        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
            
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0.0
            
        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    def _extract_cues_and_hc_feats(self, face_lms, inter_ocular_dist, width, height):
        left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        ear_l = self._calculate_ear(face_lms, left_eye_idx)
        ear_r = self._calculate_ear(face_lms, right_eye_idx)
        ear_avg = (ear_l + ear_r) / 2.0
        
        eb_raise_l = np.linalg.norm(face_lms[55] - face_lms[159]) / (inter_ocular_dist + 1e-6)
        eb_raise_r = np.linalg.norm(face_lms[285] - face_lms[386]) / (inter_ocular_dist + 1e-6)
        eb_dist = np.linalg.norm(face_lms[55] - face_lms[285]) / (inter_ocular_dist + 1e-6)
        eb_symmetry = abs(eb_raise_l - eb_raise_r)
        
        lip_width = np.linalg.norm(face_lms[61] - face_lms[291]) / (inter_ocular_dist + 1e-6)
        lip_height_inner = np.linalg.norm(face_lms[13] - face_lms[14]) / (inter_ocular_dist + 1e-6)
        lip_height_outer = np.linalg.norm(face_lms[0] - face_lms[17]) / (inter_ocular_dist + 1e-6)
        mar = lip_height_outer / (lip_width + 1e-6)
        
        left_eye_center = np.mean(face_lms[left_eye_idx], axis=0)
        right_eye_center = np.mean(face_lms[right_eye_idx], axis=0)
        
        gaze_l_x = 0.0
        gaze_l_y = 0.0
        gaze_r_x = 0.0
        gaze_r_y = 0.0
        
        if len(face_lms) >= 478:
            gaze_l_x = (face_lms[468][0] - left_eye_center[0]) / (inter_ocular_dist + 1e-6)
            gaze_l_y = (face_lms[468][1] - left_eye_center[1]) / (inter_ocular_dist + 1e-6)
            gaze_r_x = (face_lms[473][0] - right_eye_center[0]) / (inter_ocular_dist + 1e-6)
            gaze_r_y = (face_lms[473][1] - right_eye_center[1]) / (inter_ocular_dist + 1e-6)
            
        pitch, yaw, roll = self._estimate_head_pose(face_lms, width, height)
        
        cues = {
            "eyebrow_left": "normal",
            "eyebrow_right": "normal",
            "eyes": "open",
            "mouth": "closed",
            "head": "center",
            "confidence_cues": {
                "ear": float(ear_avg),
                "mar": float(mar),
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll),
                "eyebrow_distance": float(eb_dist)
            }
        }
        
        if eb_raise_l > 0.40:
            cues["eyebrow_left"] = "raised"
        elif eb_raise_l < 0.28:
            cues["eyebrow_left"] = "lowered/frowning"
            
        if eb_raise_r > 0.40:
            cues["eyebrow_right"] = "raised"
        elif eb_raise_r < 0.28:
            cues["eyebrow_right"] = "lowered/frowning"
            
        if ear_avg < 0.16:
            cues["eyes"] = "blink/closed"
        elif ear_avg < 0.24:
            cues["eyes"] = "half-open"
        else:
            cues["eyes"] = "open"
            
        if lip_height_inner > 0.15:
            cues["mouth"] = "wide open"
        elif lip_height_inner > 0.05:
            cues["mouth"] = "open"
        else:
            cues["mouth"] = "closed"
            
        head_orientations = []
        if pitch > 12.0:
            head_orientations.append("down")
        elif pitch < -12.0:
            head_orientations.append("up")
            
        if yaw > 12.0:
            head_orientations.append("right")
        elif yaw < -12.0:
            head_orientations.append("left")
            
        if roll > 8.0:
            head_orientations.append("tilt-left")
        elif roll < -8.0:
            head_orientations.append("tilt-right")
            
        if head_orientations:
            cues["head"] = "-".join(head_orientations)
        else:
            cues["head"] = "center"
            
        hc_feats = [
            eb_raise_l, eb_raise_r, eb_dist, eb_symmetry,
            ear_l, ear_r, ear_avg,
            gaze_l_x, gaze_l_y, gaze_r_x, gaze_r_y,
            lip_width, lip_height_inner, lip_height_outer, mar,
            pitch / 90.0, yaw / 90.0, roll / 90.0
        ]
        return cues, hc_feats

    def extract_features(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Outputs
        features_flat = np.zeros(config.INPUT_DIM, dtype=np.float32)
        cues = {
            "eyebrow_left": "normal",
            "eyebrow_right": "normal",
            "eyes": "open",
            "mouth": "closed",
            "head": "center",
            "confidence_cues": {}
        }
        
        if self.use_legacy:
            results = self.holistic.process(frame_rgb)
            if not results.face_landmarks:
                return features_flat, cues
                
            face_lms = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
            nose_tip = face_lms[1]
            centered_face = face_lms - nose_tip
            inter_ocular_dist = np.linalg.norm(face_lms[33] - face_lms[263])
            
            if inter_ocular_dist > 1e-6:
                normalized_face = centered_face / inter_ocular_dist
            else:
                normalized_face = centered_face
                
            cues, hc_feats = self._extract_cues_and_hc_feats(face_lms, inter_ocular_dist, width, height)
            face_feats = normalized_face[config.SELECTED_FACE_INDICES].flatten()
            
            # Left hand
            lh_feats = np.zeros(63, dtype=np.float32)
            if results.left_hand_landmarks:
                wrist = results.left_hand_landmarks.landmark[0]
                lh_feats = np.array([[(lm.x - wrist.x) / (inter_ocular_dist + 1e-6),
                                      (lm.y - wrist.y) / (inter_ocular_dist + 1e-6),
                                      (lm.z - wrist.z) / (inter_ocular_dist + 1e-6)] 
                                     for lm in results.left_hand_landmarks.landmark]).flatten()
            # Right hand
            rh_feats = np.zeros(63, dtype=np.float32)
            if results.right_hand_landmarks:
                wrist = results.right_hand_landmarks.landmark[0]
                rh_feats = np.array([[(lm.x - wrist.x) / (inter_ocular_dist + 1e-6),
                                      (lm.y - wrist.y) / (inter_ocular_dist + 1e-6),
                                      (lm.z - wrist.z) / (inter_ocular_dist + 1e-6)] 
                                     for lm in results.right_hand_landmarks.landmark]).flatten()
            # Pose
            pose_feats = np.zeros(36, dtype=np.float32)
            if results.pose_landmarks:
                sh_l = results.pose_landmarks.landmark[11]
                sh_r = results.pose_landmarks.landmark[12]
                sh_cx = (sh_l.x + sh_r.x) / 2.0
                sh_cy = (sh_l.y + sh_r.y) / 2.0
                sh_cz = (sh_l.z + sh_r.z) / 2.0
                pose_feats = np.array([[(lm.x - sh_cx) / (inter_ocular_dist + 1e-6),
                                        (lm.y - sh_cy) / (inter_ocular_dist + 1e-6),
                                        (lm.z - sh_cz) / (inter_ocular_dist + 1e-6)]
                                       for idx in config.POSE_INDICES 
                                       for lm in [results.pose_landmarks.landmark[idx]]]).flatten()
        else:
            # TASKS API FALLBACK
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            face_result = self.face_landmarker.detect(mp_image)
            if not face_result.face_landmarks:
                return features_flat, cues
                
            # Face Tasks API landmarks
            face_lms_obj = face_result.face_landmarks[0]
            face_lms = np.array([[lm.x, lm.y, lm.z] for lm in face_lms_obj])
            
            # If the base face landmarker model has 468 landmarks but not the refined iris landmarks,
            # pad them or calculate relative center.
            # In MediaPipe Tasks FaceLandmarker, it detects 478 landmarks if refine_landmarks is active.
            nose_tip = face_lms[1]
            centered_face = face_lms - nose_tip
            inter_ocular_dist = np.linalg.norm(face_lms[33] - face_lms[263])
            
            if inter_ocular_dist > 1e-6:
                normalized_face = centered_face / inter_ocular_dist
            else:
                normalized_face = centered_face
                
            cues, hc_feats = self._extract_cues_and_hc_feats(face_lms, inter_ocular_dist, width, height)
            face_feats = normalized_face[config.SELECTED_FACE_INDICES].flatten()
            
            # Detect Hands
            hand_result = self.hand_landmarker.detect(mp_image)
            lh_feats = np.zeros(63, dtype=np.float32)
            rh_feats = np.zeros(63, dtype=np.float32)
            
            if hand_result.hand_landmarks:
                for i, handedness_list in enumerate(hand_result.handedness):
                    hand_label = handedness_list[0].category_name  # "Left" or "Right"
                    hlms = hand_result.hand_landmarks[i]
                    wrist = hlms[0]
                    hand_arr = np.array([[(lm.x - wrist.x) / (inter_ocular_dist + 1e-6),
                                          (lm.y - wrist.y) / (inter_ocular_dist + 1e-6),
                                          (lm.z - wrist.z) / (inter_ocular_dist + 1e-6)] 
                                         for lm in hlms]).flatten()
                    if hand_label == "Left":
                        lh_feats = hand_arr
                    else:
                        rh_feats = hand_arr
                        
            # Detect Pose
            pose_result = self.pose_landmarker.detect(mp_image)
            pose_feats = np.zeros(36, dtype=np.float32)
            if pose_result.pose_landmarks:
                plms = pose_result.pose_landmarks[0]
                sh_l = plms[11]
                sh_r = plms[12]
                sh_cx = (sh_l.x + sh_r.x) / 2.0
                sh_cy = (sh_l.y + sh_r.y) / 2.0
                sh_cz = (sh_l.z + sh_r.z) / 2.0
                pose_feats = np.array([[(plms[idx].x - sh_cx) / (inter_ocular_dist + 1e-6),
                                        (plms[idx].y - sh_cy) / (inter_ocular_dist + 1e-6),
                                        (plms[idx].z - sh_cz) / (inter_ocular_dist + 1e-6)]
                                       for idx in config.POSE_INDICES]).flatten()
                                       
        features_flat = np.concatenate([hc_feats, face_feats, lh_feats, rh_feats, pose_feats])
        return features_flat, cues

if __name__ == "__main__":
    print("Testing FeatureExtractor on a dummy frame...")
    extractor = FeatureExtractor(static_image_mode=True)
    
    # Try drawing a dummy face
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (320, 240), 100, (255, 255, 255), -1)
    
    feats, cues = extractor.extract_features(dummy_img)
    print("Feature shape:", feats.shape)
    print("Custom Cues:", cues)
    
    extractor.close()
    print("Test passed successfully!")
