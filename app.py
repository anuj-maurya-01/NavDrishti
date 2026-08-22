import os
import sys
import base64
from collections import deque
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

# Ensure config and ml modules are importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import config
from ml.inference import ISLInferenceManager

app = Flask(__name__, static_folder="frontend", static_url_path="")

# Global variables
inference_manager = None
recent_predictions = deque(maxlen=8)
latest_session_state = {
    "prediction": None,
    "confidence": 0.0,
    "top_predictions": [],
    "sentence_buffer": [],
    "recent_predictions": [],
    "non_manual_features": {},
    "attention": {},
    "tts_text": ""
}

def build_attention_summary(attention_distribution):
    if not attention_distribution:
        return {}

    # Attention can retain singleton batch/head dimensions; frame ranking needs one value per frame.
    arr = np.asarray(attention_distribution, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {}

    top_indices = np.argsort(arr)[-3:][::-1]
    return {
        "top_frames": [int(idx) for idx in top_indices.tolist()],
        "top_weights": [float(arr[idx]) for idx in top_indices.tolist()]
    }

def format_prediction_text(prediction):
    return prediction.upper() if isinstance(prediction, str) and prediction else "No sign detected yet"

def build_explanation(context):
    prediction = context.get("prediction")
    confidence = float(context.get("confidence") or 0.0)
    features = context.get("non_manual_features") or {}
    cues = []

    eyebrow_left = features.get("eyebrow_left")
    eyebrow_right = features.get("eyebrow_right")
    if eyebrow_left and eyebrow_left != "normal":
        cues.append(f"left eyebrow was {eyebrow_left}")
    if eyebrow_right and eyebrow_right != "normal":
        cues.append(f"right eyebrow was {eyebrow_right}")

    eyes = features.get("eyes")
    if eyes and eyes != "open":
        cues.append(f"eyes appeared {eyes}")

    mouth = features.get("mouth")
    if mouth and mouth != "closed":
        cues.append(f"mouth was {mouth}")

    head = features.get("head")
    if head and head != "center":
        cues.append(f"head orientation shifted {head}")

    confidence_cues = features.get("confidence_cues") or {}
    if not cues and confidence_cues:
        if abs(confidence_cues.get("yaw", 0.0)) > 10:
            cues.append("head yaw changed noticeably")
        if abs(confidence_cues.get("pitch", 0.0)) > 10:
            cues.append("head pitch changed noticeably")
        if abs(confidence_cues.get("roll", 0.0)) > 8:
            cues.append("head roll changed noticeably")

    attention = context.get("attention") or {}
    attention_text = ""
    top_frames = attention.get("top_frames") or []
    if top_frames:
        frames_str = ", ".join(str(frame_idx) for frame_idx in top_frames)
        attention_text = f" Important frames were around steps {frames_str}."

    if not prediction:
        return "I'm waiting for an ISL sign. Start the camera and perform a sign."

    if confidence < config.CONFIDENCE_THRESHOLD:
        return "I'm not fully confident about this prediction. Please hold the sign slightly longer or try again."

    if cues:
        cues_text = ", ".join(cues)
        return f"The model predicted {prediction.upper()} because {cues_text}.{attention_text}"

    return f"The model predicted {prediction.upper()} using the current facial, pose, and temporal cues available in this session.{attention_text}"

def build_chat_response(mode, message, context):
    prediction = context.get("prediction")
    confidence = float(context.get("confidence") or 0.0)
    sentence_buffer = context.get("sentence_buffer") or []
    top_predictions = context.get("top_predictions") or []
    features = context.get("non_manual_features") or {}
    tts_text = context.get("tts_text") or " ".join(sentence_buffer).strip() or (prediction or "")
    active_text = " ".join(sentence_buffer).strip() or prediction

    if not active_text:
        response = "I'm waiting for an ISL sign. Start the camera and perform a sign."
        return {"response": response, "mode": mode, "speak_text": "", "confidence": confidence}

    if mode == "translate":
        response = f"I detected: {format_prediction_text(active_text)}."
        if confidence < config.CONFIDENCE_THRESHOLD:
            response += " I'm not fully confident about this prediction. Please hold the sign slightly longer or try again."
        return {"response": response, "mode": mode, "speak_text": tts_text, "confidence": confidence}

    if mode == "speak":
        response = f"Ready to speak: {format_prediction_text(tts_text)}." if tts_text else "I don't have enough information from the current recognition session to answer that."
        return {"response": response, "mode": mode, "speak_text": tts_text, "confidence": confidence}

    if mode == "explain":
        response = build_explanation(context)
        return {"response": response, "mode": mode, "speak_text": "", "confidence": confidence}

    message_lower = (message or "").strip().lower()
    if "what did i just sign" in message_lower or "what did i sign" in message_lower:
        response = f"You signed: {format_prediction_text(active_text)}."
    elif "what does that mean" in message_lower or "what does it mean" in message_lower:
        response = f"It means: {active_text}."
    elif "confidence" in message_lower:
        response = f"The current prediction confidence is {round(confidence * 100)}%."
    elif "top" in message_lower or "alternative" in message_lower:
        if top_predictions:
            alternatives = ", ".join(
                f"{item['label'].upper()} ({round(item['confidence'] * 100)}%)"
                for item in top_predictions[1:]
            )
            response = f"Top alternatives after the current prediction are: {alternatives}." if alternatives else "There are no additional alternatives available right now."
        else:
            response = "I don't have enough information from the current recognition session to answer that."
    elif "explain" in message_lower or "why" in message_lower:
        response = build_explanation(context)
    elif "cues" in message_lower or "feature" in message_lower:
        cue_parts = []
        for key in ["eyebrow_left", "eyebrow_right", "eyes", "mouth", "head"]:
            value = features.get(key)
            if value:
                cue_parts.append(f"{key.replace('_', ' ')}: {value}")
        response = "Current non-manual cues are " + ", ".join(cue_parts) + "." if cue_parts else "I don't have enough information from the current recognition session to answer that."
    else:
        response = f"Using the current recognition context, I detected: {format_prediction_text(active_text)}."

    return {"response": response, "mode": mode, "speak_text": tts_text, "confidence": confidence}

def init_model():
    global inference_manager
    if inference_manager is None:
        try:
            # Check if model exists before loading
            model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
            if not os.path.exists(model_path):
                print(f"[Warning] Best model checkpoint not found at '{model_path}'. Running backend in placeholder/health mode.")
                return False
                
            inference_manager = ISLInferenceManager(model_type="temporal", model_path=model_path)
            print("[Backend] Inference model loaded successfully.")
            return True
        except Exception as e:
            print(f"[Error] Failed to initialize model: {e}")
            return False

# Serve Frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Serve Style/Scripts
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/health', methods=['GET'])
def health():
    model_loaded = inference_manager is not None
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "classes_configured": config.NUM_CLASSES,
        "input_dim_configured": config.INPUT_DIM
    }), 200

@app.route('/reset', methods=['POST'])
def reset():
    global inference_manager
    if inference_manager is not None:
        inference_manager.reset_buffer()
        return jsonify({"status": "success", "message": "Buffer reset successfully."}), 200
    return jsonify({"status": "error", "message": "Model not loaded."}), 400

@app.route('/predict', methods=['POST'])
def predict():
    global inference_manager
    global latest_session_state
    
    # Lazy initialization if not initialized yet
    if inference_manager is None:
        success = init_model()
        if not success:
            return jsonify({
                "prediction": "MODEL NOT TRAINED",
                "confidence": 0.0,
                "top_predictions": [],
                "attention": {},
                "features": {
                    "eyebrow_left": "unknown",
                    "eyebrow_right": "unknown",
                    "eyes": "unknown",
                    "mouth": "unknown",
                    "head": "unknown",
                    "confidence_cues": {}
                },
                "error": "Trained weights 'best_model.pth' missing from models/ folder. Download them from Colab."
            }), 200
            
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"status": "error", "message": "Missing image data in request"}), 400
            
        base64_image = data['image']
        
        # Decode base64 image
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]
            
        img_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"status": "error", "message": "Failed to decode frame image"}), 400
            
        # Run inference
        pred_label, pred_conf, cues, top_predictions = inference_manager.process_frame(frame)
        attention_summary = build_attention_summary(cues.get("attention_distribution"))
        if pred_label:
            recent_predictions.append(pred_label)
        
        # Return result
        latest_session_state.update({
            "prediction": pred_label,
            "confidence": float(pred_conf),
            "top_predictions": top_predictions,
            "recent_predictions": list(recent_predictions),
            "non_manual_features": cues,
            "attention": attention_summary
        })

        return jsonify({
            "prediction": pred_label,
            "confidence": float(pred_conf),
            "top_predictions": top_predictions,
            "features": cues,
            "attention": attention_summary
        }), 200
        
    except Exception as e:
        print(f"[Error] Prediction endpoint error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    global latest_session_state

    try:
        data = request.get_json() or {}
        mode = (data.get("mode") or "conversation").strip().lower()
        message = data.get("message") or ""

        context = dict(latest_session_state)
        for key in ["prediction", "confidence", "top_predictions", "sentence_buffer", "recent_predictions", "non_manual_features", "attention", "tts_text"]:
            if key in data and data.get(key) is not None:
                context[key] = data.get(key)

        response = build_chat_response(mode, message, context)
        return jsonify(response), 200
    except Exception as e:
        print(f"[Error] Chat endpoint error: {e}")
        return jsonify({
            "response": "I don't have enough information from the current recognition session to answer that.",
            "mode": "conversation",
            "speak_text": "",
            "confidence": 0.0
        }), 500

if __name__ == '__main__':
    # Initialize the model on startup if possible
    init_model()
    # Run the server on port 5000
    app.run(host='127.0.0.1', port=5000, debug=False)
