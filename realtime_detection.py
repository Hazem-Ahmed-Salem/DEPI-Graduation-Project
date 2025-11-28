import os
import argparse
import pickle
import time
from typing import Dict
import cv2
import numpy as np
import mediapipe as mp

# ---------------- Config (defaults) ----------------
DEFAULT_MODEL_PICKLE = "hand_gesture_norm_model.pkl"
DEFAULT_SCALER_PICKLE = "scaler.pkl"
CAMERA_ID = 0
QUEUE_LENGTH = 5
CONF_THRESH = 0.60  # threshold to consider confident
ORIGIN_IDX = 0
SCALE_METHOD = "max_dist"
APPLY_ROTATION = True
MIRROR_LEFT = False  # Set True only if you mirrored left-hand during training

# ---------------- Helpers ----------------
def load_resources(model_path: str = DEFAULT_MODEL_PICKLE, scaler_path: str = DEFAULT_SCALER_PICKLE):
    model = None
    le = None
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, tuple) and len(obj) >= 2:
                model, le = obj[0], obj[1]
            elif isinstance(obj, dict) and "model" in obj:
                model = obj["model"]
                le = obj.get("le") or obj.get("label_encoder")
            else:
                model = obj
        except Exception as e:
            print(f"pickle load failed for {model_path}: {e}")
            model = None

    scaler = None
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Failed to load scaler ({scaler_path}): {e}")

    return model, le, scaler

def normalize_landmarks_flat(pts_flat, origin_idx=ORIGIN_IDX, scale_method=SCALE_METHOD, rotate=APPLY_ROTATION, eps=1e-8):
    pts = np.asarray(pts_flat, dtype=np.float32).reshape(-1)
    F = pts.size
    if F == 63:
        C = 3
    elif F == 42:
        C = 2
    else:
        return pts
    pts = pts.reshape(21, C)
    origin = pts[origin_idx].copy()
    pts = pts - origin
    if scale_method == "max_dist":
        d = np.linalg.norm(pts, axis=1)
        s = d.max()
    elif scale_method == "wrist_to_middle":
        s = np.linalg.norm(pts[9])
    elif scale_method == "bbox":
        s = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    else:
        s = 1.0
    s = max(s, eps)
    pts = pts / s
    if rotate and C >= 2:
        v = pts[9, :2]
        angle = np.arctan2(v[0], v[1])
        c = np.cos(-angle); sn = np.sin(-angle)
        R = np.array([[c, -sn], [sn, c]], dtype=np.float32)
        pts[:, :2] = pts[:, :2] @ R.T
    return pts.reshape(-1)

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def smooth_prediction(pred_vector, hand_id, history, queue_len=QUEUE_LENGTH):
    if hand_id not in history:
        history[hand_id] = []
    history[hand_id].append(pred_vector)
    if len(history[hand_id]) > queue_len:
        history[hand_id].pop(0)
    return np.mean(history[hand_id], axis=0)

# ---------------- Main realtime predictor ----------------
def main(args):
    model, le, scaler = load_resources(args.model, args.scaler)
    if model is None:
        raise RuntimeError("Model not found.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id={args.camera}")

    prediction_history: Dict[str, list] = {}

    try:
        print("Starting camera. Press ESC to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                    side = "Right"
                    if hasattr(results, "multi_handedness") and results.multi_handedness:
                        try:
                            side = results.multi_handedness[i].classification[0].label
                        except Exception:
                            side = "Right"

                    hand_id = f"{side}_{i}"

                    pts = []
                    for lm in hand_landmarks.landmark:
                        pts.extend([lm.x, lm.y, lm.z])
                    pts = np.array(pts, dtype=np.float32)
                    pts_norm = normalize_landmarks_flat(pts)

                    if MIRROR_LEFT and side.lower() == "left":
                        C = 3 if pts_norm.size == 63 else 2
                        tmp = pts_norm.reshape(21, C)
                        tmp[:, 0] *= -1.0
                        pts_norm = tmp.reshape(-1)

                    inp = pts_norm.reshape(1, -1)
                    if scaler is not None:
                        try:
                            inp = scaler.transform(inp)
                        except Exception:
                            pass

                    # Predict
                    try:
                        if hasattr(model, "predict"):
                            raw = model.predict(inp)
                        else:
                            raw = np.array([model.predict(inp)])
                    except Exception as e:
                        print(f"Prediction failed: {e}")
                        continue

                    pv = np.array(raw[0], dtype=float)
                    if not np.isclose(pv.sum(), 1.0, atol=1e-3):
                        pv = softmax(pv)

                    smooth = smooth_prediction(pv, hand_id, prediction_history)
                    pred_idx = int(np.argmax(smooth))
                    conf_pct = float(smooth[pred_idx] * 100.0)

                    try:
                        gesture = le.inverse_transform([pred_idx])[0] if le else str(pred_idx)
                    except Exception:
                        gesture = str(pred_idx)

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Show prediction on frame
                    x_pos = 10 if side == "Right" else frame.shape[1] - 300
                    cv2.putText(frame, f"{side}: {gesture} ({conf_pct:.1f}%)", (x_pos, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            else:
                prediction_history.clear()

            cv2.imshow("Hand Gesture Recognition (ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Exited.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Hand Gesture Predictor")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PICKLE, help="Path to model pickle")
    parser.add_argument("--scaler", type=str, default=DEFAULT_SCALER_PICKLE, help="Path to scaler pickle")
    parser.add_argument("--camera", type=int, default=CAMERA_ID, help="Camera ID for cv2.VideoCapture")
    args = parser.parse_args()
    main(args)
