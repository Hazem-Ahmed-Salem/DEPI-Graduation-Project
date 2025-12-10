import os
import argparse
import pickle
import time
from typing import Dict
import cv2
import numpy as np
import mediapipe as mp

# Preserve the original imports from your realtime controller
try:
    from actions import GestureActions
except Exception:
    GestureActions = None
    print("Couldn't import GestureActions from actions.py.")

try:
    from keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

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

def load_resources(model_path: str = DEFAULT_MODEL_PICKLE, scaler_path: str = DEFAULT_SCALER_PICKLE):
    """
    Loads model + label encoder and scaler.
    Accepts pickles in these formats:
      - (model, le)
      - {'model': model, 'le': le}
      - model only (then try to find label encoder separately or raise)
    """
    model = None
    le = None
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            # tuple (model, le)
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

def is_palm_facing_camera(landmarks, side="Right"):
    """
    Heuristic to check if palm is facing camera.
    
    Right Hand (Mirrored): Index Knuckle (5) should be to the LEFT of Pinky Knuckle (17) -> x5 < x17
    Left Hand (Mirrored): Index Knuckle (5) should be to the RIGHT of Pinky Knuckle (17) -> x5 > x17
    """
    # Landmarks: 5=Index MCP, 17=Pinky MCP
    index_mcp_x = landmarks[5].x
    pinky_mcp_x = landmarks[17].x
    
    if side == "Right":
        # In mirrored view, Right hand palm facing means Index is 'left' of Pinky
        return index_mcp_x < pinky_mcp_x
    else:
        # Left hand palm facing means Index is 'right' of Pinky
        return index_mcp_x > pinky_mcp_x

# ---------------- Overlay helper ----------------
def draw_overlay(frame, text_lines, position="top-left"):
    h, w = frame.shape[:2]
    padding = 8
    line_h = 20
    box_w = 360
    box_h = padding*2 + len(text_lines)*line_h
    if position == "top-left":
        x0, y0 = 10, 10
    elif position == "top-right":
        x0, y0 = w - box_w - 10, 10
    else:
        x0, y0 = 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    y = y0 + padding + 15
    for line in text_lines:
        cv2.putText(frame, line, (x0 + padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += line_h
    return frame

# ---------------- Main realtime controller ----------------
def main(args):
    # Load model, le, scaler using main.py style loader
    model_path = args.model
    scaler_path = args.scaler
    print(f"Loading model from {model_path}, scaler from {scaler_path}")
    model, le, scaler = load_resources(model_path, scaler_path)
    if model is None:
        raise RuntimeError("Model not found.")

    # instantiate actions (preserve original controller behavior)
    actions = None
    if GestureActions is not None:
        try:
            actions = GestureActions()
        except Exception as e:
            print(f"GestureActions() instantiation failed: {e}")
            actions = None
    else:
        print("Actions.GestureActions is not available.")

    # mediapipe initialization
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id={args.camera}")

    prediction_history: Dict[str, list] = {}

    # State for gesture timing logic
    gesture_state = {
        "current": None,
        "start_time": 0.0,
        "last_action": 0.0,
        "triggered_once": False
    }

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

            overlay_text = []

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:1]):
                    # determine handedness by location heuristics or media pipe handedness container
                    # If the original file used results.multi_handedness, keep that; otherwise fall back to index
                    side = "Right"  # default
                    if hasattr(results, "multi_handedness") and results.multi_handedness:
                        try:
                            side = results.multi_handedness[i].classification[0].label
                        except Exception:
                            side = "Right"

                    hand_id = f"{side}_{i}"

                    # collect landmarks into flat list x,y,z for 21 points
                    pts = []
                    for lm in hand_landmarks.landmark:
                        pts.extend([lm.x, lm.y, lm.z])
                    pts = np.array(pts, dtype=np.float32)

                    # normalize the points
                    pts_norm = normalize_landmarks_flat(pts)

                    # optional mirror if you used mirrored left-hand training
                    if MIRROR_LEFT and side.lower() == "left":
                        C = 3 if pts_norm.size == 63 else 2
                        tmp = pts_norm.reshape(21, C)
                        tmp[:, 0] *= -1.0
                        pts_norm = tmp.reshape(-1)

                    inp = pts_norm.reshape(1, -1)
                    if scaler is not None:
                        try:
                            inp = scaler.transform(inp)
                        except Exception as e:
                            print(f"Scaler transform failed: {e}")

                    # Predict
                    try:
                        # Keras models often return probability vector directly
                        raw = None
                        if hasattr(model, "predict"):
                            raw = model.predict(inp)
                        else:
                            # sklearn-like
                            raw = np.array([model.predict_proba(inp)[0]]) if hasattr(model, "predict_proba") else np.array([model.predict(inp)])
                    except Exception as e:
                        print(f"Prediction failed: {e}")
                        continue

                    pv = np.array(raw[0], dtype=float)
                    # Ensure probabilities
                    if not np.isclose(pv.sum(), 1.0, atol=1e-3):
                        pv = softmax(pv)

                    # Smoothing across frames
                    smooth = smooth_prediction(pv, hand_id, prediction_history, queue_len=QUEUE_LENGTH)

                    pred_idx = int(np.argmax(smooth))
                    conf_pct = float(smooth[pred_idx] * 100.0)

                    # Decode label using label encoder
                    try:
                        if le is not None:
                            gesture = le.inverse_transform([pred_idx])[0]
                        else:
                            gesture = str(pred_idx)
                    except Exception:
                        gesture = str(pred_idx)

                    # Draw landmarks from mediapipe
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Display main detection text
                    x_pos = 10 if side == "Right" else frame.shape[1] - 300
                    if conf_pct >= CONF_THRESH * 100:
                        cv2.putText(frame, f"{gesture} ({conf_pct:.1f}%)", (x_pos, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"{gesture} —", (x_pos, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

                    # Top-3 predictions overlay
                    sorted_idx = np.argsort(smooth)[-3:][::-1]
                    y_offset = 80
                    for pred_i in sorted_idx:
                        conf = smooth[pred_i] * 100
                        if conf > 1.0:  # show small confidences
                            try:
                                label_name = le.inverse_transform([pred_i])[0] if le is not None else str(pred_i)
                            except Exception:
                                label_name = str(pred_i)
                            cv2.putText(frame, f"{label_name}: {conf:.1f}%", (x_pos, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            y_offset += 20

                    # ---------------- Preserved controller logic: mapping gestures to actions ----------------
                    if actions is not None:
                        try:
                            # read index tip coords to pass to cursor movement if needed
                            index_tip = hand_landmarks.landmark[8]
                            index_x, index_y = index_tip.x, index_tip.y
                        except Exception:
                            index_x, index_y = None, None

                        # Determine active gesture based on confidence
                        # Using 90% threshold (CONF_THRESH) to filter weak detections
                        # AND check if palm is facing camera (unless it's a fist)
                        palm_facing = is_palm_facing_camera(hand_landmarks.landmark, side)
                        is_fist = gesture in ["03_fist", "04_fist_moved", "05_thumb", "08_palm_moved", "10_down", "06_index", "09_c", "02_l", "07_ok", "01_palm"]
                        
                        if conf_pct > 90 and (palm_facing or is_fist):
                            active_gesture = gesture
                        else:
                            active_gesture = None

                        # State Management
                        if active_gesture != gesture_state["current"]:
                            gesture_state["current"] = active_gesture
                            gesture_state["start_time"] = time.time()
                            gesture_state["last_action"] = 0.0
                            gesture_state["triggered_once"] = False

                        if active_gesture:
                            duration = time.time() - gesture_state["start_time"]

                            # 1. Instant Actions (Cursor) - No delay
                            if active_gesture == "06_index":
                                try:
                                    actions.move_cursor(index_x, index_y)
                                except Exception as e:
                                    print(f"Actions.move_cursor failed: {e}")


                            # 2. Delayed/Repeated Actions
                            else:
                                should_execute = False

                                # Phase 1: Initial Trigger after .8s
                                if duration >= .8 and not gesture_state["triggered_once"]:
                                    should_execute = True
                                    gesture_state["triggered_once"] = True
                                    gesture_state["last_action"] = time.time()

                                # Phase 2: Repeat if held > 2.5s, every 1.5s
                                elif duration >= 2.5:
                                    if time.time() - gesture_state["last_action"] >= 1.5:
                                        should_execute = True
                                        gesture_state["last_action"] = time.time()

                                if should_execute:
                                    try:
                                        actions.perform_action(active_gesture, x=index_x, y=index_y)
                                    except Exception as e:
                                        print(f"[WARN] actions.perform_action failed: {e}")

                    # append to overlay
                    overlay_text.append(f"{side}: {gesture} ({conf_pct:.1f}%)")
            else:
                # No hands detected -> clear history
                prediction_history.clear()
                overlay_text.append("No hands detected")

            # draw overlay box with predictions
            frame = draw_overlay(frame, overlay_text, position="top-right")

            cv2.imshow("Multi-Hand Gesture Recognition (ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Exited.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Hand Gesture Controller")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PICKLE, help="Path to model pickle")
    parser.add_argument("--scaler", type=str, default=DEFAULT_SCALER_PICKLE, help="Path to scaler pickle")
    parser.add_argument("--camera", type=int, default=CAMERA_ID, help="Camera ID for cv2.VideoCapture")
    args = parser.parse_args()
    main(args)
